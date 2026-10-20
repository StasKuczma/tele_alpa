# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Many trajectories from few VLM rollouts.

Upstream ``Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout[_cfg_nav]``
implements ``num_traj_samples=N`` as ``generation_config.num_return_sequences=N``:
HF ``generate`` copies the whole prompt (pixel_values included) N times, so the
vision encoder, the prefill and the chain-of-causation decode all run on a batch
of N, and every trajectory gets its own reasoning rollout. The flow-matching
action expert — the cheap part — then denoises once per rollout.

Here the VLM runs ``num_rollouts`` (R) times — typically once — and its KV cache,
rope deltas and EOS offsets are repeated ``samples_per_rollout`` (K) times. The
action expert then denoises R*K different noise seeds in one batch. Result: R*K
trajectories for roughly the cost of R rollouts plus one batched denoise.

The K samples of one rollout share its CoT text, so they differ only through
diffusion noise. That gives less spread than K independent rollouts, which can
reason their way to different manoeuvres.

Mirrors the upstream/``alpamayo-autoware`` sampling code (same helpers, same
expert-step runner / observer / CUDA-graph hooks when the model has them) so the
installed ``alpamayo1_5`` package stays untouched.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

import einops
import numpy as np
import torch
from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_5.nav_utils import remove_nav_text
from transformers import LogitsProcessorList, StoppingCriteriaList


def _repeat_rows(t: torch.Tensor, n_rows: int) -> torch.Tensor:
    """repeat_interleave dim 0 of ``t`` up to ``n_rows`` (no-op if already there)."""
    if t.shape[0] == n_rows:
        return t
    return t.repeat_interleave(n_rows // t.shape[0], dim=0)


@torch.no_grad()
def sample_trajectories_shared_rollout(
    model: Any,
    data: dict[str, Any],
    *,
    num_rollouts: int = 1,
    samples_per_rollout: int = 10,
    nav_cfg: bool = False,
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    max_generation_length: int | None = None,
    diffusion_kwargs: dict[str, Any] | None = None,
    return_extra: bool = False,
):
    """Drop-in for ``model.sample_trajectories_from_data_with_vlm_rollout[_cfg_nav]``.

    Returns ``pred_xyz [B, R, K, T, 3]``, ``pred_rot [B, R, K, T, 3, 3]`` and, with
    ``return_extra``, text spans shaped ``[B, R, K]`` (constant along K).
    ``nav_cfg=True`` applies classifier-free guidance against a prompt with the
    ``<|route_start|>..<|route_end|>`` span removed, as the ``_cfg_nav`` variant.
    The unguided cache is only built when the diffusion actually uses guidance
    (``use_classifier_free_guidance``, off in the released config); upstream
    always builds it, which costs a second image encode + prefill for nothing.
    """
    R, K = int(num_rollouts), int(samples_per_rollout)
    diffusion_kwargs = dict(diffusion_kwargs or {})
    use_cfg = diffusion_kwargs.get("use_classifier_free_guidance")
    if use_cfg is None:
        use_cfg = model.diffusion.use_classifier_free_guidance
    nav_cfg = nav_cfg and bool(use_cfg)
    tokenized_data = dict(data["tokenized_data"])  # shallow: don't clone pixel_values
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported for inference."
    input_ids = model.fuse_traj_tokens(
        tokenized_data.pop("input_ids"),
        {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
    )
    device = input_ids.device
    b_star = B * R * K

    # 1) VLM: images + prefill + CoT decode, R rollouts only.
    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = R
    generation_config.max_new_tokens = max_generation_length or model.config.tokens_per_future_traj
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
        logits_processor=LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        ),
        **tokenized_data,
    )
    rope_deltas = model.vlm.model.rope_deltas
    sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )  # [B*R, L]
    prompt_cache = vlm_outputs.past_key_values

    # 2) (nav CFG) unguided cache: prefill once, repeat to R, replay the R CoTs.
    if nav_cfg:
        unguided_input_ids = torch.nn.utils.rnn.pad_sequence(
            [remove_nav_text(input_ids, model.tokenizer, i)[0] for i in range(B)],
            batch_first=True,
            padding_value=model.tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        unguided_prefix_mask = unguided_input_ids.ne(model.tokenizer.pad_token_id).long()
        unguided_prefill = model.vlm(
            input_ids=unguided_input_ids,
            attention_mask=unguided_prefix_mask,
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            pixel_values=tokenized_data.get("pixel_values"),
            use_cache=True,
            logits_to_keep=1,
        )
        unguided_cache = unguided_prefill.past_key_values
        del unguided_prefill
        unguided_cache.batch_repeat_interleave(R)

        generated_tokens = sequences[:, input_ids.shape[1] :]
        unguided_prefix_len = unguided_input_ids.shape[1]
        unguided_out = model.vlm(
            input_ids=generated_tokens,
            attention_mask=torch.cat(
                [
                    unguided_prefix_mask.repeat_interleave(R, dim=0),
                    generated_tokens.ne(model.tokenizer.pad_token_id).long(),
                ],
                dim=1,
            ),
            past_key_values=unguided_cache,
            cache_position=torch.arange(
                unguided_prefix_len,
                unguided_prefix_len + generated_tokens.shape[1],
                device=device,
                dtype=torch.long,
            ),
            use_cache=True,
            logits_to_keep=1,
        )
        unguided_cache = unguided_out.past_key_values
        unguided_rope_deltas = unguided_out.rope_deltas
        unguided_sequences = torch.cat(
            [unguided_input_ids.repeat_interleave(R, dim=0), generated_tokens], dim=1
        )
        del unguided_out
        torch.cuda.empty_cache()

    # 3) Expand every per-rollout conditioning tensor from B*R to B*R*K rows.
    #    This is the only place N shows up before the action expert.
    def expert_conditioning(cache, seqs, deltas, prefix_mask, warn):
        if K > 1:
            cache.batch_repeat_interleave(K)
        seqs = seqs.repeat_interleave(K, dim=0)
        offset = model._find_eos_offset(
            sequences=seqs, eos_token_id=eos_token_id, device=device, warn=warn
        )
        position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=_repeat_rows(deltas, b_star),
            kv_cache_seq_len=cache.get_seq_length(),
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=None if prefix_mask is None else _repeat_rows(prefix_mask, b_star),
        )
        return cache, position_ids, attention_mask

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    guided = expert_conditioning(
        prompt_cache, sequences, rope_deltas, tokenized_data.get("attention_mask"), warn=True
    )
    unguided = (
        expert_conditioning(
            unguided_cache,
            unguided_sequences,
            unguided_rope_deltas,
            unguided_prefix_mask,
            warn=False,
        )
        if nav_cfg
        else None
    )

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    step_runner = getattr(model, "_expert_step_runner", None)
    step_observer = getattr(model, "_expert_step_observer", None)

    # 4) One denoising step of the action expert over the (expanded) cache.
    def step_fn(x, t, past_key_values, position_ids, attention_mask, runner_context):
        if step_observer is not None:
            step_observer(
                {
                    "x": x,
                    "t": t,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "prompt_cache": past_key_values,
                }
            )
        if runner_context is not None:
            return step_runner.step(x=x, t=t, context=runner_context)
        future_token_embeds = model.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        prefill_seq_len = past_key_values.get_seq_length()
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        past_key_values.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(
            -1, *model.action_space.get_action_space_dims()
        )

    def bind(conditioning):
        cache, position_ids, attention_mask = conditioning
        runner_context = (
            step_runner.prepare_context(
                prompt_cache=cache, position_ids=position_ids, attention_mask=attention_mask
            )
            if step_runner is not None
            else None
        )
        return partial(
            step_fn,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            runner_context=runner_context,
        )

    cuda_graph = getattr(model, "_diffusion_expert_cuda_graph", None)
    with cuda_graph.sampling() if cuda_graph is not None else nullcontext():
        sampled_action = model.diffusion.sample(
            batch_size=b_star,
            step_fn=bind(guided),
            unguided_step_fn=bind(unguided) if unguided is not None else None,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

    # 5) Actions -> trajectories, laid out (b r k) like upstream's (b ns nj).
    pred_xyz, pred_rot = model.action_space.action_to_traj(
        sampled_action,
        einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=R * K),
        einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=R * K),
    )
    pred_xyz = einops.rearrange(pred_xyz, "(b r k) ... -> b r k ...", r=R, k=K)
    pred_rot = einops.rearrange(pred_rot, "(b r k) ... -> b r k ...", r=R, k=K)

    if not return_extra:
        return pred_xyz, pred_rot
    extra = extract_text_tokens(model.tokenizer, sequences)
    for key in extra:
        extra[key] = np.repeat(np.array(extra[key]).reshape(B, R, 1), K, axis=2)
    return pred_xyz, pred_rot, extra
