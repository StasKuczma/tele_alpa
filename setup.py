import os
from glob import glob

from setuptools import find_packages, setup

package_name = "tele_alpa"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Stanislaw Kuczma",
    maintainer_email="staskuczma@gmail.com",
    description="ROS 2 interface for Alpamayo 1.5 trajectory planning and VQA.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "alpamayo_node = tele_alpa.alpamayo_node:main",
            "vqa_node = tele_alpa.vqa_node:main",
            "ask = tele_alpa.ask:main",
        ],
    },
)
