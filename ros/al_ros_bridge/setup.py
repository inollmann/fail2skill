from glob import glob

from setuptools import setup

package_name = "al_ros_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "isaac_bridge = al_ros_bridge.bridge_node:main",
            "al_episode = al_ros_bridge.episode_cli:main",
        ],
    },
)
