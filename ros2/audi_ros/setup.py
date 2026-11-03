from setuptools import setup

setup(
    name="audi_ros",
    version="0.1.0",
    packages=["audi_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/audi_ros"]),
        ("share/audi_ros", ["package.xml"]),
        ("share/audi_ros/launch", ["launch/audi.launch.py", "launch/replay_test.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "perception_node = audi_ros.perception_node:main",
        "mission_node = audi_ros.mission_node:main",
        "alert_bridge = audi_ros.alert_bridge:main",
    ]},
)
