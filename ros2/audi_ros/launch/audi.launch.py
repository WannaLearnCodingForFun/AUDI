from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        Node(package="audi_ros", executable="mission_node", parameters=[{"config": cfg}]),
        Node(package="audi_ros", executable="perception_node",
             parameters=[{"config": cfg, "image_topic": LaunchConfiguration("image_topic")}]),
        Node(package="audi_ros", executable="alert_bridge"),
    ])
