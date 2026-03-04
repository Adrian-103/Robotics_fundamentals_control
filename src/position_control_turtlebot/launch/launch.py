import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
# from launch.actions import ExecuteProcess

def generate_launch_description():

    #Get YAML path:
    config = os.path.join(
        get_package_share_directory('position_control_turtlebot'),
        'config',
        'params.yaml'
    )

    #Motor controller node
    motor_node = Node(
        package = 'position_control_turtlebot',
        executable = 'TurtleBotControllerExec',
        name = 'turtlebot_controller_node',
        output = 'screen',
        parameters = [config]
    )

    return LaunchDescription([
        motor_node,
    ])