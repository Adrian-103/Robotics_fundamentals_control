import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
# from launch.actions import ExecuteProcess

def generate_launch_description():

    #Motor controller node
    mpc_node = Node(
        package = 'position_control_turtlebot',
        executable = 'MPC_ControllerExec',
        name = 'mpc_controller_node',
        output = 'screen',
    )

    return LaunchDescription([
        mpc_node
    ])