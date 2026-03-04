import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/adrian/control_ws/src/position_control_turtlebot/install/position_control_turtlebot'
