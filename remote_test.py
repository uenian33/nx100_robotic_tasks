import sys
sys.path.append('nx100_remote/') # add relative path
import numpy as np
import time
from module import Commands, Utils, Gripper
from objects import MoveL, IO

def get_position():
    return Commands.read_current_specified_coordinate_system_position('0', '0')


c_pos = get_position()
print(c_pos.x, c_pos.y, c_pos.z, c_pos.tx, c_pos.ty, c_pos.tz)
"""
Gripper.write_gripper_open()
time.sleep(2)
gripper_open=True
gp_status = Gripper.read_gripper_acknowledge()
print(gp_status.is_acknowledge, gp_status.is_closed)
while gripper_open:
    gripper_open = not Gripper.read_gripper_acknowledge()
    time.sleep(1)
#Gripper.write_gripper_open()
Gripper.write_gripper_close()

#"""
Commands.write_linear_move(MoveL.MoveL(
                1, 5, 0,
                c_pos.x+3,
                c_pos.y,
                c_pos.z, 
                c_pos.tx,
                c_pos.ty,
                c_pos.tz, 
                Utils.binary_to_decimal(0x00000001)
            ))#"""