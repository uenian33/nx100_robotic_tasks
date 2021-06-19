import sys
sys.path.append('nx100_remote/') # add relative path
import numpy as np
import time
from module import Commands, Utils, Gripper
from objects import MoveL, IO
import time

def get_position():
    return Commands.read_current_specified_coordinate_system_position('0', '0')

#Gripper.write_gripper_open()
time.sleep(3)
c_pos = get_position()
print(c_pos.x, c_pos.y, c_pos.z, c_pos.tx, c_pos.ty, c_pos.tz)
gt_pose = [-68.848, 850.944, 259.794, 2.47, 36.4, 94.93]
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
                MoveL.MoveL.motion_speed_selection_post_speed, 45, MoveL.MoveL.coordinate_specification_base_coordinate,
                gt_pose[0],
                gt_pose[1],
                gt_pose[2], 
                gt_pose[3],
                gt_pose[4],
                gt_pose[5],
                Utils.binary_to_decimal(0x00000001)
            ))#"""
#time.sleep(5)

"""
c_pos = get_position()

Commands.write_linear_move(MoveL.MoveL(
                MoveL.MoveL.motion_speed_selection_posture_speed, 3, MoveL.MoveL.coordinate_specification_base_coordinate,
                c_pos.x,
                c_pos.y,
                c_pos.z,
                gt_pose[3],
                gt_pose[4],
                gt_pose[5],
                Utils.binary_to_decimal(0x00000001)
            ))

"""