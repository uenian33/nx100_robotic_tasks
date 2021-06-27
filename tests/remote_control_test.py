import sys
import numpy as np
import time
from nx100_remote_control.module import Commands, Utils
from nx100_remote_control.objects import MoveL

def callback_success():
    print('MoveL position has been reached')

def callback_failed():
    print('MoveL error or position not reached on given timeout')


def get_position():
    return Commands.read_current_specified_coordinate_system_position('0', '0')

#Gripper.write_gripper_open()
time.sleep(3)
c_pos = get_position()
print(c_pos.x, c_pos.y, c_pos.z, c_pos.tx, c_pos.ty, c_pos.tz)
gt_pose = [-68.848, 850.944, 259.794, 2.47, 36.4, 94.93]


move_l = MoveL.MoveL(
        MoveL.MoveL.motion_speed_selection_posture_speed,
        5,
        MoveL.MoveL.coordinate_specification_base_coordinate,
        gt_pose[0],
        gt_pose[1],
        gt_pose[2], 
        gt_pose[3],
        gt_pose[4],
        gt_pose[5],
        Utils.binary_to_decimal(0x00000001),
        0, 0, 0, 0, 0, 0, 0
    )
    
Commands.robot_in_target_point_callback(
    move_l=move_l, timeout=10, _callback_success=callback_success, _callback_failed=callback_failed
)

"""