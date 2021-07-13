from libs.grasp_estimation.inference.grasp_generator import GraspGenerator
from libs.camera.custom_camera import CustomCamera
from libs.camera.camera_data import CameraData
from nx100_remote_control.module import Commands, LinearMove, JointMove, Gripper, Utils
from nx100_remote_control.objects import MoveL, MoveJ
import nx100_remote_control
import time
import os
import sys

sys.path.append('libs/grasp_estimation')

# Starting point (uses camera here to see what to grasp)
starting_point = [-70.888, 836.813, 281.496, 0.21, 36.59, 90.14]
# Table points
top_of_table_1 = [828.622,5.405,307.428,0.43,34.43,-4.20,0,0]  # move fast to this point
top_of_table_2 = [593.274,-36.586,-2.323,-3.79,36.29,-8.03,0,0]  # move slower here


if __name__ == '__main__':
    # init camera setups
    camera = CustomCamera()
    cam_data = CameraData(include_depth=True, include_rgb=True)

    # init robot
    nx100_remote_control.NX100_IP_ADDRESS = '192.168.2.28'
    nx100_remote_control.NX100_TCP_PORT = 80
    nx100_remote_control.MOCK_RESPONSE = False


def linear_move(target):
    move_l = MoveL.MoveL(
        MoveL.MoveL.motion_speed_selection_posture_speed,
        2,
        MoveL.MoveL.coordinate_specification_base_coordinate,
        target[0], target[1], target[2],
        target[3], target[4], target[5],
        Utils.binary_to_decimal(0x00000001),
        0, 0, 0, 0, 0, 0, 0
    )
    linear_move = LinearMove.LinearMove()
    linear_move.go(move_l=move_l, wait=True, poll_limit_seconds=10)


def joint_move(target, speed=5):
    move_j = MoveJ.MoveJ(
        speed,  # speed % todo, set higher when feeling confident
        MoveJ.MoveJ.coordinate_specification_base_coordinate,
        target[0], target[1], target[2],
        target[3], target[4], target[5],
        Utils.binary_to_decimal(0x00000001),
        0, 0, 0, 0, 0, 0, 0
    )
    joint_move = JointMove.JointMove()
    joint_move.go(move_j=move_j, wait=True, poll_limit_seconds=10)


Gripper.write_gripper_close()  # actually opens..

# !!! starting point move with movj !!!
joint_move(starting_point)

# init grasping
generator = GraspGenerator(
    camera,
    cam_data,
    cam_id=830112070066,
    saved_model_path=os.getcwd() + "/libs/grasp_estimation/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_13_iou_0.96",
    visualize=True
)
generator.load_model()
generator.camera.stream()
target_robot_pose = generator.generate()

linear_move(target_robot_pose)
Gripper.write_gripper_open()  # actually closes
time.sleep(5)

# Todo, move here slowly bit up before executing next MOVJ (starting point maybe?)
joint_move(starting_point)

# Todo, move robot to point B with fast MOVJ
joint_move(top_of_table_1, 20)
joint_move(top_of_table_2)

Gripper.write_gripper_close()
time.sleep(5)

# back to start point
joint_move(top_of_table_1)
joint_move(starting_point, 20)
