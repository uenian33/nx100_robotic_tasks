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

# Todo, define proper starting point
starting_point = [-70.888, 836.813, 281.496, 0.21, 36.59, 90.14]
# Todo, define joint move position
# joint_position_1 = [123, 123, 123, 20, 20, 10]
# joint_position_2 = [123, 123, 123, 20, 20, 10]


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
        5,
        MoveL.MoveL.coordinate_specification_base_coordinate,
        target[0], target[1], target[2],
        target[3], target[4], target[5],
        Utils.binary_to_decimal(0x00000001),
        0, 0, 0, 0, 0, 0, 0
    )
    linear_move = LinearMove.LinearMove()
    linear_move.go(move_l=move_l, wait=True, poll_limit_seconds=10)


def joint_move(target):
    move_j = MoveJ.MoveJ(
        10,  # speed % todo, set higher when feeling confident
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
# linear_move(starting_point)

# Todo, move robot to point B with fast MOVJ
# joint_move(joint_position_1)
# joint_move(joint_position_2)

# Gripper.write_gripper_open()
# time.sleep(5)

# back to start point
# joint_move(starting_point)
