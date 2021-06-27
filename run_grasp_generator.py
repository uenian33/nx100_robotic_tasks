from libs.grasp_estimation.inference.grasp_generator import GraspGenerator
from libs.camera.custom_camera import CustomCamera
from libs.camera.camera_data import CameraData


from nx100_remote_control.module import Commands, Utils
from nx100_remote_control.objects import MoveL

def callback_success():
    print('MoveL position has been reached')

def callback_failed():
    print('MoveL error or position not reached on given timeout')




if __name__ == '__main__':
	# init camera setups
	camera = CustomCamera()
	cam_data = CameraData(include_depth=True, include_rgb=True)

	# init graspping
    generator = GraspGenerator(
    	camera,
    	cam_data,
        cam_id=830112070066,
        saved_model_path='grasp_estimation/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_13_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.camera.stream()
    target_robot_pose = generator.generate()

    move_l = MoveL.MoveL(
	    MoveL.MoveL.motion_speed_selection_posture_speed,
	    5,
	    MoveL.MoveL.coordinate_specification_base_coordinate,
        target_robot_pose[0],
        target_robot_pose[1],
        target_robot_pose[2],
        target_robot_pose[3],
        target_robot_pose[4],
        target_robot_pose[5],
	    Utils.binary_to_decimal(0x00000001),
	    0, 0, 0, 0, 0, 0, 0
	)
	    
	Commands.robot_in_target_point_callback(
	    move_l=move_l, timeout=10, _callback_success=callback_success, _callback_failed=callback_failed
	)

	Gripper.write_gripper_close()

