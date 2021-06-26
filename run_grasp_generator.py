from libs.grasp_estimation.inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='grasp_estimation/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_13_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.camera.stream()
    target_robot_pose = generator.generate()
	Commands.write_linear_move(MoveL.MoveL(
	        0, 45, 0,
	        target_robot_pose[0],
	        target_robot_pose[1],
	        target_robot_pose[2],
	        target_robot_pose[3],
	        target_robot_pose[4],
	        target_robot_pose[5],
	        Utils.binary_to_decimal(0x00000001)
	    ))

	time.sleep(10)
	Gripper.write_gripper_close()

