from libs.grasp_estimation.inference.grasp_generator import GraspGenerator
from libs.camera.custom_camera import CustomCamera
from libs.camera.camera_data import CameraData
import time

# Todo, define proper starting point
# starting_point = [123, 123, 123, 20, 20, 10]
# Todo, define joint move position
# joint_position_1 = [123, 123, 123, 20, 20, 10]
# joint_position_2 = [123, 123, 123, 20, 20, 10]


if __name__ == '__main__':
    # init camera setups
    camera = CustomCamera()
    cam_data = CameraData(include_depth=True, include_rgb=True)

    # init grasping
    generator = GraspGenerator(
        camera,
        cam_data,
        cam_id=830112070066,
        saved_model_path='grasp_estimation/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_13_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.camera.stream()

    while True:
        generator.generate(visualize=True)

