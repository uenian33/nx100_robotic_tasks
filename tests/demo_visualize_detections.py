import os
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

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
    camera = CustomCamera(disp_model_pth=os.getcwd() + "/libs/disparity_estimation/Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                          wb_model_pth=os.getcwd() + "/libs/disparity_estimation/Disparity/utils/WB/models/")
    cam_data = CameraData(include_depth=True, include_rgb=True)

    # init grasping
    generator = GraspGenerator(
        camera,
        cam_data,
        cam_id=830112070066,
        saved_model_path=os.getcwd() + "/libs/grasp_estimation/trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93",
        visualize=True
    )
    generator.load_model()
    generator.camera.stream()

    while True:
        generator.generate(visualize=True)

