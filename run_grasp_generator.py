import sys
sys.path.append('disparity_estimation/Disparity/utils/') # add relative path
sys.path.append('grasp_estimation/') # add relative path
sys.path.append('nx100_remote/') # add relative path
sys.path.append('disparity_estimation/') # add relative path

from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='grasp_estimation/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_13_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.camera.stream()
    generator.run()
