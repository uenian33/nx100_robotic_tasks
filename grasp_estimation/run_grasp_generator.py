sys.path.append('disparity_estimation/Disparity/utils/') # add relative path
sys.path.append('grasp_estimation/') # add relative path
from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=830112070066,
        saved_model_path='saved_data/cornell_rgbd_iou_0.96',
        visualize=True
    )
    generator.load_model()
    generator.run()
