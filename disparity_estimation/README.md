# Disparity estimation
Disparity estimation is a difficult problem in stereo vision because the correspondence technique fails in images with textureless and repetitive regions. This work simply tested traditional pipeline method and DL based method for customized stereo camera disparity estimation: 1) Traiditional SGBM feature matching based approach and 2) Transformer based sequential feature matching approahc (STTR). Both of the approaches currently is able to run nearly realtime and appliable to any stereo camera setups.

# To-do list
- [ ] Readme documentation
- [x] Simple disparity pipeline: image preprocessing + estimation
- [ ] Optimize the structure and retrain the model
- [ ] image rectifying based epipolar geometry + camera parameters
- [ ] implement different image denoising+white balance methods


# Description

## Pipeline 

The disparity follows the pipeline of preprocessing -> disparity estimation.

### preprocessing
Precprocessing is a necessary part for traditional feature matching method. Due to the bad image quality might caused by hardwares, the image would be noisy and color is not correct. The simple preprocessing uses a  



## Depth estimation
To be added

## code example

```python
from Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition


# example of using DL pipeline for disparity estimation
destimation = DisparityEstimationDL(
                model_file_name="Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                wb_model_file_name="Disparity/utils/WB/models/")
left_image_path='../images/test2/camL/'+str(0)+'.jpg'
right_image_path='../images/test2/camR/'+str(0)+'.jpg'
left_image = cv2.resize(cv2.imread(left_image_path, cv2.IMREAD_COLOR), dsize=(1200, 600), interpolation=cv2.INTER_CUBIC)
right_image = cv2.resize(cv2.imread(right_image_path, cv2.IMREAD_COLOR), dsize=(1200, 600), interpolation=cv2.INTER_CUBIC)



disp_pred, occ_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)
plt.figure(3)
plt.imshow(disp_pred)
"""
plt.figure(5)
plt.imshow(occ_pred)
plt.show()
"""

# example of using traditional pipeline for disparity estimation
destimation = DisparityEstimationTradition(
                wb_model_file_name="Disparity/utils/WB/models/")

disp_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)

plt.figure(4)
plt.imshow(disp_pred)
plt.show()
```

## Reference
The source code for amazing white balance adjustment is from work (https://github.com/mahmoudnafifi/WB_sRGB)
```
@inproceedings{afifi2019color,
  title={When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images},
  author={Afifi, Mahmoud and Price, Brian and Cohen, Scott and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1535--1544},
  year={2019}
}
```

The source code for great SOTA disparity estimation is using method (https://github.com/mli0603/stereo-transformer)
```
@article{li2020revisiting,
  title={Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers},
  author={Li, Zhaoshuo and Liu, Xingtong and Drenkow, Nathan and Ding, Andy and Creighton, Francis X and Taylor, Russell H and Unberath, Mathias},
  journal={arXiv preprint arXiv:2011.02910},
  year={2020}
}
```

