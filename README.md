# Motoman NX100 learning task pipeline

Potential tasks implementation for [nx100-remote-control](https://github.com/norkator/nx100-remote-control) repository 
NX100 control backend. The tasks designed so far (implemented and to be implemented):
- [x] Dsiparity estimation from stereo camera.
- [x] Object grasping pose estimation.
- [ ] Depth-only aware grasping for generalized objects.
- [ ] Semantic visual detections. 
- [ ] Map reconstruction.

Table of contents
=================
* [Updates](#updates)
* [To-do list](#to-do-list)
* [Usage examples](#usage-examples)
* [Getting started](#getting-started)
    * [Activate Env](#activate-env)
    * [Testing 3D camera feed](#testing-3d-camera-feed)
    * [Visualize grasping](#visualize-grasping)
    * [Test grasping](#test-grasping)


Updates
============
- 2021.26.6: refactored some class implementations
- 2021.6.6: fixed disparity estimation, two disparity pipeline added for more robust estimation


To-do list
============
- [ ] Readme documentation
- [ ] Combine previous camera calibration with new disparity pipeline
- [ ] Object sgementation feature extraction
- [ ] Semantic features + depth features for imitation learning


Usage examples
============
See [disparity_estimation](./libs/disparity_estimation/README.md) and 
[stereo folders](./libs/camera_calibration/stereo_vision/stereo_calibration+undistortion+depth_pipeline.ipynb) correspondingly.


Getting started
============
Basic steps to try setup and parts.

Activate env
-----
```shell script
conda activate pytorch
```


Testing 3D camera feed
-----
To see if 3D camera is available. Should list `/dev/video0`  maybe `/dev/video1` ... maybe others?
```shell script
ls /dev/video*
```
Try feed:
```shell script
vlc v4l2:///dev/video0
```


Visualize grasping
----
Meant for testing different objects on table and how camera can see them.
```shell script
...coming
```


Test grasping
----
```shell script
python ./run_grasp_generator.py
```
