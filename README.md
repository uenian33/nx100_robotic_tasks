# Stereo camera + Motoman NX100 learning tasks
Potential tasks implementation for [nx100-remote-control](https://github.com/norkator/nx100-remote-control) repository 
NX100 control backend.


Table of contents
=================
* [Updates](#updates)
* [To-do list](#to-do-list)
* [Usage examples](#usage-examples)


Updates
============
- 2021.6.6: fixed disparity estimation, two disparity pipeline added for more robust estimation


To-do list
============
- [ ] Readme documentation
- [ ] Combine previous camera calibration with new disparity pipeline
- [ ] Object sgementation feature extraction
- [ ] Semantic features + depth features for imitation learning


Usage examples
============
See [disparity_estimation](./disparity_estimation/README.md) and 
[stereo folders](./stereo_vision/stereo_calibration+undistortion+depth_pipeline.ipynb) correspondingly.
