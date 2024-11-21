# HeroMaker: Motion Priors Acquiring

## Introduction

In this section, we mainly obtain accurate motion priors through two steps.

### Step 1: Obtain the prepared 2D key points.

You can refer to the [mmpose environment](https://github.com/open-mmlab/mmpose/tree/main/projects/mmpose4aigc) for preparation and then run:

```shell
python openpose_estimation.py --input_folder_path /path/to/imgs_folder
```

### Step 2: Obtain human recovery mesh.

You can refer to the [OSX environment](https://github.com/open-mmlab/mmpose/tree/main/projects/mmpose4aigc) for preparation.

```shell
cd osx_estimator/demo/osx_estimation.py
python osx_estimation.py --img_path /path/to/imgs_folder
```

### Step 3: Optimize human recovery mesh.

```shell
python motion_refinement.py --name video_name
```

## Acknowledge
We build our code base from: [mmpose](https://github.com/open-mmlab/mmpose/tree/main), [OSX](https://github.com/IDEA-Research/OSX)