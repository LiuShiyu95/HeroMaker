# HeroMaker: Video Reconstruction

## Introduction
In this section, we mainly use edited canonical images to edit video.

### Edit

Since we use ngp to construct the canonical field, we first use [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to extract the effective region.

First, please follow the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to install the environments.

Then run

```shell
python grounded_sam.py --input_name video_name
```

After that, you can run 

```shell
python edit_video.py
```

### Video Edit

After editing, please put the edited images into 
```shell
video_reconstruction/all_sequences/{name}/base_control/
```

Then run

```shell
./scripts/test_canonical_deform.sh
```


## Acknowledge
We build our code base from: [ControlNet](https://github.com/lllyasviel/ControlNet), [SDEdit](https://github.com/ermongroup/SDEdit)