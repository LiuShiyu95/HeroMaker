# HeroMaker: Video Reconstruction

## Introduction

In this section, we mainly use motion priority to reconstruct videos and obtain deformation fields and dual canonical fields.

First, we use [Neural Mesh Renderer](https://github.com/svip-lab/impersonator/tree/master/thirdparty/neural_renderer) to build human motion warping.

```shell
cd thirdparty
git clone https://github.com/StevenLiuWen/neural_renderer.git
cd neural_renderer
python setup.py install
```

For additional Python libraries, please install with

```shell
pip install -r requirements.txt
```

Our code also depends on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
See [this repository](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)
for Pytorch extension install instructions.

## Data

After obtaining the files, please organize your own data as follows:

```
Video_reconstruction
│
└─── all_sequences
    │
    └─── NAME1
           └─ NAME1
           └─ NAME1_masks_0 (optional)
           └─ NAME1_masks_1 (optional)
    │
    └─── NAME2
           └─ NAME2
           └─ NAME2_masks_0 (optional)
           └─ NAME2_masks_1 (optional)
    │
    └─── ...
```

## Training and Reconstruction

### Train
```shell
./scripts/train_multi_deform.sh
```
Please check configuration files in [configs](./configs/), and you can always add your own model config.

### Test
```shell
./scripts/test_multi_deform.sh
```
After running the script, the reconstructed videos can be found in [results/all_sequences/{NAME}/{EXP_NAME}](./all_sequences/), along with the canonical image.

### Reconstruct

After obtaining the canonical image through this step, use your preferred text prompts or user input to transfer it using [ControlNet](https://github.com/lllyasviel/ControlNet) and [SDEdit](https://github.com/ermongroup/SDEdit). Please refer to the [video editing section](../video_editing/) for the video editing code

Then run

```shell
./scripts/test_canonical_deform.sh
```


## Acknowledge
We build our code base from: [iPERCore](https://github.com/iPERDance/iPERCore), [CoDeF](https://github.com/qiuyu96/CoDeF)