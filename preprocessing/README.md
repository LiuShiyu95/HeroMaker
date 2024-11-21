# HeroMaker: Preprocessing

First, we split the video into frames by executing the following command:

```shell
python preproc_video.py /path/to/video.mp4
```

Then, we segement video sequences using [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything) like [CoDeF](https://github.com/qiuyu96/CoDeF).

Once you obtain the mask files, place them in the folder `video_reconstruction/all_sequences/{YOUR_SEQUENCE_NAME}/{YOUR_SEQUENCE_NAME}_masks`. Next, execute the following command:

```shell
python preproc_mask.py
```