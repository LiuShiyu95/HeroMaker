# Copyright (c) OpenMMLab. All rights reserved.

import math
import mimetypes
import os
import glob
import sys
from argparse import ArgumentParser
from itertools import product

import cv2
import mmcv
import numpy as np

from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Global variables
limb_seq = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], 
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], [14, 19],
    [19, 20], [14, 21], [11, 22], [22, 23],[11, 24]
]
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255,0], 
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
    [0, 0, 255], [85, 0, 255], [170, 0,255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85], [0, 255, 85], [0, 255, 170], 
    [0, 255, 255], [0, 170, 255], [0, 255, 85], [0, 255, 170], 
    [0, 255, 255], [0, 170, 255]
]

STICKWIDTH = 4
NUM_OPENPOSE_KPT = 25
NUM_LINK = len(LIMB_SEQ)


def mmpose_to_openpose_visualization(args, img_path, detector, pose_estimator, zxzx):
    """Visualize predicted keypoints of one image in openpose format."""
    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    det_result = inference_detector(detector, img_path)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[
        np.logical_and(pred_instance.labels == args.det_cat_id,
                        pred_instance.scores > args.bbox_thr)
    ]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # concatenate scores and keypoints
    keypoints = np.concatenate(
        (data_samples.pred_instances.keypoints,
         data_samples.pred_instances.keypoint_scores.reshape(-1, 26, 1)),
        axis=-1
    )
    if keypoints.shape[0] != 1:
        keypoints = keypoints[:1,:,:]
    print(keypoints.shape)

    # compute neck joint
    neck = (keypoints[:, 5] + keypoints[:, 6]) / 2
    if keypoints[:, 5, 2] < args.kpt_thr or keypoints[:, 6, 2] < args.kpt_thr:
        neck[:, 2] = 0

    # Map mmpose format to openpose format
    openpose_idx = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 1, 8, 19, 22, 20, 23, 21, 24]
    mmpose_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25]

    new_keypoints = keypoints.copy()
    new_keypoints[:, openpose_idx, :] = keypoints[:, mmpose_idx, :]
    new_keypoints = new_keypoints[:,:NUM_OPENPOSE_KPT,:]

    # Visualize results on a black background
    img = mmcv.imread(img_path, channel_order='rgb')
    # white background
    back_img = np.full_like(img, 255)  # Initialize white background
    num_instance = new_keypoints.shape[0]
    
    # Draw keypoints
    for i, j in product(range(num_instance), range(NUM_OPENPOSE_KPT)):
        x, y, conf = new_keypoints[i][j]
        if conf > args.kpt_thr:
            cv2.circle(back_img, (int(x), int(y)), 4, colors[j], thickness=-1)

    # Draw links
    cur_back_img = back_img.copy()
    for i, link_idx in product(range(num_instance), range(NUM_LINK)):
        conf = new_keypoints[i][np.array(limb_seq[link_idx]), 2]
        if np.sum(conf > args.kpt_thr) == 2:
            Y = new_keypoints[i][np.array(limb_seq[link_idx]), 0]
            X = new_keypoints[i][np.array(limb_seq[link_idx]), 1]
            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), STICKWIDTH), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_back_img, polygon, colors[link_idx])
    back_img = cv2.addWeighted(back_img, 0.4, cur_back_img, 0.6, 0)

    # save image
    out_file = "/public/home/liushy/OSX/demo/pose_test/" + f'{zxzx}.png'
    cv2.imwrite(out_file, back_img[:, :, [2, 1, 0]])
    return new_keypoints




def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', default="./rtmdet_nano_320-8xb32_coco-person.py", help='Config file for detection')
    parser.add_argument('det_checkpoint', default="./openpose_model/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth", help='Checkpoint file for detection')
    parser.add_argument('pose_config', default="./rtmpose-m_8xb256-420e_coco-256x192.py", help='Config file for pose')
    parser.add_argument('pose_checkpoint', default="./openpose_model/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth", help='Checkpoint file for pose')
    parser.add_argument('input_folder_path', default="imgs_folder", type=str, help='input images folder path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.4,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.4, help='Keypoint score threshold')

    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()

    # Build detector and pose estimator
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    # Process input images
    pose_data_list = []
    input_images_path = "images_folder_path"
    image_path = sorted(glob.glob(f'{input_images_path}/*'))
    for i in range(len(image_path)):
        output =mmpose_to_openpose_visualization(args, image_path[i], detector,
                                        pose_estimator, i)
        pose_data_list.append(output)
    pose_data_array = np.stack(pose_data_list, axis=0)  
    np.save(f"./pose_true.npy", pose_data_array)

if __name__ == '__main__':
    main()