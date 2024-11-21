import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, resize_bbox
from common.utils.transforms import rigid_align
from torch.utils.data.dataset import Dataset
import random

class UBody_Part(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, scene):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join(cfg.data_dir, 'UBody', 'images', scene)
        self.annot_path = osp.join(cfg.data_dir, 'UBody', 'annotations', scene, 'keypoint_annotation.json')
        self.smplx_annot_path = osp.join(cfg.data_dir, 'UBody', 'annotations', scene, 'smplx_annotation.json')
        self.test_video_list_path = osp.join(cfg.data_dir, 'UBody', 'splits', 'intra_scene_test_list.npy')

        # mscoco joint set
        self.joint_set = {
            'joint_num': 134,  # body 24 (23 + pelvis), lhand 21, rhand 21, face 68
            'joints_name': \
                (
                'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe',
                'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
                'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
                'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
                'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
                'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
                'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
                'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
                *['Face_' + str(i) for i in range(56, 73)],  # face contour
                *['Face_' + str(i) for i in range(5, 56)]  # face
                ),
            'flip_pairs': \
                ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
                 # body part
                 (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54),
                 (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64),
                 (44, 65),  # hand part
                 (66, 82), (67, 81), (68, 80), (69, 79), (70, 78), (71, 77), (72, 76), (73, 75),  # face contour
                 (83, 92), (84, 91), (85, 90), (86, 89), (87, 88),  # face eyebrow
                 (97, 101), (98, 100),  # face below nose
                 (102, 111), (103, 110), (104, 109), (105, 108), (106, 113), (107, 112),  # face eyes
                 (114, 120), (115, 119), (116, 118), (121, 125), (122, 124),  # face mouth
                 (126, 130), (127, 129), (131, 133)  # face lip
                 )
        }
        self.datalist = self.load_data()

    def merge_joint(self, joint_img, feet_img, lhand_img, rhand_img, face_img):
        # pelvis
        lhip_idx = self.joint_set['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['joints_name'].index('R_Hip')
        pelvis = (joint_img[lhip_idx, :] + joint_img[rhip_idx, :]) * 0.5
        pelvis[2] = joint_img[lhip_idx, 2] * joint_img[rhip_idx, 2]  # joint_valid
        pelvis = pelvis.reshape(1, 3)

        # feet
        lfoot = feet_img[:3, :]
        rfoot = feet_img[3:, :]

        joint_img = np.concatenate((joint_img, pelvis, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(
            np.float32)  # self.joint_set['joint_num'], 3
        return joint_img

    def load_data(self):
        db = COCO(self.annot_path)
        with open(self.smplx_annot_path) as f:
            print(f'load smplx param from {self.smplx_annot_path}.')
            smplx_params = json.load(f)
        test_video_list = np.load(self.test_video_list_path)
        print(f'load test_video_list from {self.test_video_list_path}.')
        test_video_list = test_video_list.tolist()
        # train mode
        if self.data_split == 'train':
            datalist = []
            i = 0
            for aid in db.anns.keys():
                i = i + 1
                if i % cfg.train_sample_interval != 0:
                    continue
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                if img['file_name'].startswith('/'):
                    file_name = img['file_name'][1:]   # [1:] means delete '/'
                else:
                    file_name = img['file_name']
                video_name = file_name.split('/')[-2]
                if 'Trim' in video_name:
                    video_name = video_name.split('_Trim')[0]
                if video_name in test_video_list: continue   # exclude the test video
                img_path = osp.join(self.img_path, file_name)
                if not os.path.exists(img_path): continue

                # exclude the samples that are crowd or have few visible keypoints
                if ann['iscrowd'] or (ann['num_keypoints']==0): continue

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'])
                if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (
                ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                    lhand_bbox = resize_bbox(lhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                    rhand_bbox = resize_bbox(rhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    # rough face bbox
                    face_bbox = [min(face_img[:, 0]), min(face_img[:, 1]),
                            max(face_img[:, 0]), max(face_img[:, 1])]
                    face_bbox = resize_bbox(face_bbox, scale=1.2)
                    face_bbox = np.array(face_bbox)
                    # face_bbox = np.array(ann['face_box']).reshape(4)
                    # face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
                else:
                    face_bbox = None

                if ann['valid_label'] == 0 or str(aid) not in smplx_params: continue

                smplx_param = smplx_params[str(aid)]
                if 'lhand_valid' not in smplx_param['smplx_param']:
                    smplx_param['smplx_param']['lhand_valid'] = ann['lefthand_valid']
                    smplx_param['smplx_param']['rhand_valid'] = ann['righthand_valid']
                    smplx_param['smplx_param']['face_valid'] = ann['face_valid']


                data_dict = {'img_path': img_path, 'img_shape': (img['height'], img['width']), 'bbox': bbox,
                             'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param,
                             'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)

            return datalist

        # test mode
        else:
            datalist = []
            i = 0
            for aid in db.anns.keys():
                i = i + 1
                if i % cfg.test_sample_interval != 0:
                    continue
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                if img['file_name'].startswith('/'):
                    file_name = img['file_name'][1:]  # [1:] means delete '/'
                else:
                    file_name = img['file_name']
                video_name = file_name.split('/')[-2]
                if 'Trim' in video_name:
                    video_name = video_name.split('_Trim')[0]
                if video_name not in test_video_list: continue  # exclude the train video
                img_path = osp.join(self.img_path, file_name)
                if not os.path.exists(img_path): continue

                # exclude the samples that are crowd or have few visible keypoints
                if ann['iscrowd'] or (ann['num_keypoints']==0): continue

                if ann['valid_label'] == 0 or str(aid) not in smplx_params: continue

                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height'])
                if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
                lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
                rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
                face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                joint_img = self.merge_joint(joint_img, foot_img, lhand_img, rhand_img, face_img)

                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = 0

                # use body annotation to fill hand/face annotation
                for body_name, part_name in (
                        ('L_Wrist', 'L_Wrist_Hand'), ('R_Wrist', 'R_Wrist_Hand'), ('Nose', 'Face_18')):
                    if joint_valid[self.joint_set['joints_name'].index(part_name), 0] == 0:
                        joint_img[self.joint_set['joints_name'].index(part_name)] = joint_img[
                            self.joint_set['joints_name'].index(body_name)]
                        joint_valid[self.joint_set['joints_name'].index(part_name)] = joint_valid[
                            self.joint_set['joints_name'].index(body_name)]

                # hand/face bbox
                if ann['lefthand_valid']:
                    lhand_bbox = np.array(ann['lefthand_box']).reshape(4)
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
                    lhand_bbox = resize_bbox(lhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    lhand_bbox = None
                if ann['righthand_valid']:
                    rhand_bbox = np.array(ann['righthand_box']).reshape(4)
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
                    rhand_bbox = resize_bbox(rhand_bbox, scale=0.65)  # 1.5*0.8=1.2
                else:
                    rhand_bbox = None
                if ann['face_valid']:
                    face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
                    face_bbox = [min(face_img[:, 0]), min(face_img[:, 1]),
                                 max(face_img[:, 0]), max(face_img[:, 1])]
                    face_bbox = resize_bbox(face_bbox, scale=1.2)
                    face_bbox = np.array(face_bbox)
                else:
                    face_bbox = None

                if str(aid) in smplx_params:
                    smplx_param = smplx_params[str(aid)]
                    if 'lhand_valid' not in smplx_param['smplx_param']:
                        smplx_param['smplx_param']['lhand_valid'] = ann['lefthand_valid']
                        smplx_param['smplx_param']['rhand_valid'] = ann['righthand_valid']
                        smplx_param['smplx_param']['face_valid'] = ann['face_valid']
                else:
                    smplx_param = None

                data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox,
                             'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param,
                             'lhand_bbox': lhand_bbox, 'rhand_bbox': rhand_bbox, 'face_bbox': face_bbox}
                datalist.append(data_dict)
            return datalist

    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            bbox_valid = float(True)
            bbox = bbox.reshape(2, 2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # train mode
        if self.data_split == 'train':
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape,
                                                                     img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coco gt
            dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)  # x, y, dummy depth
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord,
                                                                              data['joint_valid'], do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)

            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            if smplx_param is not None:
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                    smplx_param['smplx_param'], smplx_param['cam_param'], do_flip, img_shape, img2bb_trans, rot,
                    'smplx')
                is_valid_fit = True

                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

            else:
                # dummy values
                smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
                smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
                smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
                smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
                smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
                smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
                smplx_expr_valid = False
                is_valid_fit = False

            # SMPLX pose parameter validity
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

            # make zero mask for invalid fit
            if not is_valid_fit:
                smplx_pose_valid[:] = 0
                smplx_joint_valid[:] = 0
                smplx_joint_trunc[:] = 0
                smplx_shape_valid = False
            else:
                smplx_shape_valid = True

            inputs = {'img': img, }
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smplx_joint_img': smplx_joint_img,
                       'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr, 'lhand_bbox_center': lhand_bbox_center,
                       'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center,
                       'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False), 'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            return inputs, targets, meta_info

        # test mode
        else:
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape,
                                                                       img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape,
                                                                     img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.;
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.;
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0];
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
            face_bbox_size = face_bbox[1] - face_bbox[0];

            # coco gt
            dummy_coord = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], np.zeros_like(joint_img[:, :1])), 1)  # x, y, dummy depth
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord,
                                                                              data['joint_valid'], do_flip, img_shape,
                                                                              self.joint_set['flip_pairs'],
                                                                              img2bb_trans, rot,
                                                                              self.joint_set['joints_name'],
                                                                              smpl_x.joints_name)


            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            # if str(data['ann_id'])=='184516':
            #     print(data['ann_id'], smplx_param)
            if smplx_param is not None:
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                    smplx_param['smplx_param'], smplx_param['cam_param'], do_flip, img_shape, img2bb_trans, rot,
                    'smplx')
                is_valid_fit = True
                smplx_cam_trans = np.array(smplx_param['smplx_param']['trans'])
                # if str(data['ann_id']) == '184516':
                #     print(data['ann_id'], smplx_pose)
                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

            else:
                # dummy values
                smplx_joint_img = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_cam = np.zeros((smpl_x.joint_num, 3), dtype=np.float32)
                smplx_joint_trunc = np.zeros((smpl_x.joint_num, 1), dtype=np.float32)
                smplx_joint_valid = np.zeros((smpl_x.joint_num), dtype=np.float32)
                smplx_pose = np.zeros((smpl_x.orig_joint_num * 3), dtype=np.float32)
                smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
                smplx_expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
                smplx_pose_valid = np.zeros((smpl_x.orig_joint_num), dtype=np.float32)
                cam_param_num = 3
                smplx_cam_trans = np.zeros((cam_param_num), dtype=np.float32)
                smplx_expr_valid = False
                is_valid_fit = False

            # SMPLX pose parameter validity
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc

            # make zero mask for invalid fit
            if not is_valid_fit:
                smplx_pose_valid[:] = 0
                smplx_joint_valid[:] = 0
                smplx_joint_trunc[:] = 0
                smplx_shape_valid = False
            else:
                smplx_shape_valid = True

            inputs = {'img': img, }
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smplx_joint_img': smplx_joint_img,
                       'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape,
                       'smplx_expr': smplx_expr, 'smplx_cam_trans': smplx_cam_trans,
                       'lhand_bbox_center': lhand_bbox_center,
                       'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center,
                       'rhand_bbox_size': rhand_bbox_size, 'face_bbox_center': face_bbox_center,
                       'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'smplx_joint_trunc': smplx_joint_trunc, 'smplx_pose_valid': smplx_pose_valid,
                         'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid),
                         'is_3D': float(False), 'lhand_bbox_valid': lhand_bbox_valid,
                         'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid,
                         'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

class UBody(Dataset):
    def __init__(self, transform, mode='test'):
        self.dbs = []

        self.aids = []
        # self.img_paths = []
        self.parts = []
        self.datalist = []
        folder = osp.join(cfg.data_dir, 'UBody', 'images')
        for scene in os.listdir(folder):
            db = UBody_Part(transform, mode, scene=scene)
            self.dbs.append(db)
            self.datalist += db.datalist

        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in self.dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])
        self.make_same_len = cfg.make_same_len
        print(f'Number of images: {sum([len(db) for db in self.dbs])}')

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        # print(self.__len__(), len(self.parts))
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]

    def perspective_transform(self, cam_trans, bb2img_trans, inputs):
        # inputs: [num_points, 3], world coordinate
        x = (inputs[:, 0] + cam_trans[None, 0]) / (
                    inputs[:, 2] + cam_trans[None, 2] + 1e-4) * \
            cfg.focal[0] + cfg.princpt[0]
        y = (inputs[:, 1] + cam_trans[None, 1]) / (
                    inputs[:, 2] + cam_trans[None, 2] + 1e-4) * \
            cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.input_img_shape[1]  # input_img_shape
        y = y / cfg.input_body_shape[0] * cfg.input_img_shape[0]  # input_img_shape
        out = np.stack((x, y, np.ones_like(x)), 1)  # [num_points, 3], input_img_shape
        out = np.dot(bb2img_trans, out.transpose(1, 0)).transpose(1, 0)  # [num_points, 2], original image space
        return out

    def validate_within_img(self, img, points):  # check whether the points is within the image
        # img: (h, w, c), points: (num_points, 2)
        h, w, c = img.shape
        valid_mask = np.logical_and(np.logical_and(0 < points[:, 0], points[:, 0] < w),
                                    np.logical_and(0 < points[:, 1], points[:, 1] < h))

        return valid_mask

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'mpvpe_all': [], 'mpvpe_hand': [],
                       'mpvpe_face': [], 'pa_mpjpe_body': [], 'pa_mpjpe_hand': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # perspective transformation of the joints
            mesh_gt = out['smplx_mesh_cam_pseudo_gt']
            cam_trans = out['cam_trans']
            joint_gt_body_wo_trans = np.dot(smpl_x.j14_regressor, mesh_gt - cam_trans)
            joint_gt_body_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                            joint_gt_body_wo_trans)  # origin image space
            joint_gt_lhand_wo_trans = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt - cam_trans)
            joint_gt_lhand_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                             joint_gt_lhand_wo_trans)  # origin image space
            joint_gt_rhand_wo_trans = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt - cam_trans)
            joint_gt_rhand_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'],
                                                             joint_gt_rhand_wo_trans)  # origin image space
            mesh_gt_proj = self.perspective_transform(out['cam_trans'], out['bb2img_trans'], mesh_gt-cam_trans)

            img_path = annot['img_path']
            img = load_img(img_path)[:, :, ::-1]

            # We only calculate the error of the joints/vertices within the image plane
            joint_gt_body_valid = self.validate_within_img(img, joint_gt_body_proj)
            joint_gt_lhand_valid = self.validate_within_img(img, joint_gt_lhand_proj)
            joint_gt_rhand_valid = self.validate_within_img(img, joint_gt_rhand_proj)
            mesh_valid = self.validate_within_img(img, mesh_gt_proj)
            mesh_lhand_valid = mesh_valid[smpl_x.hand_vertex_idx['left_hand']]
            mesh_rhand_valid = mesh_valid[smpl_x.hand_vertex_idx['right_hand']]
            mesh_face_valid = mesh_valid[smpl_x.face_vertex_idx]

            # MPVPE from all vertices
            mesh_out = out['smplx_mesh_cam']
            mesh_out_align = rigid_align(mesh_out, mesh_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1))[mesh_valid].mean() * 1000)
            mesh_out_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] + \
                             np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None, :]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, 1))[mesh_valid].mean() * 1000)

            # MPVPE from hand vertices
            mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
            mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
            mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
            mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
            pa_mpvpe_hand = []
            if sum(mesh_lhand_valid) != 0:
                pa_mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1))[
                                     mesh_lhand_valid].mean() * 1000
                pa_mpvpe_hand.append(pa_mpvpe_lhand)
            if sum(mesh_rhand_valid) != 0:
                pa_mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1))[
                                     mesh_rhand_valid].mean() * 1000
                pa_mpvpe_hand.append(pa_mpvpe_rhand)
            if len(pa_mpvpe_hand) > 0:
                eval_result['pa_mpvpe_hand'].append(np.mean(pa_mpvpe_hand))

            mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
            mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                                    smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]

            mpvpe_hand = []
            if sum(mesh_lhand_valid) != 0:
                mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1))[mesh_lhand_valid].mean() * 1000
                mpvpe_hand.append(mpvpe_lhand)
            if sum(mesh_rhand_valid) != 0:
                mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1))[mesh_rhand_valid].mean() * 1000
                mpvpe_hand.append(mpvpe_rhand)
            if len(mpvpe_hand) > 0:
                eval_result['mpvpe_hand'].append(np.mean(mpvpe_hand))

            # MPVPE from face vertices
            mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
            mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
            mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
            if sum(mesh_face_valid) != 0:
                eval_result['pa_mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1))[mesh_face_valid].mean() * 1000)
            mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                                  None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                             smpl_x.J_regressor_idx['neck'], None, :]
            if sum(mesh_face_valid) != 0:
                eval_result['mpvpe_face'].append(
                    np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1))[mesh_face_valid].mean() * 1000)

            # MPJPE from body joints
            joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
            joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1))[joint_gt_body_valid].mean() * 1000)

            # MPJPE from hand joints
            joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
            joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
            joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
            joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
            joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
            joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)

            pa_mpjpe_hand = []
            if sum(joint_gt_lhand_valid)!=0:
                pa_mpjpe_lhand = np.sqrt(np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1))[joint_gt_lhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_lhand)
            if sum(joint_gt_rhand_valid)!=0:
                pa_mpjpe_rhand = np.sqrt(np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1))[joint_gt_rhand_valid].mean() * 1000
                pa_mpjpe_hand.append(pa_mpjpe_rhand)
            if len(pa_mpjpe_hand)>0:
                eval_result['pa_mpjpe_hand'].append(np.mean(pa_mpjpe_hand))

            # data_dict = {}
            # data_dict['mpvpe_all'] = eval_result['mpvpe_all'][-1]
            # data_dict['mpvpe_hand'] = eval_result['mpvpe_hand'][-1]
            # data_dict['mpvpe_face'] = eval_result['mpvpe_face'][-1]
            # data_dict['mesh'] = mesh_out
            # data_dict['mesh_gt'] = mesh_gt

            vis = cfg.vis
            save_folder = cfg.vis_dir
            data_folder = os.path.join(cfg.root_dir, 'dataset', 'UBody', 'images')
            if vis:
                from common.utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
                img_path = annot['img_path']
                render_img_save_path = img_path.replace(data_folder, f'{save_folder}/render/')
                if os.path.exists(render_img_save_path):
                    img = load_img(render_img_save_path)[:, :, ::-1]
                else:
                    img = load_img(img_path)[:, :, ::-1]


                ''' for debug
                kpt_path = render_img_save_path.replace('/render/', '/keypoints/')
                kpt_img = img.copy()
                kpt_img = vis_keypoints(kpt_img, joint_proj)
                # kpt_img = vis_keypoints(kpt_img, mesh_gt_proj)
                os.makedirs(os.path.dirname(kpt_path), exist_ok=True)
                cv2.imwrite(kpt_path, kpt_img)    
                '''

                bbox = annot['bbox']
                focal = list(cfg.focal)
                princpt = list(cfg.princpt)
                focal[0] = focal[0] / cfg.input_body_shape[1] * bbox[2]
                focal[1] = focal[1] / cfg.input_body_shape[0] * bbox[3]
                princpt[0] = princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0]
                princpt[1] = princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]
                img = render_mesh(img, out['smplx_mesh_cam'], smpl_x.face, {'focal': focal, 'princpt': princpt})
                os.makedirs(os.path.dirname(render_img_save_path), exist_ok=True)
                cv2.imwrite(render_img_save_path, img)

        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))
        print()

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))
        print()

        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('PA MPJPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpjpe_hand']))

        f = open(os.path.join(cfg.result_dir, 'result.txt'), 'w')
        f.write(f'UBody dataset: \n')
        f.write('PA MPVPE (All): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_all']))
        f.write('PA MPVPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_hand']))
        f.write('PA MPVPE (Face): %.2f mm\n' % np.mean(eval_result['pa_mpvpe_face']))
        f.write('MPVPE (All): %.2f mm\n' % np.mean(eval_result['mpvpe_all']))
        f.write('MPVPE (Handsls): %.2f mm\n' % np.mean(eval_result['mpvpe_hand']))
        f.write('MPVPE (Face): %.2f mm\n' % np.mean(eval_result['mpvpe_face']))
        f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        f.write('PA MPJPE (Hands): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_hand']))

