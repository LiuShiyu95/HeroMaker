import os
import sys
import argparse
import glob
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy
from config import cfg
from scipy.ndimage import distance_transform_edt
import torch
sys.path.insert(0, os.path.join('..', 'main'))
sys.path.insert(0, os.path.join('..', 'data'))
sys.path.insert(0, "../../osx_estimator")
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_path', type=str, default='imgs_folder')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args


args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj, render_mesh_deg, render_mesh_black, vis_keypoints
from common.utils.human_models import smpl_x
from common.utils.utils import smpl_to_openpose

smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
input_img_shape = (512, 384)

def get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, delta=None, pose_type="posed"):

    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
    output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, delta=delta,
                                reye_pose=zero_pose, expression=expr, pose_type=pose_type)
    
    yuan_mesh = output.vertices
    
    mesh_cam = output.vertices

    joint_cam = output.joints[:, smpl_x.joint_idx, :]

    x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        cfg.focal[0] + cfg.princpt[0]
    y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        cfg.focal[1] + cfg.princpt[1]
    x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
    y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
    joint_proj = torch.stack((x, y), 2)
    # root-relative 3D coordinates
    root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
    joint_cam = joint_cam - root_cam

    mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
    # left hand root (left wrist)-relative 3D coordinates
    lhand_idx = smpl_x.joint_part['lhand']
    lhand_cam = joint_cam[:, lhand_idx, :]
    lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
    lhand_cam = lhand_cam - lwrist_cam
    joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

    # right hand root (right wrist)-relative 3D coordinates
    rhand_idx = smpl_x.joint_part['rhand']
    rhand_cam = joint_cam[:, rhand_idx, :]
    rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
    rhand_cam = rhand_cam - rwrist_cam
    joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

    # face root (neck)-relative 3D coordinates
    face_idx = smpl_x.joint_part['face']
    face_cam = joint_cam[:, face_idx, :]
    neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
    face_cam = face_cam - neck_cam
    joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

    return joint_proj, joint_cam, mesh_cam, yuan_mesh


def save_img(img, path):
    """Save an image to the specified path."""
    numpy_image = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    bgr_image = ((bgr_image+1) / 2 * 255).astype('uint8') # Scale values from [0, 1] to [0, 255]
    cv2.imwrite(path, bgr_image)


def read_cv2_img(path):
    """
    Read color images
    :param path: Path to image
    :return: Only returns color images
    """
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def transform_img(image, image_size, transpose=False):
    """Resize and normalize the image."""
    image = cv2.resize(image, (image_size, image_size)).astype(np.float32) / 255.0
    if transpose:
        image = image.transpose((2, 0, 1))
    return image

model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path

comp_device ="cuda"
transform = transforms.ToTensor()
imgs_list = sorted(glob.glob(f'{args.img_path}/*'))

detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
faces = smpl_x.face.astype(np.int32)
faces = torch.from_numpy(faces).unsqueeze(0).to("cuda")
out_list = []

for i in range(len(imgs_list)):
    original_img = load_img(imgs_list[i])
    original_img_height, original_img_width = original_img.shape[:2]
    vis_img = original_img.copy()
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]

    class_ids, confidences, boxes = [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    for num, indice in enumerate(indices):
        if num != 0:
            break
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with torch.no_grad():            
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
        joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
        joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
        joint_proj = joint_proj[:25,:]

        mesh = mesh[0]
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]

        out['bb2img_trans'] = torch.tensor(bb2img_trans).unsqueeze(0)
        out['focal'] = torch.tensor(focal).unsqueeze(0)
        out['princpt'] = torch.tensor(princpt).unsqueeze(0)
        out_list.append(out)

out_list = {
    k: np.concatenate([r[k].detach().cpu().numpy() for r in out_list], axis=0) for k in out
}

keys_to_cuda = [
    'focal', 'princpt', 'smplx_root_pose', 'smplx_body_pose', 
    'smplx_lhand_pose', 'smplx_rhand_pose', 'smplx_jaw_pose', 
    'smplx_shape', 'smplx_expr', 'cam_trans'
]
data = {key: torch.from_numpy(out_list[key]).cuda() for key in keys_to_cuda}

def generate_mesh(data, pose_type):
    joint_proj, joint_cam, mesh_cam, yuan_mesh = get_coord(
        data['smplx_root_pose'][0].unsqueeze(0),
        data['smplx_body_pose'][0].unsqueeze(0),
        data['smplx_lhand_pose'][0].unsqueeze(0),
        data['smplx_rhand_pose'][0].unsqueeze(0),
        data['smplx_jaw_pose'][0].unsqueeze(0),
        data['smplx_shape'][0].unsqueeze(0),
        data['smplx_expr'][0].unsqueeze(0),
        data['cam_trans'][0].unsqueeze(0),
        delta=None,
        pose_type=pose_type
    )
    return mesh_cam.detach().cpu().numpy()

out_list['smplx_mesh_front'] = generate_mesh(data, pose_type='a-pose-front')
out_list['smplx_mesh_back'] = generate_mesh(data, pose_type='a-pose-back')

np.savez(f'./{imgs_folder_path}_osx_smpl.npz', **out_list)




