import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from einops import rearrange, reduce, repeat
from torchvision import transforms as T
import glob
import cv2
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from tqdm import tqdm
from networks.networks import NetworksFactory, HumanModelRecovery
from util.nmr import SMPLRenderer
import util.cv_utils as cv_utils
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
sys.path.insert(0, "/public/home/liushy/OSX")
sys.path.insert(0, "/public/home/liushy/OSX/main")
import glob
import cv2
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img


def save_img(img, path):
    numpy_image = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    bgr_image = (bgr_image * 255).astype('uint8')
    cv2.imwrite(path, bgr_image)


def read_cv2_img(path):
    """
    Read color images
    :param path: Path to image
    :return: Only returns color images
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def transform_img(image, image_size, transpose=False):
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)
    image /= 255.0
    if transpose:
        image = image.transpose((2, 0, 1))
    return image


# load model
class VideoDatasetDeform(Dataset):

    def __init__(self,
                 root_dir,
                 split='train',
                 img_wh=(512, 512),
                 mask_dir=None,
                 canonical_wh=None,
                 ref_idx=None,
                 canonical_dir=None,
                 name=None,
                 test=False):
        self.test = test
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.mask_dir = mask_dir
        self.canonical_wh = canonical_wh
        self.ref_idx = ref_idx
        self.canonical_dir = canonical_dir
        self.name = name

        # follow iPERDance: https://github.com/iPERDance/iPERCore/tree/main
        self.render = SMPLRenderer(map_name="uv_seg",
                              uv_map_path="assets/pretrains/mapper.txt",
                              tex_size=3,
                              image_size=img_wh[1], fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front=False).cuda()

        self.smpl_list = np.load(f"../../motion_acquiring/{self.name}_out_data_deform.npz")

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.faces = self.smplx_layer.faces_tensor[None].repeat(1, 1, 1).cuda()

        image_path = sorted(glob.glob(f'{self.root_dir}/*'))
        root_pose = self.smpl_list['smplx_root_pose']
        
        min_index = 0
        max_index = 0
        for i in range(root_pose.shape[0]):
            if np.abs(root_pose[i][2]) < np.abs(root_pose[min_index][2]):
                min_index = i
            if np.abs(root_pose[i][2]) > np.abs(root_pose[max_index][2]):
                max_index = i

        first_image_path = image_path[min_index]
        input_image = cv_utils.read_cv2_img(first_image_path)
        input_image1 = cv2.resize(input_image, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
        self.front_next_image_tensor = torch.from_numpy(input_image1).float() / 255

        second_image_path = image_path[max_index]
        input_image = cv_utils.read_cv2_img(second_image_path)
        input_image2 = cv2.resize(input_image, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
        self.back_next_image_tensor = torch.from_numpy(input_image2).float() / 255

        gamma = self.smpl_list['gamma'][min_index]
        scale = self.smpl_list['focal'][min_index] / 5000
        scale = torch.tensor(scale).cuda()
        gamma = torch.tensor(gamma).cuda()
        princpt = self.smpl_list['princpt'][min_index]
        deviation = [(princpt[0]-256)/256,(princpt[1]-256)/256,0]
        deviation = torch.Tensor(deviation).unsqueeze(0).unsqueeze(0).cuda()
        front_verts = torch.from_numpy(self.smpl_list["smplx_mesh_cam"][min_index:min_index+1]).cuda()
        next_front_src_f2verts, next_front_src_fim, next_front_src_wim = self.render.render_fim_wim_new(front_verts, self.faces, gamma, deviation, scale[0])
        next_front_src_f2verts = next_front_src_f2verts[:, :, :, 0:2]
        next_front_src_f2verts[:, :, :, 1] *= -1
        

        gamma = self.smpl_list['gamma'][max_index]
        scale = self.smpl_list['focal'][max_index] / 5000
        scale = torch.tensor(scale).cuda()
        gamma = torch.tensor(gamma).cuda()
        princpt = self.smpl_list['princpt'][max_index]
        deviation = [(princpt[0]-256)/256,(princpt[1]-256)/256,0]
        deviation = torch.Tensor(deviation).unsqueeze(0).unsqueeze(0).cuda()
        verts = torch.from_numpy(self.smpl_list["smplx_mesh_cam"][max_index:max_index+1]).cuda()
        next_back_src_f2verts, next_back_src_fim, next_back_src_wim = self.render.render_fim_wim_new(verts, self.faces, gamma, deviation, scale[0])
        next_back_src_f2verts = next_back_src_f2verts[:, :, :, 0:2]
        next_back_src_f2verts[:, :, :, 1] *= -1

        gamma = self.smpl_list['gamma'][0]
        scale = 8000 / 5000
        scale = torch.tensor(scale).cuda()
        gamma = torch.tensor(gamma).cuda()
        princpt = self.smpl_list['princpt'][0]
        deviation = [(princpt[0]-256)/256,(princpt[1]-256)/256,0]
        deviation = torch.Tensor(deviation).unsqueeze(0).unsqueeze(0).cuda()
        deviation = deviation * 0 
        princpt = torch.tensor([256,256],device="cuda:0", dtype=torch.float32)

        verts = torch.from_numpy(self.smpl_list["smplx_mesh_front"]).cuda()
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim_new(verts, self.faces, gamma, deviation, scale)

        self.next_front_to_first_t = self.render.cal_bc_transform(next_front_src_f2verts, src_fim, src_wim).detach().cpu()
        mask = (self.next_front_to_first_t[..., 0] >= -1) | (self.next_front_to_first_t[..., 1] >= -1)
        mask = mask.squeeze().cpu().numpy()
        self.f_mask = torch.from_numpy(mask)
        self.first_fim = src_fim.detach().cpu()
        self.first_wim = src_wim.detach().cpu()
        only_vis_f2pts = self.render.get_vis_f2pts(src_f2verts, src_fim)
        src_f2verts = src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        self.first_to_next_front_t = self.render.cal_bc_transform(src_f2verts, next_front_src_fim, next_front_src_wim).detach().cpu()
        
        t1 = self.render.cal_bc_transform(src_f2verts, src_fim, src_wim)
        mask = (t1[..., 0] >= -1) | (t1[..., 1] >= -1)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel_size = 5 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        self.first_mask = torch.from_numpy(thresh.astype(np.float32) / 255)
        self.first_f2verts = src_f2verts.detach().cpu()
        only_vis_f2pts = only_vis_f2pts[:, :, :, 0:2]
        only_vis_f2pts[:, :, :, 1] *= -1
        self.first_vis_f2verts = only_vis_f2pts.detach().cpu()
        verts = torch.from_numpy(self.smpl_list["smplx_mesh_back"]).cuda()
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim_new(verts, self.faces, gamma, deviation, scale)
        
        self.next_back_to_second_t = self.render.cal_bc_transform(next_back_src_f2verts, src_fim, src_wim).detach().cpu()
        mask = (self.next_back_to_second_t[..., 0] >= -1) | (self.next_back_to_second_t[..., 1] >= -1)
        mask = mask.squeeze().cpu().numpy()
        self.b_mask = torch.from_numpy(mask)

        self.second_fim = src_fim.detach().cpu()
        self.second_wim = src_wim.detach().cpu()
        only_vis_f2pts = self.render.get_vis_f2pts(src_f2verts, src_fim)
        src_f2verts = src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        self.second_to_next_back_t = self.render.cal_bc_transform(src_f2verts, next_back_src_fim, next_back_src_wim).detach().cpu()

        t2 = self.render.cal_bc_transform(src_f2verts, src_fim, src_wim)
        mask = (t2[..., 0] >= -1) | (t2[..., 1] >= -1)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel_size = 5  
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        self.second_mask = torch.from_numpy(thresh.astype(np.float32) / 255)
        self.second_f2verts = src_f2verts.detach().cpu()
        only_vis_f2pts = only_vis_f2pts[:, :, :, 0:2]
        only_vis_f2pts[:, :, :, 1] *= -1
        self.second_vis_f2verts = only_vis_f2pts.detach().cpu()
        
        shuru1 = rearrange(self.front_next_image_tensor.unsqueeze(0).cuda(), 'b h w c -> b c h w')
        shuru2 = rearrange(self.back_next_image_tensor.unsqueeze(0).cuda(), 'b h w c -> b c h w')
        qiandetupian = torch.nn.functional.grid_sample(shuru1, self.next_front_to_first_t.cuda())
        houdetupian = torch.nn.functional.grid_sample(shuru2, self.next_back_to_second_t.cuda())
        self.next_front_to_first_tensor = qiandetupian.detach().cpu()
        self.next_back_to_back_tensor = houdetupian.detach().cpu()
        self.read_meta()

    def read_meta(self):
        h = self.img_wh[1]
        w = self.img_wh[0]
        all_images_path = []
        self.ts_w = []
        self.all_images = []
        self.f2verts = []
        self.fim = []
        self.wim = []
        self.smpl = []

        # construct grid
        grid = np.indices((h, w)).astype(np.float32)
        # normalize
        grid[0,:,:] = grid[0,:,:] / h * 2 - 1
        grid[1,:,:] = grid[1,:,:] / w * 2 - 1

        grid = torch.from_numpy(rearrange(grid, 'c h w -> (h w) c'))
        grid_T = grid.clone()
        grid_T[...,1] = grid[...,0] 
        grid_T[...,0] = grid[...,1] 
        self.grid = grid_T
        # construct grid
        grid_hr = np.indices((h*2, w*2)).astype(np.float32)

        # normalize
        grid_hr[0,:,:] = grid_hr[0,:,:] / (2*h) * 2 - 1
        grid_hr[1,:,:] = grid_hr[1,:,:] / (2*w) * 2 - 1
        grid_hr = torch.from_numpy(rearrange(grid_hr, 'c h w -> (h w) c'))
        grid_T = grid_hr.clone()
        grid_T[...,1] = grid_hr[...,0] 
        grid_T[...,0] = grid_hr[...,1] 
        self.grid_h = grid_T

        warp_code = 1
        for input_image_path in sorted(glob.glob(f'{self.root_dir}/*')):
            all_images_path.append(input_image_path)
            self.ts_w.append(torch.Tensor([warp_code]).long())
            warp_code += 1
        if self.canonical_wh:
            h_c = self.canonical_wh[1]
            w_c = self.canonical_wh[0]
            grid_c = np.indices((h_c, w_c)).astype(np.float32)
            grid_c[0,:,:] = (grid_c[0,:,:] - (h_c - h) / 2) / h * 2 - 1
            grid_c[1,:,:] = (grid_c[1,:,:] - (w_c - w) / 2) / w * 2 - 1
            grid_c = torch.from_numpy(rearrange(grid_c, 'c h w -> (h w) c'))
            grid_c_T = grid_c.clone()
            grid_c_T[...,1] = grid_c[...,0] 
            grid_c_T[...,0] = grid_c[...,1] 
            self.grid_c = grid_c_T
        else:
            self.grid_c = self.grid
            self.canonical_wh = self.img_wh

        if self.mask_dir:
            self.all_masks = []

        if self.split == 'train' or self.split == 'val':
            if self.canonical_dir is not None:
                all_images_path_ = sorted(glob.glob(f'{self.canonical_dir}/*.png'))
                self.canonical_img = []
                for input_image_path in all_images_path_:
                    input_image = cv2.imread(input_image_path)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    input_image_tensor = torch.from_numpy(input_image).float() / 255
                    self.canonical_img.append(input_image_tensor)
                self.canonical_img = torch.stack(self.canonical_img, dim=0)

            for input_image_path in all_images_path:
                input_image = cv2.imread(input_image_path)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = cv2.resize(input_image, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
                input_image_tensor = torch.from_numpy(input_image).float() / 255
                self.all_images.append(input_image_tensor)
                if self.mask_dir:
                    input_image_name = input_image_path.split("/")[-1][:-4]
                    for i in range(len(self.mask_dir)):
                        input_mask = cv2.imread(f'{self.mask_dir[i]}/{input_image_name}.png')
                        input_mask = cv2.resize(input_mask, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
                        input_mask_tensor = torch.from_numpy(input_mask).float() / 255
                        self.all_masks.append(input_mask_tensor)
            
            for z in range(len(all_images_path)):
                gamma = self.smpl_list['gamma'][z]
                scale = self.smpl_list['focal'][z] / 5000
                scale = torch.from_numpy(scale).cuda()
                gamma = torch.tensor(gamma).cuda()
                princpt = self.smpl_list['princpt'][z]
                deviation = [(princpt[0]-256)/256,(princpt[1]-256)/256,0]
                deviation = torch.Tensor(deviation).unsqueeze(0).unsqueeze(0).cuda()
                verts = torch.from_numpy(self.smpl_list["smplx_mesh_cam"][z:z+1]).cuda()
                _, src_fim, src_wim = self.render.render_fim_wim_new(verts, self.faces, gamma, deviation, scale[0])

                self.fim.append(src_fim.detach().cpu())
                self.wim.append(src_wim.detach().cpu())

        if self.split == 'val':
            input_image = cv2.imread(all_images_path[0])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
            input_image_tensor = torch.from_numpy(input_image).float() / 255
            self.all_images.append(input_image_tensor)
            if self.mask_dir:
                input_image_name = all_images_path[0].split("/")[-1][:-4]
                for i in range(len(self.mask_dir)):
                    input_mask = cv2.imread(f'{self.mask_dir[i]}/{input_image_name}.png')
                    input_mask = cv2.resize(input_mask, (self.img_wh[0], self.img_wh[1]), interpolation = cv2.INTER_AREA)
                    input_mask_tensor = torch.from_numpy(input_mask).float() / 255
                    self.all_masks.append(input_mask_tensor)
        
        if self.split == 'val':
            self.ref_idx = 0

    def __len__(self):
        if self.test:
            return len(self.all_images)
        return len(self.all_images)

    def __getitem__(self, idx):
        if self.split == 'train' or self.split == 'val':
            idx = idx % len(self.all_images)
            sample = {'rgbs': self.all_images[idx],
                      'fim': self.fim[idx],
                      'wim': self.wim[idx],
                      'canonical_img': self.all_images[idx] if self.canonical_dir is None else self.canonical_img,
                      'ts_w': self.ts_w[idx],
                      'grid': self.grid,
                      'grid_h': self.grid_h,
                      'canonical_wh': self.canonical_wh,
                      'img_wh': self.img_wh,
                      'first_f2verts': self.first_f2verts,
                      'second_f2verts': self.second_f2verts,
                      'first_vis_f2verts': self.first_vis_f2verts,
                      'second_vis_f2verts': self.second_vis_f2verts,
                      'first_fim': self.first_fim,
                      'first_wim': self.first_wim,
                      'second_fim': self.second_fim,
                      'second_wim': self.second_wim,
                      'first_mask': self.first_mask,
                      'second_mask': self.second_mask,
                      'f_mask': self.f_mask,
                      'b_mask': self.b_mask,
                      'front_next_image_tensor': self.front_next_image_tensor,
                      'back_next_image_tensor': self.back_next_image_tensor,
                      'first_to_next_front_t': self.first_to_next_front_t,
                      'second_to_next_back_t': self.second_to_next_back_t,
                      'next_back_to_second_t': self.next_back_to_second_t,
                      'next_front_to_first_t': self.next_front_to_first_t,
                      'next_front_to_first_tensor': self.next_front_to_first_tensor,
                      'next_back_to_back_tensor': self.next_back_to_back_tensor,
                      'masks': self.all_masks[len(self.mask_dir)*idx:len(self.mask_dir)*idx+len(self.mask_dir)] if self.mask_dir else [torch.ones((self.img_wh[1], self.img_wh[0], 1))],
                      'grid_c': self.grid_c,
                      'reference': [self.all_images[self.ref_idx], self.all_masks[len(self.mask_dir)*idx:len(self.mask_dir)*idx+len(self.mask_dir)]] if not self.ref_idx is None else -1e5,
                      'seq_len': len(self.all_images) }

        return sample
