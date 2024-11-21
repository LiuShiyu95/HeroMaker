import os
import json
import cv2
import numpy as np

from einops import rearrange
from einops import repeat
from pathlib import Path
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import dataset_dict
from losses import loss_dict
from losses import compute_gradient_loss

from model.implicit_model import TranslationField
from model.implicit_model import ImplicitVideo
from model.implicit_model import ImplicitVideo_Hash
from model.implicit_model import Embedding
from model.implicit_model import AnnealedEmbedding
from model.implicit_model import AnnealedHash
from model.implicit_model import Deform_Hash3d_Warp

from PIL import Image
from util import get_optimizer, get_parameters
from util import get_scheduler
from util import get_learning_rate
from util import load_ckpt
from util import VideoVisualizer
import util.cv_utils as cv_utils
from opt import get_opts
from metrics import psnr

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from util.nmr import SMPLRenderer
import cv2
import sys
from scipy.ndimage import distance_transform_edt


def save_img(img, path):
    numpy_image = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    bgr_image = ((bgr_image+1) / 2 * 255).astype('uint8')

    cv2.imwrite(path, bgr_image)


def fill_invalid_with_nearest_valid(tensor1, tensor2, invalid_mask1, invalid_mask2, invalid_mask_full):
    arr1 = tensor1.detach().cpu().numpy()
    arr2 = tensor2.detach().cpu().numpy()
    invalid_mask1 = np.logical_or(invalid_mask1, invalid_mask_full)
    invalid_mask2 = np.logical_or(invalid_mask2, invalid_mask_full)
    invalid_mask2 = np.logical_or(invalid_mask2, np.logical_not(invalid_mask1))
    mask1 = np.logical_xor(invalid_mask1, invalid_mask_full)
    mask2 = np.logical_xor(invalid_mask2, invalid_mask_full)
    mask3 = np.logical_and(mask1, mask2)
    invalid_mask3 = np.logical_not(mask3) 

    invalid_mask1 = np.logical_and(invalid_mask3, invalid_mask1)

    out_mask1 = invalid_mask1
    out_mask2 = invalid_mask2
    combined_invalid_mask = invalid_mask2 & invalid_mask1

    distances, indices = distance_transform_edt(combined_invalid_mask, return_indices=True)

    # Calculate distances to both masks
    distances_mask1 = distance_transform_edt(invalid_mask1)
    distances_mask2 = distance_transform_edt(invalid_mask2)
    # Create a source mask (0 for mask1, 1 for mask2) based on minimum distance
    source_mask = (distances_mask1 >= distances_mask2).astype(int)

    filled_arr1 = arr1[tuple(indices)]
    filled_arr2 = arr2[tuple(indices)]
    
    filled_tensor1 = torch.from_numpy(filled_arr1).to(tensor1.device)
    filled_tensor2 = torch.from_numpy(filled_arr2).to(tensor2.device)
    source_mask = torch.from_numpy(source_mask).to(tensor2.device)
    return filled_tensor1, filled_tensor2, source_mask, out_mask1, out_mask2


def largest_component_mask(tensor):
    arr = tensor.detach().cpu().numpy()
    invalid_mask = (255 - (arr <= -1).any(axis=-1)[0] * 255)
    invalid_mask=invalid_mask.astype(np.uint8)
    _, thresh = cv2.threshold(invalid_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels == 1:
        invalid_mask = (arr <= -1).any(axis=-1)
        return invalid_mask
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component_mask = (labels == largest_component_index)
    largest_component_mask = np.logical_not(largest_component_mask)
    largest_component_mask = np.expand_dims(largest_component_mask, axis=0)

    return largest_component_mask


class ImplicitVideoSystem(LightningModule):
    def __init__(self, hparams):
        super(ImplicitVideoSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.color_loss = loss_dict['mse'](coef=1)
        if hparams.save_video:
            self.video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.raw_video_visualizer = VideoVisualizer(fps=hparams.fps)
            self.dual_video_visualizer = VideoVisualizer(fps=hparams.fps)

        self.models_to_train=[]
        self.embedding_xyz = Embedding(2, 8)
        self.embeddings = {'xyz': self.embedding_xyz}
        self.models = {}

        # Construct normalized meshgrid.
        h = self.hparams.img_wh[1]
        w = self.hparams.img_wh[0]
        self.h = h
        self.w = w
        if self.hparams.mask_dir:
            self.num_models = len(self.hparams.mask_dir) + 1
        else:
            self.num_models = 1
        
        # follow iPERDance: https://github.com/iPERDance/iPERCore/tree/main
        self.render = SMPLRenderer(map_name="uv_seg",
                              uv_map_path="assets/pretrains/mapper.txt",
                              tex_size=3,
                              image_size=h, fill_back=False,
                              anti_aliasing=True, background_color=(0, 0, 0), has_front=False).cuda()

        # Decide the number of deformable mlp.
        if hparams.encode_w:
            # No trainable parameters.
            self.embeddings['xyz_w'] = []
            assert (isinstance(self.hparams.N_xyz_w, list))
            in_channels_xyz = []
            for i in range(self.num_models):
                N_xyz_w = self.hparams.N_xyz_w[i]
                in_channels_xyz += [2 + 2 * N_xyz_w * 2]
                if hparams.annealed:
                    if hparams.deform_hash:
                        self.embedding_hash = AnnealedHash(
                        in_channels=2,
                        annealed_step=hparams.annealed_step,
                        annealed_begin_step=hparams.annealed_begin_step)
                        self.embeddings['aneal_hash'] = self.embedding_hash
                    else:
                        self.embedding_xyz_w = AnnealedEmbedding(
                            in_channels=2,
                            N_freqs=N_xyz_w,
                            annealed_step=hparams.annealed_step,
                            annealed_begin_step=hparams.annealed_begin_step)
                        self.embeddings['xyz_w'] += [self.embedding_xyz_w]
                else:
                    self.embedding_xyz_w = Embedding(2, N_xyz_w)
                    self.embeddings['xyz_w'] += [self.embedding_xyz_w]

            for i in range(self.num_models):
                embedding_w = torch.nn.Embedding(hparams.N_vocab_w, hparams.N_w)
                torch.nn.init.uniform_(embedding_w.weight, -0.05, 0.05)
                load_ckpt(embedding_w, hparams.weight_path, model_name=f'w_{i}')
                self.embeddings[f'w_{i}'] = embedding_w
                self.models_to_train += [self.embeddings[f'w_{i}']]

                # Add warping field mlp.
                if hparams.deform_hash:
                    with open('configs/hash.json') as f:
                        config = json.load(f)
                    warping_field = Deform_Hash3d_Warp(config=config)
                else:
                    warping_field = TranslationField(
                        D=self.hparams.deform_D,
                        W=self.hparams.deform_W,
                        in_channels_xyz=in_channels_xyz[i])

                load_ckpt(warping_field,
                          hparams.weight_path,
                          model_name=f'warping_field_{i}')
                
                self.models[f'warping_field_{i}'] = warping_field

        # Set up the canonical model.
        if hparams.canonical_dir is None:
            for i in range(self.num_models):
                if hparams.vid_hash:
                    with open('configs/hash.json') as f:
                        config = json.load(f)
                    implicit_video = ImplicitVideo_Hash(config=config)
                else:
                    implicit_video = ImplicitVideo(
                        D=hparams.vid_D,
                        W=hparams.vid_W,
                        sigmoid_offset=hparams.sigmoid_offset)
                load_ckpt(implicit_video, hparams.weight_path,
                            f'implicit_video_{i}')

                self.models[f'implicit_video_{i}'] = implicit_video

        for key in self.embeddings:
            setattr(self, key, self.embeddings[key])
        for key in self.models:
            setattr(self, key, self.models[key])
        self.models_to_train += [self.models]


    def deform_pts(self, ts_w, grid, encode_w, step=0, i=0):
        if hparams.deform_hash:
            ts_w_norm = ts_w / self.seq_len[0]
            repeat_size = int(grid.shape[0] / self.batch_size)
            ts_w_norm = ts_w_norm.repeat(1, repeat_size).reshape(-1,1)
            input_xyt = torch.cat([grid, ts_w_norm], dim=-1)
            if 'aneal_hash' in self.embeddings.keys():
                deform = self.models[f'warping_field_{i}'](
                    input_xyt,
                    step=step,
                    aneal_func=self.embeddings['aneal_hash'])
            else:
                deform = self.models[f'warping_field_{i}'](input_xyt)
            if encode_w:
                deformed_grid = deform + grid
            else:
                deformed_grid = grid
        else:
            if encode_w:
                e_w = self.embeddings[f'w_{i}'](repeat(ts_w, 'b n ->  (b l) n ',
                                                    l=grid.shape[0])[:, 0])
                # Whether to use annealed positional encoding.
                if self.hparams.annealed:
                    pe_w = self.embeddings['xyz_w'][i](grid, step)
                else:
                    pe_w = self.embeddings['xyz_w'][i](grid)

                # Warping field type.
                deform = self.models[f'warping_field_{i}'](torch.cat(
                    [e_w, pe_w], 1))
                deformed_grid = deform + grid
            else:
                deformed_grid = grid

        return deformed_grid, deform

    def forward(self,
                ts_w,
                grid,
                encode_w,
                step=0,
                rgbs=None,
                fim=None,
                wim=None,
                mk=None,
                idx=0,
                ):

        grid = rearrange(grid, 'b n c -> (b n) c')
        results_list = []
        l1_loss_list = []
        deform_list = []
        mk_12_list = []
        mask1_list = []
        mask2_list = []
        cano_f = []
        cano_b = []
        cano_f_h = []
        cano_b_h = []
        back = []

        for i in range(self.num_models-1):
            if i == 0:
                deformed_grid, deform = self.deform_pts(ts_w, grid, encode_w, step, i)  # [batch * num_pixels, 2]
                deformed_grid1, deform1 = self.deform_pts(ts_w, grid, encode_w, step, i+1)  # [batch * num_pixels, 2]
                deform_list.append(deformed_grid)
                
                t1 = self.render.cal_bc_transform(self.first_vis_f2verts, fim, wim)
                t2 = self.render.cal_bc_transform(self.second_vis_f2verts, fim, wim)
                t3 = self.render.cal_bc_transform(self.first_f2verts, fim, wim)
                t4 = self.render.cal_bc_transform(self.second_f2verts, fim, wim)
                arr3 = t3.detach().cpu().numpy()
                invalid_mask_full = (arr3 <= -1).any(axis=-1)

                largest_component_mask1 = largest_component_mask(t1)
                largest_component_mask1 = np.logical_or(largest_component_mask1, invalid_mask_full)
                largest_component_mask2 = largest_component_mask(t2)
                largest_component_mask2 = np.logical_or(largest_component_mask2, invalid_mask_full)


                filled_tensor1, filled_tensor2, source_mask, mask1, mask2 = fill_invalid_with_nearest_valid(t3, t4, largest_component_mask1, largest_component_mask2, invalid_mask_full)
                mask1_list.append(mask1)
                mask2_list.append(mask2)

                t1 = filled_tensor1
                t2 = filled_tensor2

                t1 = rearrange(t1,'b h w c ->(b h w) c')
                t2 = rearrange(t2,'b h w c ->(b h w) c')

                if idx > 202:
                    t1 = t1 + deform
                    t2 = t2 + deform1

                zero_tensor = torch.zeros_like(deform)
                l1_loss_0 = (deform, deform1, zero_tensor)
                l1_loss_list.append(l1_loss_0)

            else:
                deformed_grid, deform = self.deform_pts(ts_w, grid, encode_w, step, i+1)  # [batch * num_pixels, 2]
                zero_tensor = torch.zeros_like(deform)
                l1_loss_1 = (deform, zero_tensor)
                l1_loss_list.append(l1_loss_1)

            if not self.training and self.hparams.canonical_dir is not None:
                if i == 0:
                    w, h = self.img_wh
                    canonical_img = self.canonical_img.squeeze(0)
                    h_c, w_c = canonical_img.shape[1:3]
                    results1 = torch.nn.functional.grid_sample(
                    canonical_img[i:i + 1].permute(0, 3, 1, 2),
                    t1.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    )
                    results2 = torch.nn.functional.grid_sample(
                    canonical_img[i+1:i + 2].permute(0, 3, 1, 2),
                    t2.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    )

                    results1 = results1.squeeze().permute(1,0)
                    results2 = results2.squeeze().permute(1,0)
                    source_mask = rearrange(source_mask.unsqueeze(0),'c b h w -> (b h w) c') 
                    results = results2 * source_mask + results1 * (1-source_mask)
                else:
                    w, h = self.img_wh
                    canonical_img = self.canonical_img.squeeze(0)
                    h_c, w_c = canonical_img.shape[1:3]
                    results = torch.nn.functional.grid_sample(
                    canonical_img[i+1:i + 2].permute(0, 3, 1, 2),
                    deformed_grid.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    padding_mode='border'
                    )
                    results = results.squeeze().permute(1,0)

            else:
                if i == 0:

                    cano_result_f = self.models[f'implicit_video_{i}'](grid)
                    cano_result_b = self.models[f'implicit_video_{i+1}'](grid)
                    
                    cano_f.append(cano_result_f)
                    cano_b.append(cano_result_b)
                    
                    cano_result_f = rearrange(cano_result_f,'(b h w) c -> b c h w', b=self.batch_size,h=self.h,w=self.w)
                    cano_result_b = rearrange(cano_result_b,'(b h w) c -> b c h w', b=self.batch_size,h=self.h,w=self.w)
                    results1 = torch.nn.functional.grid_sample(
                    cano_result_f,
                    t1.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    )
                    results2 = torch.nn.functional.grid_sample(
                    cano_result_b,
                    t2.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    )
                    results1 = results1.squeeze().permute(1,0)
                    results2 = results2.squeeze().permute(1,0)
                    source_mask = rearrange(source_mask.unsqueeze(0),'c b h w -> (b h w) c')                    
                    results = results2 * source_mask + results1 * (1-source_mask)
                else:
                    back_results = self.models[f'implicit_video_{i+1}'](grid)
                    back.append(back_results)
                    back_results = rearrange(back_results,'(b h w) c -> b c h w', b=self.batch_size,h=self.h,w=self.w)
                    results = torch.nn.functional.grid_sample(
                    back_results,
                    deformed_grid.unsqueeze(1).unsqueeze(0),
                    mode='bilinear',
                    )
                    results = results.squeeze().permute(1,0)

            results_list.append(results)
        
        ret = edict(rgbs=results_list,
                    l1_loss = l1_loss_list,
                    deform=deform_list,
                    mk_12_list=mk_12_list,
                    mask1_list =mask1_list,
                    mask2_list=mask2_list,
                    cano_f=cano_f,
                    cano_b=cano_b,
                    back=back)
        return ret


    def setup(self, stage):
        if not self.hparams.test:
            dataset = dataset_dict["video_deform"]
            kwargs = {
                'root_dir': self.hparams.root_dir,
                'img_wh': tuple(self.hparams.img_wh),
                'mask_dir': self.hparams.mask_dir,
                'canonical_wh': self.hparams.canonical_wh,
                'ref_idx': self.hparams.ref_idx,
                'canonical_dir': self.hparams.canonical_dir,
                'name': self.hparams.name
            }
            self.train_dataset = dataset(split='train', **kwargs)
            self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        lr_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.optimizer], [lr_dict]

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          sampler=sampler,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

    def test_dataloader(self):
        dataset = dataset_dict["video_deform"]
        # dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'img_wh': tuple(self.hparams.img_wh),
            'mask_dir': self.hparams.mask_dir,
            'canonical_wh': self.hparams.canonical_wh,
            'canonical_dir': self.hparams.canonical_dir,
            'test': self.hparams.test,
            'name': self.hparams.name
        }
        self.train_dataset = dataset(split='train', **kwargs)
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time.
            pin_memory=True)

    def training_step(self, batch, batch_idx):

        self.batch_size = self.hparams.batch_size
        # Fetch training data.
        rgbs = batch['rgbs']
        fim = batch['fim'].squeeze(1)
        wim = batch['wim'].squeeze(1)
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        self.seq_len = batch['seq_len']
        self.first_f2verts = batch['first_f2verts'].squeeze(1)
        self.first_vis_f2verts = batch['first_vis_f2verts'].squeeze(1)
        self.second_f2verts = batch['second_f2verts'].squeeze(1)
        self.second_vis_f2verts = batch['second_vis_f2verts'].squeeze(1)
        self.first_mask = batch['first_mask']
        self.second_mask = batch['second_mask']
        self.next_front_to_first_tensor = batch['next_front_to_first_tensor'].squeeze(1)
        self.next_back_to_back_tensor = batch['next_back_to_back_tensor'].squeeze(1)
        self.f_mask = batch['f_mask'].unsqueeze(-1).repeat(1,1,1,3)
        self.b_mask = batch['b_mask'].unsqueeze(-1).repeat(1,1,1,3)
        fb_mask = [self.f_mask, self.b_mask]
        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')
        next_front_tensor = rearrange(self.next_front_to_first_tensor, 'b c h w -> (b h w) c')
        next_back_tensor = rearrange(self.next_back_to_back_tensor, 'b c h w -> (b h w) c')

        fb_rgb = [next_front_tensor, next_back_tensor]

        loss = 0
        ret = self.forward(ts_w,
                            grid,
                            self.hparams.encode_w,
                            self.global_step,
                            rgbs=rgbs,
                            fim=fim,
                            wim=wim,
                            mk=mk,
                            idx=batch_idx)
        cano_list = [ret.cano_f[0],ret.cano_b[0]]

        if batch_idx < 200:
            for i in range(2):
                results_p = cano_list[i]
                fb_rgbs_flattend = fb_rgb[i]

                mk_t = rearrange(fb_mask[i], 'b h w c -> (b h w) c')
                mk_t = mk_t.sum(dim=-1) > 0.05

                loss = loss + self.color_loss(results_p[mk_t],
                                                fb_rgbs_flattend[mk_t])
                # # PSNR metric.
                with torch.no_grad():
                    if i == 0:
                        psnr_ = psnr(results_p[mk_t], fb_rgbs_flattend[mk_t])

        else:
            cano_list = [ret.cano_f[0],ret.cano_b[0]]

            for i in range(self.num_models-1):
                results_p = cano_list[i]
                fb_rgbs_flattend = fb_rgb[i]

                mk_t = rearrange(fb_mask[i], 'b h w c -> (b h w) c')
                mk_t = mk_t.sum(dim=-1) > 0.05

                loss = loss + self.color_loss(results_p[mk_t],
                                                fb_rgbs_flattend[mk_t])


            for i in range(self.num_models-1):
                results = ret.rgbs[i]
                mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
                mk_t = mk_t.sum(dim=-1) > 0.05
                loss = loss + self.color_loss(results[mk_t],
                                                rgbs_flattend[mk_t])
                
                # Image gradient loss.
                img_pred = rearrange(results,
                                    '(b h w) c -> b h w c',
                                    b=self.batch_size,
                                    h=self.h,
                                    w=self.w)
                rgbs_gt = rearrange(rgbs_flattend,
                                    '(b h w) c -> b h w c',
                                    b=self.batch_size,
                                    h=self.h,
                                    w=self.w)

                mk_t_re = rearrange(mk_t,
                                    '(b h w c) -> b h w c',
                                    b=self.batch_size,
                                    h=self.h,
                                    w=self.w)
                grad_loss = compute_gradient_loss(rgbs_gt.permute(0, 3, 1, 2),
                                                img_pred.permute(0, 3, 1, 2),
                                                mask=mk_t_re.permute(0, 3, 1, 2))
                loss = loss + grad_loss * self.hparams.grad_loss
                
                if ret.l1_loss[0] != 0:
                    if i == 0:
                        loss = loss + 0.3 * torch.nn.functional.l1_loss(ret.l1_loss[i][0][mk_t], ret.l1_loss[i][2][mk_t])
                        loss = loss + 0.3 * torch.nn.functional.l1_loss(ret.l1_loss[i][1][mk_t], ret.l1_loss[i][2][mk_t])
                    else:
                        loss = loss + 0.3 * torch.nn.functional.l1_loss(ret.l1_loss[i][0][mk_t], ret.l1_loss[i][1][mk_t])

                # # PSNR metric.
                with torch.no_grad():
                    if i == 0:
                        psnr_ = psnr(results[mk_t], rgbs_flattend[mk_t])
    
        self.log('lr', get_learning_rate(self.optimizer), prog_bar=True)
        self.log('loss', loss, prog_bar=True)
        self.log('psnr', psnr_, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        rgbs = batch['rgbs']
        fim = batch['fim']
        wim = batch['wim']
        ts_w = batch['ts_w']
        grid = batch['grid']
        mk = batch['masks']
        grid_c = grid  # batch['grid_c']
        self.seq_len = batch['seq_len']
        ret = self(ts_w, grid, self.hparams.encode_w, self.global_step, fim=fim, wim=wim)
        ret_c = self(ts_w, grid_c, False, self.global_step, fim=fim, wim=wim)

        log = {}
        W, H = self.hparams.img_wh

        rgbs_flattend = rearrange(rgbs, 'b h w c -> (b h w) c')
        img_gt = rgbs_flattend.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        stack_list = [img_gt]
        for i in range(self.num_models):
            results = ret.rgbs[i]
            results_c = ret_c.rgbs[i]
            mk_t = rearrange(mk[i], 'b h w c -> (b h w) c')
            if batch_idx == 0:
                results[mk_t.sum(dim=-1) <= 0.05] = 0
                img = results.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                img_c = results_c.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                stack_list.append(img)
                stack_list.append(img_c)

        stack = torch.stack(stack_list) # (3, 3, H, W)
        self.logger.experiment.add_images('val/GT_Reconstructed', stack,
                                          self.global_step)

        return log

    def test_step(self, batch, batch_idx):
        self.batch_size = 1
        ts_w = batch['ts_w']
        fim = batch['fim'].squeeze(1)
        wim = batch['wim'].squeeze(1)
        grid = batch['grid']
        mk = batch['masks']
        grid_c = batch['grid_c']
        W, H = self.hparams.img_wh
        self.seq_len = batch['seq_len']
        self.first_mask = batch['first_mask']
        self.second_mask = batch['second_mask']
        self.first_f2verts = batch['first_f2verts'].squeeze(1)
        self.second_f2verts = batch['second_f2verts'].squeeze(1)
        self.first_vis_f2verts = batch['first_vis_f2verts'].squeeze(1)
        self.second_vis_f2verts = batch['second_vis_f2verts'].squeeze(1)
        if self.hparams.canonical_dir is not None:
            self.canonical_img = batch['canonical_img']
            self.img_wh = batch['img_wh']

        save_dir = os.path.join('results',
                                self.hparams.root_dir.split('/')[0],
                                self.hparams.root_dir.split('/')[1],
                                self.hparams.exp_name)
        sample_name = self.hparams.root_dir.split('/')[1]
        if self.hparams.canonical_dir is not None:
            test_dir = f'{save_dir}_transformed'
            video_name = f'{sample_name}_{self.hparams.exp_name}_transformed'
        else:
            test_dir = f'{save_dir}'
            video_name = f'{sample_name}_{self.hparams.exp_name}'
        Path(test_dir).mkdir(parents=True, exist_ok=True)

        if batch_idx > 0 and self.hparams.save_video:
            self.video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}.mp4'))
            self.raw_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_raw.mp4'))
            self.dual_video_visualizer.set_path(os.path.join(
                test_dir, f'{video_name}_dual.mp4'))

        if batch_idx == 0 and self.hparams.canonical_dir is None:
            # Save the canonical image.
            ret = self(ts_w, grid_c, False, self.global_step, fim=fim, wim=wim,mk=mk, idx=10000)
            
            cano_results_f = ret.cano_f[0]
            cano_results_b = ret.cano_b[0]
            back = ret.back[0]

            img_c_f = cano_results_f.view(H, W, 3).float().cpu().numpy()
            img_c = cv2.cvtColor(img_c_f, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{test_dir}/canonical_0.png', img_c * 255)
            img_c_b = cano_results_b.view(H, W, 3).float().cpu().numpy()
            img_c = cv2.cvtColor(img_c_b, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{test_dir}/canonical_1.png', img_c * 255)
            img_b = back.view(H, W, 3).float().cpu().numpy()
            img_c = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{test_dir}/canonical_2.png', img_c * 255)

        ret_n = self(ts_w, 
                     grid, 
                     self.hparams.encode_w, 
                     self.global_step, 
                     fim = fim,
                     wim = wim,
                     mk=mk, idx=10000)


        img = np.zeros((H * W, 3), dtype=np.float32)

        for i in range(self.num_models-1):
            if batch_idx == 0 and self.hparams.canonical_dir is None:
                results_c = ret.rgbs[i]

                if self.hparams.canonical_wh:
                    img_c = results_c.view(self.hparams.canonical_wh[1],
                                           self.hparams.canonical_wh[0],
                                           3).float().cpu().numpy()
                else:
                    img_c = results_c.view(H, W, 3).float().cpu().numpy()

                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'{test_dir}/canonical_w_{i}.png', img_c * 255)
            mk_n = rearrange(mk[i], 'b h w c -> (b h w) c')
            mk_n = mk_n.sum(dim=-1) > 0.05
            mk_n = mk_n.cpu().numpy()
            results = ret_n.rgbs[i]
            results = results.cpu().numpy()  # (3, H, W)
            img[mk_n] = results[mk_n]

        img = rearrange(img, '(h w) c -> h w c', h=H, w=W)
        img = img * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{test_dir}/{batch_idx:05d}.png', img)

        if batch_idx > 0 and self.hparams.save_video:
            img = img[..., ::-1]
            self.video_visualizer.add(img)
            rgbs = batch['rgbs'].view(H, W, 3).cpu().numpy() * 255
            rgbs = rgbs.astype(np.uint8)
            self.raw_video_visualizer.add(rgbs)
            wim = wim.squeeze().float().cpu().numpy() * 255
            wim = wim.astype(np.uint8)
            dual_img = np.concatenate((rgbs, img, wim), axis=1)
            self.dual_video_visualizer.add(dual_img)

        if self.hparams.save_deform:
            save_deform_dir = f'{test_dir}_deform'
            Path(save_deform_dir).mkdir(parents=True, exist_ok=True)
            deformation_field = ret_n.deform[0]
            deformation_field = rearrange(deformation_field,
                                          '(h w) c -> h w c', h=H, w=W)
            grid_ = rearrange(grid[0], '(h w) c -> h w c', h=H, w=W)
            deformation_delta = deformation_field - grid_
            np.save(f'{save_deform_dir}/{batch_idx:05d}.npy',
                    deformation_delta.cpu().numpy())

    def on_test_epoch_end(self):
        if self.hparams.save_video:
            self.video_visualizer.save()
            self.raw_video_visualizer.save()
            self.dual_video_visualizer.save()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

def main(hparams):
    system = ImplicitVideoSystem(hparams)

    if not hparams.test:
        os.makedirs(f'{hparams.model_save_path}/{hparams.exp_name}',
                    exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{hparams.model_save_path}/{hparams.exp_name}',
        filename='{step:d}',
        mode='max',
        save_top_k=-1,
        every_n_train_steps=hparams.save_model_iters,
        save_last=True)

    logger = TensorBoardLogger(save_dir=hparams.log_save_path,
                               name=hparams.exp_name)

    trainer = Trainer(max_steps=hparams.num_steps,
                      precision=16 if hparams.vid_hash == True else 32,
                    #   precision=16,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      accelerator='gpu',
                      devices=hparams.gpus,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if len(hparams.gpus) == 1 else None,
                      val_check_interval=hparams.valid_iters,
                      limit_val_batches=hparams.valid_batches,
                      strategy="ddp_find_unused_parameters_true")
                    # )
    if hparams.test:
        trainer.test(system, dataloaders=system.test_dataloader())
    else:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
