import numpy as np
import smplx
from tqdm import tqdm
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os.path as osp
import copy
from PIL import Image
import pyrender
import trimesh
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
sys.path.insert(0, os.path.join('..', 'main'))
sys.path.insert(0, "../../osx_estimator")

from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
import pytorch3d.transforms as transforms
from pytorch3d.ops import laplacian
from pytorch3d.ops import cot_laplacian

import scipy.sparse as sp
from common.utils.human_models import smpl_x
from common.utils.vis import render_mesh
from common.utils.vis import render_mesh_white
from common.utils.preprocessing import load_img

openpose_idx = [8, 12, 9, 13, 10, 14, 11, 1, 5 ,2, 6, 3, 7 ,4 ,19 ,20 ,21, 22, 23, 24, 18, 17, 16, 15, 0]
smpl_pose_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

def optimize(optimizer, closure, max_iter=10):
    pbar = tqdm(range(max_iter))
    for i in pbar:
        loss = optimizer.step(closure)
        pbar.set_postfix_str(f"loss: {loss.detach().cpu().numpy():.6f}")



def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N


def relative_edge_loss(vertices1_edge, vertices2_edge, lossfunc=torch.nn.functional.mse_loss):
    # edge1 = vertices1_edge[:, 0] - vertices1_edge[:, 1]
    # edge2 = vertices2_edge[:, 0] - vertices2_edge[:, 1]
    edge1 = vertices1_edge.float()
    edge2 = vertices2_edge.float()
    
    # edges_for = lambda x: x[vertices1_edge[:, 0], :] - x[vertices1_edge[:, 1], :]
    return lossfunc(edge1, edge2)


smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
focal = (5000, 5000)
input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)
input_img_shape = (512, 384)

def get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, delta=None, pose_type="posed"):

    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
    output = smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, delta=delta,
                                reye_pose=zero_pose, expression=expr, pose_type=pose_type)
    
    source_mesh = output.vertices
    mesh_cam = output.vertices

    joint_cam = output.joints[:, smpl_x.joint_idx, :]

    x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        focal[0] + princpt[0]
    y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
        focal[1] + princpt[1]
    x = x / input_body_shape[1] * output_hm_shape[2]
    y = y / input_body_shape[0] * output_hm_shape[1]
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

    return joint_proj, joint_cam, mesh_cam, source_mesh

DEVICE = "cuda"
keypoints_threshold = 0.5

@torch.no_grad()
def main(args):

    name = args.name

    smpl_list = np.load(f"{name}_osx_smpl.npz")
    bb2img_trans = smpl_list['bb2img_trans']
    focal_xin = torch.from_numpy(smpl_list['focal']).cuda()
    princpt_xin = torch.from_numpy(smpl_list['princpt']).cuda()
    smplx_root_pose = torch.from_numpy(smpl_list['smplx_root_pose']).cuda()
    smplx_body_pose = torch.from_numpy(smpl_list['smplx_body_pose']).cuda()
    smplx_lhand_pose = torch.from_numpy(smpl_list['smplx_lhand_pose']).cuda()
    smplx_rhand_pose = torch.from_numpy(smpl_list['smplx_rhand_pose']).cuda()
    smplx_jaw_pose = torch.from_numpy(smpl_list['smplx_jaw_pose']).cuda()
    smplx_shape = torch.from_numpy(smpl_list['smplx_shape']).cuda()
    smplx_expr = torch.from_numpy(smpl_list['smplx_expr']).cuda()
    cam_trans = torch.from_numpy(smpl_list['cam_trans']).cuda()

    src_path = f'../../../vudeo_reconstruction/all_sequences/{name}/{name}/'
    image_path = sorted(glob.glob(f'{src_path}/*'))
    bb2img_trans = torch.from_numpy(bb2img_trans).cuda()

    # load 2d keypoints information
    keypoints_2d = np.load(f"{root}/{name}_pose_true.npy")
    keypoints_2d = torch.from_numpy(keypoints_2d).float().to(DEVICE)
    keypoints_2d = keypoints_2d.squeeze()
    
    params = {}
    params["smplx_root_pose"] = nn.Parameter(smplx_root_pose.clone())
    params["smplx_body_pose"] = nn.Parameter(smplx_body_pose.clone())
    params["smplx_lhand_pose"] = nn.Parameter(smplx_lhand_pose.clone())
    params["smplx_rhand_pose"] = nn.Parameter(smplx_rhand_pose.clone())
    params["smplx_jaw_pose"] = nn.Parameter(smplx_jaw_pose.clone())
    params["smplx_shape"] = nn.Parameter(smplx_shape.clone())    
    params["smplx_expr"] = nn.Parameter(smplx_expr.clone())
    params["smplx_delta"] = nn.Parameter(torch.zeros([smplx_body_pose.shape[0], 10475, 3], dtype=torch.float32, device="cuda:0"))

    # optimize with keypoints
    optimizer = torch.optim.Adam(params.values(), lr=1e-4)

    def closure():
        optimizer.zero_grad()

        joint_proj, joint_cam, mesh_cam, yuan_mesh = get_coord(params["smplx_root_pose"], params["smplx_body_pose"], params["smplx_lhand_pose"],
                                                                params["smplx_rhand_pose"], params["smplx_jaw_pose"], params["smplx_shape"], params["smplx_expr"], cam_trans, delta=None, pose_type="posed")
        joint_proj[:, openpose_idx, :] = joint_proj[:, smpl_pose_idx, :]
        for i in range(joint_proj.shape[0]):

            joint_proj_i = joint_proj[i, :, :]
            bb2img_trans_i = bb2img_trans[i, :, :]
            joint_proj_i[:, 0] = joint_proj_i[:, 0] / output_hm_shape[2] * input_img_shape[1]
            joint_proj_i[:, 1] = joint_proj_i[:, 1] / output_hm_shape[1] * input_img_shape[0]
            
            ones_column = torch.ones(joint_proj_i.shape[0], 1, dtype=joint_proj.dtype, device=joint_proj.device)
            joint_proj_i = torch.cat((joint_proj_i, ones_column), dim=1)
            joint_proj_i = torch.matmul(bb2img_trans_i, joint_proj_i.t()).t()
            joint_proj[i, :, :] = joint_proj_i

        joint_proj = joint_proj[:,:25,:]
        error = (keypoints_2d[..., :2] - joint_proj).square().sum(-1).sqrt()        
        m1 = (keypoints_2d[..., 2] > keypoints_threshold)
        mask = m1
        error = error * mask.float() * 5
        loss = error.mean()
        reg = (yuan_mesh[1:] - yuan_mesh[:-1]).square().sum(-1).sqrt()
        loss += reg.mean() 
        loss.backward()
        return loss
    optimize(optimizer, closure, max_iter=200)   

    smplx_root_pose = params["smplx_root_pose"]
    smplx_body_pose = params["smplx_body_pose"]
    smplx_lhand_pose = params["smplx_lhand_pose"]
    smplx_rhand_pose = params["smplx_rhand_pose"]
    smplx_jaw_pose = params["smplx_jaw_pose"]
    smplx_shape = params["smplx_shape"]
    smplx_expr = params["smplx_expr"]

    # Stage 1: Use only mask to constrain without deforming
    masks = sorted(glob.glob(f"../../../vudeo_reconstruction/all_sequences/{name}/{name}_masks_0/*"))
    masks = [cv2.imread(p)[..., 0] for p in masks]
    masks = np.stack(masks, axis=0)

    for i in range(len(masks)):
        mask = torch.from_numpy(masks[i:i+1]).float().to(DEVICE) / 255
        optimizer = torch.optim.Adam(params.values(), lr=5e-4)
        def closure():
            optimizer.zero_grad()
            joint_proj, joint_cam, mesh_cam, yuan_mesh = get_coord(params["smplx_root_pose"][i].unsqueeze(0), params["smplx_body_pose"][i].unsqueeze(0), params["smplx_lhand_pose"][i].unsqueeze(0),
                                                                            params["smplx_rhand_pose"][i].unsqueeze(0), params["smplx_jaw_pose"][i].unsqueeze(0), params["smplx_shape"][i].unsqueeze(0),
                                                                            params["smplx_expr"][i].unsqueeze(0), cam_trans[i].unsqueeze(0), delta=None, pose_type="posed")
            verts = mesh_cam
            verts[:,:,0] = -verts[:,:,0]
            verts[:,:,1] = -verts[:,:,1]
            faces = smplx_layer.faces_tensor[None].repeat(1, 1, 1)

            meshes = Meshes(
                verts=verts,
                faces = faces
            )
            R = torch.eye(3, device=DEVICE)[None]
            t = torch.zeros(1, 3, device=DEVICE)

            cameras = PerspectiveCameras(
                focal_length=focal_xin[i].float().unsqueeze(0),
                principal_point=princpt_xin[i].float().unsqueeze(0),
                R=R,
                T=t,
                image_size=((512,512),),
                in_ndc=False,
                device=DEVICE,
            )
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

            raster_settings = RasterizationSettings(
                image_size=((512,512)),
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma  *0.005,
                faces_per_pixel=150,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(
                    blend_params=blend_params
                )
            )
            out = renderer(meshes)
            silhouette = out[..., 3]
            loss_silhouette = F.mse_loss(silhouette, mask)
            loss = loss_silhouette

            loss.backward()
            return loss
 
        optimize(optimizer, closure, max_iter=20)

    masks = sorted(glob.glob(f"/public/home/liushy/new_codef/CoDeF/all_sequences/{name}/{name}_masks_0/*"))
    masks = [cv2.imread(p)[..., 0] for p in masks]
    masks = np.stack(masks, axis=0)
    for i in range(len(masks)):
        mask = torch.from_numpy(masks[i:i+1]).float().to(DEVICE) / 255
        optimizer = torch.optim.Adam(params.values(), lr=5e-4)
        def closure():
            optimizer.zero_grad()
            joint_proj, joint_cam, mesh_cam, yuan_mesh = get_coord(smplx_root_pose[i].unsqueeze(0), smplx_body_pose[i].unsqueeze(0), smplx_lhand_pose[i].unsqueeze(0),
                                                                            smplx_rhand_pose[i].unsqueeze(0), smplx_jaw_pose[i].unsqueeze(0), smplx_shape[i].unsqueeze(0),
                                                                            smplx_expr[i].unsqueeze(0), cam_trans[i].unsqueeze(0), delta=params["smplx_delta"][i].unsqueeze(0), pose_type="posed")
            verts = mesh_cam
            verts[:,:,0] = -verts[:,:,0]
            verts[:,:,1] = -verts[:,:,1]
            faces = smplx_layer.faces_tensor[None].repeat(1, 1, 1)
            meshes = Meshes(
                verts=verts,
                faces = faces
            )

            mls_loss = mesh_laplacian_smoothing(meshes)

            R = torch.eye(3, device=DEVICE)[None]
            t = torch.zeros(1, 3, device=DEVICE)

            cameras = PerspectiveCameras(
                focal_length=focal_xin[i].float().unsqueeze(0),
                principal_point=princpt_xin[i].float().unsqueeze(0),
                R=R,
                T=t,
                image_size=((512,512),),
                in_ndc=False,
                device=DEVICE,
            )
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

            raster_settings = RasterizationSettings(
                image_size=((512,512)),
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma  *0.005,
                faces_per_pixel=150,
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(
                    blend_params=blend_params
                )
            )
            out = renderer(meshes)
            silhouette = out[..., 3]
            loss_silhouette = F.mse_loss(silhouette, mask)
            min_deform_loss = torch.sum(params["smplx_delta"][i].unsqueeze(0)**2) / 10475
            loss = loss_silhouette  + 0.5 * mls_loss  +  10 * min_deform_loss
            loss.backward()
            return loss
 
        optimize(optimizer, closure, max_iter=20)
    
    out_params = dict(smpl_list)
    out_params['smplx_mesh_cam'] = mesh_cam.detach().cpu().numpy()
    

    mean_tensor = torch.mean(params["smplx_delta"], dim=0, keepdim=True)
    
    _, _, mesh_cam, _ = get_coord(smplx_root_pose[0].unsqueeze(0), smplx_body_pose[0].unsqueeze(0), smplx_lhand_pose[0].unsqueeze(0), smplx_rhand_pose[0].unsqueeze(0), smplx_jaw_pose[0].unsqueeze(0),
                                                            smplx_shape[0].unsqueeze(0), smplx_expr[0].unsqueeze(0), cam_trans[0].unsqueeze(0), delta=mean_tensor, pose_type='a-pose-front')
    out_params['smplx_mesh_front'] = mesh_cam.detach().cpu().numpy()
    
    _, _, mesh_cam, _ = get_coord(smplx_root_pose[0].unsqueeze(0), smplx_body_pose[0].unsqueeze(0), smplx_lhand_pose[0].unsqueeze(0), smplx_rhand_pose[0].unsqueeze(0), smplx_jaw_pose[0].unsqueeze(0),
                                                        smplx_shape[0].unsqueeze(0), smplx_expr[0].unsqueeze(0), cam_trans[0].unsqueeze(0), delta=mean_tensor, pose_type='a-pose-back')
    out_params['smplx_mesh_back'] = mesh_cam.detach().cpu().numpy()
    np.savez(f'./{name}_out_data_deform.npz', **out_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimization script")
    parser.add_argument("--name", type=str, required=True, help="Name for the dataset and output files")
    args = parser.parse_args()
    main(args)