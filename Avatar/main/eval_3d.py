# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import json
from config import cfg
from utils.uhm import uhm
from utils.transforms import transform_joint_to_other_db
from pytorch3d.io import save_obj, load_ply
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import knn_points, knn_gather

def load_joint(path):
    joint_world = np.ones((21,3),dtype=np.float32)
    with open(path) as f:
        for line in f:
            parsed_line = line.split()
            parsed_line = [float(x) for x in parsed_line]
            joint_idx, x_world, y_world, z_world, score_sum, view_num = parsed_line
            joint_idx = int(joint_idx) 

            if joint_idx > 20:
                continue

            joint_world[joint_idx] = np.array([x_world, y_world, z_world])
     
    return joint_world

# distance between mesh and recon
class Point2PointLoss(nn.Module):
    def __init__(self, vertex_num, face):
        super(Point2PointLoss, self).__init__()
        self.vertex_num = vertex_num
        self.face = face
        
    def mesh2recon(self, meshes, recons, recons_faces):
        batch_size = meshes.shape[0]
        
        # get closest point
        recon_vertex_num = [len(recon) for recon in recons]
        recon_max_vertex_num = max(recon_vertex_num)
        recons_padded = torch.stack([torch.cat((recon, torch.ones((recon_max_vertex_num-recon.shape[0],3)).float().cuda().fill_(9999))) for recon in recons])
        points = knn_points(meshes, recons_padded, lengths2=torch.LongTensor(recon_vertex_num).cuda(), K=1, return_nn=True)
        
        # if dot product of normals are smaller than threshold, discard
        meshes_normals = Meshes(verts=meshes, faces=torch.LongTensor(self.face).cuda()[None,:,:].repeat(batch_size,1,1)).verts_normals_packed().reshape(batch_size,self.vertex_num,3)
        recons_normals = Meshes(verts=recons, faces=recons_faces).verts_normals_packed()
        recons_normals_padded = []; cnt = 0;
        for i in range(batch_size):
            vertex_num = len(recons[i])
            normal = recons_normals[cnt:cnt+vertex_num]
            recons_normals_padded.append(torch.cat((normal, torch.ones((recon_max_vertex_num-vertex_num,3)).float().cuda().fill_(9999))))
            cnt += vertex_num
        recons_normals_padded = torch.stack(recons_normals_padded)
        recons_normals = knn_gather(recons_normals_padded, points.idx, torch.LongTensor(recon_vertex_num).cuda())[:,:,0,:]
        normal_mask = (torch.sum(meshes_normals * recons_normals, 2) >= 0.45).float()[:,:,None]
        
        # if too far, discard
        dist_mask = (torch.sqrt(torch.sum((meshes - points.knn[:,:,0,:])**2,2)) <= 0.005).float()[:,:,None]
         
        loss = torch.abs((meshes - points.knn[:,:,0,:]) * recons_normals) * normal_mask * dist_mask 
        return loss
    
    def recon2mesh(self, joints, meshes, recons):
        batch_size = meshes.shape[0]

        # get closest point
        recon_vertex_num = [len(recon) for recon in recons]
        recon_max_vertex_num = max(recon_vertex_num)
        recons_padded = torch.stack([torch.cat((recon, torch.ones((recon_max_vertex_num-recon.shape[0],3)).float().cuda().fill_(9999))) for recon in recons])
        points = knn_points(recons_padded, meshes, lengths1=torch.LongTensor(recon_vertex_num).cuda(), K=1, return_nn=True)

        # cut padded dummy vertices
        is_padded = []
        for i in range(batch_size):
            vertex_num = len(recons[i])
            is_padded.append(torch.cat((torch.zeros((vertex_num)), torch.ones((recon_max_vertex_num-vertex_num)))).float().cuda())
        is_padded = torch.stack(is_padded)[:,:,None]

        # if too far, discard
        dist_mask = (torch.sqrt(torch.sum((recons_padded - points.knn[:,:,0,:])**2,2)) <= 0.005).float()[:,:,None]

        loss = torch.abs(recons_padded - points.knn[:,:,0,:]) * (1 - is_padded) * dist_mask
        return loss

    def forward(self, joints, meshes, recons, recons_faces):
        mesh2recon = self.mesh2recon(meshes, recons, recons_faces)
        recon2mesh = self.recon2mesh(joints, meshes, recons)
        loss = mesh2recon.mean(1) + recon2mesh.mean(1)
        return loss 


# PointFaceDistance
_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3
from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.
            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`
        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, idxs):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None

# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
point_face_distance = _PointFaceDistance.apply

def point_mesh_face_distance(
    meshes,
    pcls,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`
    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.
    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.
    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.
    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)


    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, face_idxs = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    point_to_face = torch.sqrt(point_to_face) # squared Euclidean distance -> Euclidean distance

    # distance thresholding
    mask = point_to_face < 5

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)

    # mask out forearm vertices
    face = meshes.faces_list()[0]
    face_idxs = face_idxs - point_to_cloud_idx * face.shape[0] # packed -> unpacked
    vertex_idxs = face[face_idxs,:].view(-1,3)
    is_forearm = torch.FloatTensor(np.argmax(uhm_layer.skinning_weight, 1) == uhm.joints_name.index('Forearm')).cuda()
    is_forearm = (is_forearm[vertex_idxs[:,0]] + is_forearm[vertex_idxs[:,1]] + is_forearm[vertex_idxs[:,2]]) > 0
    mask = mask * (1 - is_forearm.float())

    out = []
    for i in range(N):
        dist = point_to_face[point_to_cloud_idx==i] * mask[point_to_cloud_idx==i]
        dist = torch.sum(dist) / torch.sum(mask[point_to_cloud_idx==i])
        out.append(float(dist))
    return out

def prepare_eval(wrist, meshes, recons):
    pcl = []
    batch_size = wrist.shape[0]
    for i in range(batch_size):
        # cut vertices too far from mesh
        mesh_center = torch.mean(meshes[i],0)[None,:]
        dist_from_mesh = torch.sqrt(torch.sum((recons[i] - mesh_center)**2,1))
        dist_mask = (dist_from_mesh[:,None] < 0.15).float()

        # cut 3D recons below the wrist
        root_joint = wrist[i,None,:]
        mesh_center = torch.mean(meshes[i], 0)[None,:]
        root_mask = (((mesh_center - root_joint) * (recons[i] - root_joint)).sum(1) > 0).float()[:,None]

        mask = (dist_mask * root_mask) > 0
        pcl.append(recons[i][mask.repeat(1,3)].view(-1,3) * 1000) # meter to millimeter

    pcls = Pointclouds(pcl)
    return pcls

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    args = parser.parse_args()
 
    assert args.subject_id, "Please set subject id."
    return args

args = parse_args()
subject_id = args.subject_id

if subject_id == 'AXE977':
    frame_idx = 2892
elif subject_id == 'QVC422':
    frame_idx = 3678
elif subject_id == 'QZX685':
    frame_idx = 2847
elif subject_id == 'XKT970':
    frame_idx = 3116

# argument parse and create log
args = parse_args()
cfg.set_args('test', args.subject_id, False)

# load target keypoint
joint_path = osp.join('..', 'data', 'Ours', 'data', subject_id, 'geo_3d_for_eval', '%06d.3d' % frame_idx)
joint_gt = load_joint(joint_path)
joints_name_mugsy = ('Thumb_4', 'Thumb_3', 'Thumb_2', 'Thumb_1', 'Index_4', 'Index_3', 'Index_2', 'Index_1', 'Middle_4', 'Middle_3', 'Middle_2', 'Middle_1', 'Ring_4', 'Ring_3', 'Ring_2', 'Ring_1', 'Pinky_4', 'Pinky_3', 'Pinky_2', 'Pinky_1', 'Wrist')
joint_gt = transform_joint_to_other_db(joint_gt, joints_name_mugsy, uhm.joints_name)
joint_gt = torch.FloatTensor(joint_gt).cuda()[None,:,:] / 1000 # millimeter to meter

# load taget recon
recon_vert, recon_face = load_ply(osp.join('..', 'data', 'Ours', 'data', subject_id, 'geo_3d_for_eval', 'mesh%06d.ply' % frame_idx))
recon_vert, recon_face = recon_vert.cuda()[None,:,:] / 1000, recon_face.cuda()[None,:,:] # millimeter to meter

# get rigid alignment matrix for the initialization
rigid_align_joint_idxs = [i for i in range(uhm.joint_num) if uhm.joints_name[i] in ['Wrist', 'Index_1', 'Middle_1', 'Ring_1', 'Pinky_1']]
joint_init = torch.FloatTensor(uhm.joint).cuda()[None,:,:]
RTs = corresponding_points_alignment(joint_init[:,rigid_align_joint_idxs,:], joint_gt[:,rigid_align_joint_idxs,:])
R = RTs.R.permute(0,2,1).detach()
t = RTs.T.detach()

# UHM and loss functions
uhm_layer = uhm.layer.cuda()
p2p_loss = Point2PointLoss(uhm.vertex_num, uhm.face)

# optimize features to get mesh_gt
with open(osp.join(cfg.result_dir, 'id_code.json')) as f:
    id_code = torch.FloatTensor(json.load(f)).view(1,-1).cuda() # id code from the phone capture. do not optimize this.
root_pose = nn.Parameter(R)
hand_pose = nn.Parameter(torch.eye(3)[None,None,:,:].repeat(1,uhm.joint_num-1,1,1).float().cuda())
trans = nn.Parameter(t)
lr = 1e-2
optimizer = torch.optim.Adam([root_pose, hand_pose, trans], lr=lr)
for itr in range(200):
    if itr == 170:
        for g in optimizer.param_groups:
            g['lr'] = lr / 10
    optimizer.zero_grad()
    
    mesh_out, joint_out = uhm_layer(root_pose, hand_pose, id_code, trans)

    loss = {}
    loss['joint'] = torch.abs(joint_out - joint_gt)
    loss['p2p'] = p2p_loss(joint_out, mesh_out, recon_vert, recon_face) * 10
    
    loss = {k:loss[k].mean() for k in loss}
    sum(loss[k] for k in loss).backward()
    optimizer.step()
mesh_out = mesh_out.detach()

meshes = Meshes(mesh_out*1000, torch.LongTensor(uhm.face).cuda()[None,:,:]) # meter to millimeter
pcls = prepare_eval(joint_gt[:,uhm.joints_name.index('Wrist')], mesh_out, recon_vert)
err = float(point_mesh_face_distance(meshes, pcls)[0])

# save
recon_vert, recon_face,  mesh_out = recon_vert[0].cpu(), recon_face[0].cpu(), mesh_out[0].cpu()
save_path = osp.join('./eval_results_3d', subject_id)
os.makedirs(save_path, exist_ok=True)
save_obj(osp.join(save_path, 'recon.obj'), recon_vert, recon_face)
save_obj(osp.join(save_path, 'uhm.obj'), mesh_out, torch.LongTensor(uhm.face))
with open(osp.join(save_path, 'err'), 'w') as f:
    f.write(str(err) + ' mm')
print('Error of ' + subject_id + ': ' + str(err) + ' mm')

