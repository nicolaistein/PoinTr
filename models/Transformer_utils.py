##############################################################
# % Author: Castle
# % Dara:12/11/2022
# % Content:
# Accelerate the block by pre-calculate knn idx
# Support to choose Block component by config
# Support Concatenation/OnebyOne for local and global Attn
###############################################################

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
from utils.logger import *
import einops
import open3d as o3d
import numpy as np

########################################################################################################################

def transform_pointcloud(pointcloud):

    p = pointcloud
    n = calculate_normals_open3d(p)
    x = find_nearest_neighbors(p)

    alpha = torch.sqrt(torch.sum((x - p) ** 2, dim=-1) - torch.sum(n * (x - p), dim=-1) ** 2)
    alpha = alpha.unsqueeze(-1)

    beta = torch.sum(n * (x - p), dim=-1).unsqueeze(-1)

#    print("p shape: ", p.shape)
#    print("n shape: ", n.shape)
#    print("beta shape: ", beta.shape)

#    shape_index = calculate_shape_index(p)

#    print("shape_index shape: ", shape_index.shape)

    transformed_points = torch.cat([
        p,
        n,
        alpha,
        beta,
  #      shape_index
    ], dim=-1)
    
    return transformed_points

def calculate_normals_open3d(points):

    pointcloud_tensor = points.cpu()
    pointcloud_np = pointcloud_tensor.numpy()

    normals_tensor = torch.zeros_like(pointcloud_tensor, device='cuda:0')

    for i in range(pointcloud_tensor.shape[0]):
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud_np[i])
        o3d_pointcloud.estimate_normals()

        normals_np = np.asarray(o3d_pointcloud.normals)

        normals_tensor[i] = torch.from_numpy(normals_np)


    return normals_tensor

def find_nearest_neighbors(points):
    distances = torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1)
    _, indices = torch.topk(distances, k=2, dim=-1, largest=False)
    
    # Choosing the second nearest neighbor (the first one is the point itself)
    x = torch.gather(points, dim=1, index=indices[:, :, 1].unsqueeze(-1).expand(-1, -1, 3))
    
    return x

def calculate_shape_index(points, k=10):
    """
    Calculates the shape index for each point in the batch.
    For each point p, the shape index is computed as follows:
        1. Fit a bi-quadratic surface to the k nearest neighbors of p and use the least squares method to find the coefficients
        2. Use differential geometry to calculate principal curvatures and principal directions
        3. With k1 and k2 being the minimum and maximum principal curvatures, the shape index is calculated as:
            shape_index = 0.5 - 1/pi * arctan((k1 + k2) / (k1 - k2))
    Input:
        points: all points, [B, N, C]
        k: number of nearest neighbors to consider
    Return:
        shape_index: grouped points index, [B, N]
    """
    B, N, C = points.size()
    device = points.device

    # Initialize shape_index tensor
    shape_index = torch.zeros((B, N), dtype=torch.float32, device=device)

    for b in range(B):
        # Calculate pairwise distances
        dist_matrix = torch.cdist(points[b], points[b])

        # Get indices of k nearest neighbors
        _, indices = torch.topk(dist_matrix, k=k+1, largest=False, sorted=True)
        indices = indices[:, 1:]  # Exclude self

        # Gather k nearest neighbor points
        neighbor_points = torch.gather(points[b], 1, indices.unsqueeze(2).expand(-1, -1, C))

        # Calculate surface coefficients using least squares method
        A = torch.cat([torch.ones_like(indices[:, :, 0:1]), neighbor_points], dim=-1)
        coeffs, _ = torch.lstsq(points[b].unsqueeze(1).repeat(1, N, 1), A)

        # Compute principal curvatures
        p0 = coeffs[:, :, 1]
        p1 = coeffs[:, :, 2]
        p2 = coeffs[:, :, 3]
        E = torch.sqrt(1 + p1 ** 2)
        F = p1 * p2
        G = torch.sqrt(1 + p2 ** 2)
        L = p0
        M = p1 * p0 + p2
        N = p0

        k1 = (L + N - torch.sqrt((L - N) ** 2 + 4 * M ** 2)) / 2
        k2 = (L + N + torch.sqrt((L - N) ** 2 + 4 * M ** 2)) / 2

        # Compute shape index
        shape_index[b] = 0.5 - (1 / torch.pi) * torch.atan((k1 + k2) / (k1 - k2))

    return shape_index


########################################################################################################################


def get_neighborhood(nsample, xyz, new_xyz):
    """
    Calculates the neighborhood for each point. Returns the indices of the selected neighbors.
    For each point p, the neighborhood is selected as follows:
        1. Define 4 regions around p made up of 90 degrees each
        2. Assign each point x of the batch to one of the 4 regions based on the angle between p and x
        3. Select the nsample/4 nearest neighbors from each region
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    # Calculate knn for each point
    sqrdists = torch.cdist(new_xyz, xyz, p=2)  # L2 distance
    # sqrdists shape: [B, S, N]

    # Define the number of regions
    num_regions = 4
    points_per_region = nsample // num_regions

    # Create a mask for selecting points from each region
    region_mask = torch.arange(num_regions).view(1, 1, -1) * (360.0 / num_regions)
    region_mask = region_mask.to(new_xyz.device)

    # Calculate angles between query points and all points
    delta_y = new_xyz[:, :, 1].unsqueeze(2) - xyz[:, :, 1].unsqueeze(1)  # Shape: [B, S, N]
    delta_x = new_xyz[:, :, 0].unsqueeze(2) - xyz[:, :, 0].unsqueeze(1)  # Shape: [B, S, N]
    angles = torch.atan2(delta_y, delta_x)  # Shape: [B, S, N]


    # Calculate angles between query points and all points
    angles = (angles * (180.0 / torch.pi) + 180.0) % 360.0  # Convert angles to degrees and ensure positive values
    # angles shape: [B, S, N]

    region_idx = torch.floor(angles / (360.0 / num_regions)).int()  # Calculate the region index for each point in the batch
    #print("region_idx shape: ", region_idx.shape)

    # Initialize grouped indices
    group_idx = torch.zeros((xyz.shape[0], new_xyz.shape[1], nsample), dtype=torch.long, device=new_xyz.device)

    for i in range(num_regions):
        # Mask for points in this region
        region_mask = (region_idx == i) # Shape: [B, S, N]

        # Calculate the squared distances for points in this region
        sqrdists_region = sqrdists.clone()
        sqrdists_region[~region_mask] = float('inf')  # Set distances for points not in this region to infinity

        #print("sqrdists_region shape: ", sqrdists_region.shape)

        # Calculate the number of points in this region
        num_points_region = torch.sum(region_mask, dim=-1)  # Shape: [B, S]

        # Sort distances within the region
        _, indices_region = torch.topk(sqrdists_region, xyz.shape[1], dim=-1, largest=False, sorted=True) # Shape: [B, S, N]
       # print("indices_region shape: ", indices_region.shape)

    
        # Number of points to select from this region
        points_to_select = points_per_region
        
        
        # Select the nearest points from this region
        selected_indices = indices_region[:, :, :points_to_select]  # Shape: [B, S, points_to_select]
        # selected_indices = indices_region[region_mask][:, :, :points_to_select]  # Shape: [B, S, points_to_select]
        
    #    print("selected_indices shape: ", selected_indices.shape)

        # Fill in the selected indices in the group index tensor
        group_idx[:, :, i * points_per_region:(i + 1) * points_per_region] = selected_indices


    #    print("group_idx final shape: ", group_idx.shape)
        
    return group_idx


def get_neighborhood_new(nsample, xyz, new_xyz):
    """
    Calculates the neighborhood for each point. Returns the indices of the selected neighbors.
    For each point p, the neighborhood is selected as follows:
        1. Define 4 regions around p made up of 90 degrees each
        2. Assign each point x of the batch to one of the 4 regions based on the angle between p and x
        3. Select the nsample/4 nearest neighbors from each region
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    # Calculate knn for each point
    sqrdists = torch.cdist(new_xyz, xyz, p=2)  # L2 distance
    # sqrdists shape: [B, S, N]

    # Define the number of regions
    num_regions = 8
    points_per_region = nsample // num_regions

    # Create a mask for selecting points from each region
    region_mask = torch.arange(num_regions).view(1, 1, -1) * (360.0 / num_regions)
    region_mask = region_mask.to(new_xyz.device)

    # Calculate angles between query points and all points
    delta_x = new_xyz[:, :, 0].unsqueeze(2) - xyz[:, :, 0].unsqueeze(1)  # Shape: [B, S, N]
    delta_y = new_xyz[:, :, 1].unsqueeze(2) - xyz[:, :, 1].unsqueeze(1)  # Shape: [B, S, N]
    delta_z = new_xyz[:, :, 2].unsqueeze(2) - xyz[:, :, 2].unsqueeze(1)  # Shape: [B, S, N]

    angles = torch.atan2(delta_y, delta_x)  # Shape: [B, S, N]
    angles2 = torch.atan2(delta_z, torch.sqrt(delta_x ** 2 + delta_y ** 2))  # Shape: [B, S, N]

    # Calculate angles between query points and all points
    angles = (angles * (180.0 / torch.pi) + 180.0) % 360.0  # Convert angles to degrees and ensure positive values
    angles2 = (angles2 * (180.0 / torch.pi) + 180.0) % 360.0  # Convert angles to degrees and ensure positive values
    # angles shape: [B, S, N]

    # Horizontal region index
    idx_count = 4

    # Vertical region index
    idx2_count = 2

    region_idx = torch.floor(angles / (360.0 / idx_count)).int()  # Calculate the region index for each point in the batch
    region_idx[region_idx == 4] = 3

    region_idx2 = torch.floor(angles2 / (360.0 / idx2_count)).int()  # Calculate the region index for each point in the batch
    region_idx2[region_idx2 == 2] = 1

    region_idx = region_idx + region_idx2 * 4
    #print("region_idx shape: ", region_idx.shape)

    # Initialize grouped indices
    group_idx = torch.zeros((xyz.shape[0], new_xyz.shape[1], nsample), dtype=torch.long, device=new_xyz.device)

    for i in range(num_regions):
        # Mask for points in this region
        region_mask = (region_idx == i) # Shape: [B, S, N]

        # Calculate the squared distances for points in this region
        sqrdists_region = sqrdists.clone()
        sqrdists_region[~region_mask] = float('inf')  # Set distances for points not in this region to infinity

        #print("sqrdists_region shape: ", sqrdists_region.shape)

        # Calculate the number of points in this region
        # num_points_region = torch.sum(region_mask, dim=-1)  # Shape: [B, S]

        # Sort distances within the region
        _, indices_region = torch.topk(sqrdists_region, xyz.shape[1], dim=-1, largest=False, sorted=True) # Shape: [B, S, N]
       # print("indices_region shape: ", indices_region.shape)

    
        # Number of points to select from this region
        points_to_select = points_per_region
        
        
        # Select the nearest points from this region
        selected_indices = indices_region[:, :, :points_to_select]  # Shape: [B, S, points_to_select]
        # selected_indices = indices_region[region_mask][:, :, :points_to_select]  # Shape: [B, S, points_to_select]
        
    #    print("selected_indices shape: ", selected_indices.shape)

        # Fill in the selected indices in the group index tensor
        group_idx[:, :, i * points_per_region:(i + 1) * points_per_region] = selected_indices


    #    print("group_idx final shape: ", group_idx.shape)
        
    return group_idx



def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # 1 for mask, 0 for not mask
            # mask shape N, N
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attn = attn.masked_fill(mask, mask_value) # B h N N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DeformableLocalAttention(nn.Module):
    r''' DeformabelLocalAttention for only self attn
        Query a local region for each token (k x C)
        Conduct the Self-Attn among them and use the region feat after maxpooling to update the token feat
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
    def forward(self, x, pos, idx=None):
        B, N, C = x.shape
        # given N token and pos
        assert len(pos.shape) == 3 and pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for pos, expect it to be B N 3, but got {pos.shape}'
        # first query a neighborhood for one query token
        if idx is None:
            idx = knn_point(self.k, pos, pos) # B N k 
        assert idx.size(-1) == self.k
        # project the qeury feat into shared space
        q = self.proj_q(x)
        v_off = self.proj_v_off(x)
        # Then we extract the region feat for a neighborhood
        local_v = index_points(v_off, idx) # B N k C 
        # And we split it into several group on channels
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # calculate offset
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset  = self.linear_offset(shift_feat) # Bg N k 3
        offset = offset.tanh() # Bg N k 3
        # add offset for each point
        # The position in R3 for these points
        local_v_pos = index_points(pos, idx) # B N k 3     
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
        shift_pos = local_v_pos + offset # Bg N 2*k 3
        # interpolate
        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
        pos = pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g N 3  
        pos = einops.rearrange(pos, 'b g n c -> (b g) n c') # Bg N 3
        v = einops.rearrange(x, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # three_nn and three_interpolate
        dist, _idx = pointnet2_utils.three_nn(shift_pos.contiguous(), pos.contiguous())  #  Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), _idx, weight).transpose(-1, -2).contiguous() 
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

        # some assert to ensure the right feature shape
        assert interpolated_feats.size(1) == local_v.size(1)
        assert interpolated_feats.size(2) == local_v.size(2)
        assert interpolated_feats.size(3) == local_v.size(3)
        # SE module to select 1/2k out of k
        pass

        # calculate local attn
        # local_q : B N k C 
        # interpolated_feats : B N k C 
        # extrate the feat for a local region
        local_q = index_points(q, idx) # B N k C
        q = einops.rearrange(local_q, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, k, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN k c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N k C
        out = out.max(dim=2, keepdim=False)[0]  # B N C
        out = self.proj(out)
        out = self.proj_drop(out)

        assert out.size(0) == B
        assert out.size(1) == N
        assert out.size(2) == C

        return out
        
# support denois task
class DeformableLocalCrossAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a cross attn among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            k = v
            NK = k.size(1)
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # project the qeury feat into shared space
            q = self.proj_q(q)
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            local_v = index_points(v_off, idx) # B N k C 
            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset  = self.linear_offset(shift_feat) # Bg N k 3
            offset = offset.tanh() # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx) # B N k 3     
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
            shift_pos = local_v_pos + offset # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g Nk 3  
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c') # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C 
            # interpolated_feats : B N k C 
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2) # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            q = self.proj_q(q)
            v_off = self.proj_v_off(v)

            ######################################### produce local_v by two knn #########################################
            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v_off[:, :-denoise_length], idx) # B N_r k C 
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx) # B N_r k 3     
           
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v_off, idx) # B N_n k C 
            local_v_n_pos = index_points(v_pos, idx) # B N_n k 3     
            ######################################### produce local_v by two knn #########################################
            
            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1) # B N k C 

            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
            
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset  = self.linear_offset(shift_feat) # Bg N k 3
            offset = offset.tanh() # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
            shift_pos = local_v_pos + offset # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g Nk 3  
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c') # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C 
            # interpolated_feats : B N k C 
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2) # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        return out

class improvedDeformableLocalCrossAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a cross attn among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
        $ improved:
            Deformable within a local ball
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if v is None:
            v = q
        if v_pos is None:
            v_pos = q_pos

        B, N, C = q.shape
        k = v
        NK = k.size(1)
        # given N token and pos
        assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
        assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
        
        # first query a neighborhood for one query token
        if idx is None:
            idx = knn_point(self.k, v_pos, q_pos) # B N k 
        assert idx.size(-1) == self.k
        # project the qeury feat into shared space
        q = self.proj_q(q)
        v_off = self.proj_v_off(v)
        # Then we extract the region feat for a neighborhood
        local_v = index_points(v_off, idx) # B N k C 
        # And we split it into several group on channels
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # calculate offset
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset  = self.linear_offset(shift_feat) # Bg N k 3
        offset = offset.tanh() # Bg N k 3

        # add offset for each point
        # The position in R3 for these points
        local_v_pos = index_points(v_pos, idx) # B N k 3     
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3

        # calculate scale
        scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0] # Bg N 3
        scale = scale.unsqueeze(-2) * 0.5 # Bg N 1 3
        shift_pos = local_v_pos + offset * scale # Bg N k 3
        
        # interpolate
        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
        v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g Nk 3  
        v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c') # Bg Nk 3
        v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg Nk c
        # three_nn and three_interpolate
        dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

        # some assert to ensure the right feature shape
        assert interpolated_feats.size(1) == local_v.size(1)
        assert interpolated_feats.size(2) == local_v.size(2)
        assert interpolated_feats.size(3) == local_v.size(3)
        # SE module to select 1/2k out of k
        pass

        # calculate local attn
        # local_q : B N k C 
        # interpolated_feats : B N k C 
        q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2) # BHN 1 c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, 1, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN 1 c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N 1 C
        assert out.size(2) == 1
        out = out.squeeze(2)
        out = self.proj(out)
        out = self.proj_drop(out)

        assert out.size(0) == B
        assert out.size(1) == N
        assert out.size(2) == C

        return out

class improvedDeformableLocalGraphAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a graph conv among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
        $ improved:
            Deformable within a local ball
    '''
    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim

        self.proj_v_off = nn.Linear(dim, dim)

        # Deformable related
        self.k = k  # To be controlled 
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # project the local feat into shared space
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            off_local_v = index_points(v_off, idx) # B N k C 
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset  = self.linear_offset(shift_feat) # B N k 3
            offset = offset.tanh() # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx) # B N k 3     

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0] # B N 3
            scale = scale.unsqueeze(-2) * 0.5 # B N 1 3
            shift_pos = local_v_pos + offset * scale # B N k 3
            
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c') # B k*N 3
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k) # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            v_off = self.proj_v_off(v)

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r_off = index_points(v_off[:, :-denoise_length], idx) # B N_r k C 
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx) # B N_r k 3     
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n_off = index_points(v_off, idx) # B N_n k C 
            local_v_n_pos = index_points(v_pos, idx) # B N_n k 3     
            # Concat two part
            off_local_v = torch.cat([local_v_r_off, local_v_n_off], dim=1) # B N k C 
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset  = self.linear_offset(shift_feat) # B N k 3
            offset = offset.tanh() # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0] # B N 3
            scale = scale.unsqueeze(-2) * 0.5 # B N 1 3
            shift_pos = local_v_pos + offset * scale # B N k 3
            
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c') # B k*N 3
            # three_nn and three_interpolate
            dist, idx = pointnet2_utils.three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = pointnet2_utils.three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k) # B N k c
            
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out

class DynamicGraphAttention(nn.Module):
    r''' DynamicGraphAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform Conv2d with maxpooling to build the graph feature for each token
        These can convert local self-attn as a local cross-attn
    '''
    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim
        # Deformable related
        self.k = k  # To be controlled 
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v = index_points(v, idx) # B N k C 
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((local_v - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v[:, :-denoise_length], idx) # B N_r k C 
            
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v, idx) # B N_n k C 

            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((local_v - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

######################################## Self Attention Block ########################################

class Block(nn.Module):
    r''' Normal Self-Attention block
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class RegionWiseBlock(nn.Module):
    r''' Block with region-wise deformable attn.
        Using The maxpool for token feat update
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.deformable_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos):
        x = x + self.drop_path1(self.ls1(self.deformable_attn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm1(x))))
        return x

class DeformableAttnBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.deformable_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos):
        x = x + self.drop_path1(self.ls1(self.deformable_attn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class GraphConvBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.graphattn = DynamicGraphAttention(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos):
        x = x + self.drop_path1(self.ls1(self.graphattn(self.norm1(x), pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
######################################## Cross Attention Block ########################################  
    
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q, v, q_pos, v_pos):
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(self.norm_q(q), self.norm_v(v))))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q

class DeformableAttnDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = DeformableLocalCrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q, v, q_pos, v_pos):
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(q=self.norm_q(q), v=self.norm_v(v), q_pos=q_pos, v_pos=v_pos)))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q

class GraphConvDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = DynamicGraphAttention(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, q, v, q_pos, v_pos):
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.attn(q=self.norm_q(q), v=self.norm_v(v), q_pos=q_pos, v_pos=v_pos)))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm2(q))))
        return q