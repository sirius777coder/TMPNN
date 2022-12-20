import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import rigid_utils
import utils
from rigid_utils import Rigid

# The following gather functions
def gather_edges(edges, neighbor_idx):
    """
    给定一个batch数据的边特征,给每个节点取前K个特征
    这个边特征可以是mask信息,CA之间的距离,N原子之间的距离等等
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = (neighbor_idx.unsqueeze(-1)).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """
    给每一个batch里的node找到包括自身在内的其他node的特征,输出和gather_edge是一样的
    输出能狗找到每个节点邻居k个节点的特征
    """
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    """
    在某一个时刻t,找到一个batch里面的前k个节点
    """
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """
    因为gather_nodes和gather_edges的维度是一样的都是B,L,K,C,将边的信息和点的信息拼接起来
    应用于E_ij边的信息融合全部的V_J信息
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings=128, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        # -32,-31,...-1,0,1,...,31,32
        self.linear = nn.Linear(2*max_relative_feature+1, num_embeddings)

        # self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)
        # mpnn多加了一个token是为了表示这两个氨基酸是否在同一个链上

    def forward(self, offset, mask=None):
        # d = torch.clip(offset + self.max_relative_feature, 0, 2 *self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        # torch.clip() * mask保证同一条链的offset token距离为0,64(-32,32);加(1-mask)*65保证不同链的offset token为65(33)
        d = torch.clip(offset + self.max_relative_feature, 0, 2 *
                       self.max_relative_feature)  # 因为TMPNN为同一条链的数据，并不考虑multichian的情况
        d_onehot = torch.nn.functional.one_hot(
            d, 2*self.max_relative_feature+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features,  num_rbf=16, top_k=30, augment_eps=0., dropout=0.1):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = edge_features

# From ProteinMPNN : edge embeddings
        self.pos_embeddings = PositionalEncodings(edge_features)
        # node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        node_in, edge_in = 6,  num_rbf*25  # node in : 6个二面角；edge in : 25个距离 * 16 + 氢键(1)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

# From NIPS 2019  : node embeddings ------ 需要重新考虑点的特征
        self.node_embedding = nn.Linear(node_in,  node_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)

    def _dist(self, X, mask, eps=1E-6):
        """
        X为CA原子的坐标 : [B,L,3]
        mask          : [B,L] 0代表mask,1代表非mask     
        """
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        # mask_2D : [B,L,L]
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        # dX      : [B,L,L,3]
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        # D       : [B,L,L] 表示两两氨基酸之间的距离
        D_max, _ = torch.max(D, -1, keepdim=True)
        # D_max   : [B,L] 统计每一个氨基酸最大的距离,给mask节点用的
        D_adjust = D + (1. - mask_2D) * D_max
        # D_adjust: [B,L,L] 根据D_max和mask来调整节点,因为在原始的D中mask节点的距离都是0,现在将mask节点的距离调整为最大
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(
            self.top_k, X.shape[1]), dim=-1, largest=False)
        # D_neighbors : [B,L,K]根据最小距离选择的top_k个节点的距离
        # E_idx       : [B,L,K]对应最小top_k个节点的索引

        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # D : [B,L,K]代表的CA之间Distance
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf  # 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :]-B[:, None, :, :])**2, dim=-1)+1e-6)  # [B,L,L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0]  # [B,L,L] -> [B,L,K] A,B代表两种相同或不同原子距离矩阵
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = nn.functional.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = nn.functional.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = nn.functional.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = nn.functional.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
             F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
          +  F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
        , -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
              _inv_distance(X_atoms['O'], X_atoms['N'])
            + _inv_distance(X_atoms['C'], X_atoms['H'])
            - _inv_distance(X_atoms['O'], X_atoms['H'])
            - _inv_distance(X_atoms['C'], X_atoms['N'])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    @staticmethod
    def get_bb_frames(coords):
        """
        Returns a local rotation frame defined by N, CA, C positions.
        Args:
            coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
            where the third dimension is in order of N, CA, C
        Returns:
            Local relative rotation frames in shape (batch_size x length x 3 x 3)
            Local translation in shape (batch_size x length x 3)
        """
        v1 = coords[:, :, 2] - coords[:, :, 1]
        v2 = coords[:, :, 0] - coords[:, :, 1]
        e1 = utils.normalize(v1, dim=-1)
        u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
        e2 = utils.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-2)
        t = coords[:, :, 1]  # translation is just the CA atom coordinate
        return R, t

    @staticmethod
    def backbone_frame(coord) -> rigid_utils.Rigid:
        """
        convert the coord to global frames
        Input : 
        coord [B x L x 3/4 x 3]_float32
        Ouput :
        bb_frame [B x L]_Rigid
        """
        bb_frame_atom = coord[:,:,0:3,:]
        bb_rotation,bb_translation = ProteinFeatures.get_bb_frames(bb_frame_atom)
        bb_frame = torch.zeros((*bb_rotation.shape[:-2],4,4),device=coord.device)
        bb_frame[...,:3,:3] = bb_rotation
        bb_frame[...,:3,3] = bb_translation # [B, L, 4, 4]
        bb_frame = Rigid.from_tensor_4x4(bb_frame)
        return bb_frame



    def forward(self, X, mask, L,device):
        # 给数据添加噪音
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        N  = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C  = X[:, :, 2, :]
        CB = X[:, :, 3, :]
        O  = X[:, :, 4, :]
        bb_frame = self.backbone_frame(X)


        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(CB, CB, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, CB, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, CB, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(CB, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(CB, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(CB, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(CB, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, CB, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, CB, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # 只看一个batch: 从mpnn和nips2019结合而来的简化版本
        # residue_idx[0,:,None]表示横坐标i为第i个氨基酸
        # residue_idx[0,None,:]表示纵坐标j为第j个氨基酸
        # offset[0,i,j]为第i个氨基酸index和第j个氨基酸坐标的差值
        residue_idx = [np.arange(X.size(1)) for _ in range(len(L))]
        # 其他tensor都是生成的，而residue_idx是第一次出现，所以应该添加tensor元素
        residue_idx = torch.from_numpy(np.array(residue_idx))
        residue_idx = residue_idx.to(device)
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[
            :, :, :, 0]  # [B, L, K]
        # offset此时是氨基酸序列的相对位置的信息

        # Pairwise embeddings
        E_positional = self.pos_embeddings(offset.long(), mask_neighbors)
        # E_hb = self._hbonds(X,E_idx,mask_neighbors)
        E = RBF_all

        E = self.edge_embedding(E)
        E = E + E_positional
        E = self.norm_edges(E)

        # Node embeddings
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)

        return V, E, E_idx,bb_frame
        # V : [B, L, node_features]
        # E : [B, L, k, edge_features]
        # E_idx : [B,L,K]
