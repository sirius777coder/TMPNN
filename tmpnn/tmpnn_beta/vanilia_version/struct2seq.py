import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import data
import protein_features
import numpy as np
from protein_features import gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, ProteinFeatures
from typing import Optional, Tuple, Sequence


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_dff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_dff, bias=True)
        self.W_out = nn.Linear(num_dff , num_hidden, bias=True)

    def forward(self, h_V):
        h = F.gelu(self.W_in(h_V))
        h = self.W_out(h)
        return h


# Update Nodes
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, device ,num_heads=4):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.device = device
        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax 
        mask_attend : 1代表非mask, 0代表mask,应该加一个非常大的负数
        """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(
            mask_attend > 0, attend_logits, torch.tensor(negative_inf,device=self.device))
        attend = nn.functional.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, top_k, N_hidden]
            bias:           Bias for attn_logits    [N_batch, N_nodes, top_k]
            mask_attend:    Mask for attention      [N_batch, N_nodes, top_k]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, top_k = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, top_k, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, top_k, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view(
            [n_batch, n_nodes, top_k, n_heads]).transpose(-2, -1)
        # attend_logits : [N_batch, N_nodes, N_heads, top_k]
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2, 3))
        # h_V_update : [N_batch, N_nodes, N_heads, d]
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        # h_V_update : [N_batch, N_nodes, num_hidden]
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class OuterProduct(nn.Module):
    def __init__(self, num_hidden1, num_hidden2):
        super(OuterProduct, self).__init__()
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.linear1 = nn.Linear(num_hidden1, num_hidden2)
        self.linear2 = nn.Linear(num_hidden2, num_hidden1)
        self.norm = nn.LayerNorm(num_hidden2)

    def forward(self, h_V, E_idx):
        """
        h_V : [Batch, Length, hidden]
        给定边的信息,做外积,取前k个节点得到相应的边信息
        """
        h_V = self.linear1(self.norm(h_V))

        outer = h_V[:, :, None, :] * h_V[:,None, :, :]
        # outer : [Batch, 1, Length, C] * [Batch, Length, 1, C] = [Batch, Length, Length, C]
        outer = protein_features.gather_edges(outer, E_idx)
        # outer : [Batch, Length, top_k, C]
        return self.linear2(outer)


class EdgeSelfAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super(EdgeSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_V = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_E, mask_attend=None):
        """ Self-attention, graph-structured on edge O(k^2)
        Args:
            h_E:            Neighbor features       [N_batch, N_nodes, top_k, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, top_k]
        Returns:
            h_E:            Edge update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, top_k = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_O(h_E).view([n_batch, n_nodes, n_heads, top_k, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_heads, top_k, d])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_heads, top_k, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K.transpose(-1, -2)).view(
            [n_batch, n_nodes, top_k, top_k, n_heads]).transpose(-1, -3)
        # attend_logits : [N_batch, N_nodes, N_heads, top_k, top_k]
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            negative_inf = np.finfo(np.float32).min
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            # mask_attend:    Mask for attention      [N_batch, N_nodes, N_head, top_k]
            # mask : [N_batch, N_nodes, N_head, top_k]
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            # mask_2d : [N_batch, N_nodes, N_head, top_k, top_k]
            attend = torch.where(mask_2d > 0, attend_logits, torch.tensor(negative_inf,device=self.W_K.weight.device))
            attend = nn.functional.softmax(attend, dim=-1)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_E_update = (torch.matmul(attend, V)).transpose(-2,-3)
        # h_E : [N_batch, N_nodes, top_k, N_heads ,d]
        h_E_update = h_E_update.reshape(n_batch, n_nodes, top_k, -1)
        # h_E : [N_batch, N_nodes, top_k, hidden]
        h_E_update = self.W_O(h_E_update)
        return h_E_update


class EncoderLayer(nn.Module):
    def __init__(self, num_hidden, num_in, device ,num_heads=4, dropout=0.1):
        super(EncoderLayer, self).__init__()
        """
        注意这个num_in是边和点的信息concat到一起之后的结果
        """
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.device = device
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.norm4 = nn.LayerNorm(num_hidden)

        self.node_attention = NeighborAttention(num_hidden, num_in, device,num_heads )
        self.edge_attention = EdgeSelfAttention(
            num_hidden, num_heads)
        self.dense1 = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.dense2 = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.OuterProduct = OuterProduct(num_hidden, num_hidden)
        self.act = nn.GELU()

    def forward(self, h_V,  h_E, h_EV, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.act(self.node_attention(h_V, h_EV, mask_attend))
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.act(self.dense1(h_V))
        h_V = self.norm1(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Outer product information
        h_E += self.OuterProduct(h_V, E_idx)

        # Self attention
        dh_E = self.act(self.edge_attention(h_E, mask_attend))
        h_E = self.norm3(h_E + self.dropout3(dh_E))

        # Position-wise feedforward
        dhE = self.act(self.dense2(h_E))
        h_E = self.norm4(h_E + self.dropout4(dhE))

        return h_V, h_E



class DecoderLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([torch.nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def step(self, t, h_V, h_E, mask_V=None, mask_attend=None):
        """ Sequential computation of step t of a transformer layer """
        # Self-attention
        h_V_t = h_V[:,t,:]
        dh_t = self.attention.step(t, h_V, h_E, mask_attend)
        h_V_t = self.norm[0](h_V_t + self.dropout(dh_t))

        # Position-wise feedforward
        dh_t = self.dense(h_V_t)
        h_V_t = self.norm[1](h_V_t + self.dropout(dh_t))

        if mask_V is not None:
            mask_V_t = mask_V[:,t].unsqueeze(-1)
            h_V_t = mask_V_t * h_V_t
        return h_V_t


class TMPNN(nn.Module):
    def __init__(self,device,node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, num_tags=5, k_neighbors=30, augment_eps=0.,dropout=0.1):
        super().__init__()
        self.device=device
        # Hypeparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags

        # Featurization layers
        self.features = ProteinFeatures(node_features,edge_features,top_k=k_neighbors,augment_eps=augment_eps,dropout=dropout)

        # Embedding layers and nn modules
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_seq = nn.Embedding(vocab, hidden_dim)         # vocab token is 21 token
        self.W_cctop = nn.Embedding(self.num_tags,hidden_dim)
        self.W_cv = nn.Linear(hidden_dim,hidden_dim)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.merge_seq_1 = nn.Linear(hidden_dim,hidden_dim)
        self.merge_seq_2 = nn.Linear(hidden_dim,hidden_dim)

        # Encoder  +  cctop output
        self.Encoder = nn.ModuleList([EncoderLayer(hidden_dim,hidden_dim * 2, dropout=dropout, device=device) for _ in range(num_encoder_layers)])
        self.W_out_cctop = nn.Linear(hidden_dim,self.num_tags,bias=True)

        # Decoder + seq output
        self.Decoder = nn.ModuleList([DecoderLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)])
        self.W_out_seq = nn.Linear(hidden_dim, vocab, bias=True)

        # GELU activation layers
        self.act = nn.GELU()

        # # Gaussian filter 
        # self.filter_kernel = torch.tensor([0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044],device=device)
        # CRF
        self.crf = CRF(self.num_tags,batch_first=True)


        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self,E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes)
        ii = ii.to(self.W_v.weight.device)
        ii = ii.view((1, -1, 1))
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)

        return mask

    def forward(self,X, S, S_mask, L, mask,device):
        # Prepare node and edge embeddings and sequence embeddings
        V, E, E_idx = self.features(X, mask, L,device)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_S = self.W_seq(S)
        h_S_mask = self.W_seq(S_mask)
        V = V + self.merge_seq_2(self.act(self.merge_seq_1(V+h_S_mask)))
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.Encoder:
            # Encoder中同时更新h_V和h_E
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V ,h_E = layer(h_V, h_E, h_EV, E_idx ,mask_V=mask, mask_attend=mask_attend)
        #     h_V, h_E, h_EV,E_idx, mask_V=None, mask_attend=None



        # Decoder module just from the
        # Concatenate sequence embeddings for autoregressive decoder
        # h_S = self.W_seq(S) move to line 318
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        mask_attend = (self._autoregressive_mask(E_idx=E_idx)).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend


        mask_fw = mask_1D * (1. - mask_attend)
        h_ESV_encoder_fw = mask_fw * h_ESV_encoder

        for layer in self.Decoder:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_ESV, mask_V=mask)

        logits_seq = self.W_out_seq(h_V)
        log_probs_seq = F.log_softmax(logits_seq, dim=-1)

        # Predict the cctop label using sequence embedding h_S and encoder backbone information h_V
        h_temp = h_V + h_S
        h_cv = self.act(self.W_cv(h_temp)) + h_temp
        logits_cctop = self.W_out_cctop(h_cv)
        # logits_cctop [B, N, C]

        return log_probs_seq,logits_cctop
    
    def neg_loss_crf(self,emission,tag,mask):
        """
        CRF score for the cctop
        Input 
        - emission  [B, N, C] (batch_first = True)
        - tag       [B, N]
        - mask      [B, N]
        output : 
        scaler
        """
        if not isinstance(mask.dtype,torch.ByteTensor):
            return (self.crf(emission, tag, mask=mask.byte(),reduction="token_mean")).neg()
        else:
            return (self.crf(emission, tag, mask=mask,reduction="token_mean")).neg()
    
    def decode_crf(self,emission,mask):
        """
        CRF decode the sequence
        """
        if not isinstance(mask.dtype,torch.ByteTensor):
            return self.crf.decode(emission,mask=mask.byte())
        else:
            return self.crf.decode(emission,mask=mask)

    def sample(self, X, L, mask=None, temperature=1.0):
        """ Autoregressive decoding of a model
        X : [B, N ,4 ,3]
        L : a list contains a batch of length

        """
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, mask, L,self.device)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.Encoder:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V, h_E = layer(h_V, h_E, h_EV, E_idx, mask_V=mask, mask_attend=mask_attend)

        # h_V_encoder = h_V
        # Decoder alternates masked self-attention
        mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 20))
        h_S = torch.zeros_like(h_V,device=self.device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64,device=self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.Decoder))]
        for t in range(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:, t:t + 1, :]
            h_E_t = h_E[:, t:t + 1, :, :]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:, t:t + 1, :, :] * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
            for l, layer in enumerate(self.Decoder):
                # Updated relational features for future states
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = h_V_stack[l][:, t:t + 1, :]
                h_ESV_t = mask_bw[:, t:t + 1, :, :] * h_ESV_decoder_t + h_ESV_encoder_t
                h_V_stack[l + 1][:, t, :] = layer(
                    h_V_t, h_ESV_t, mask_V=mask[:, t:t + 1]
                ).squeeze(1)

            # Sampling step
            h_V_t = h_V_stack[-1][:, t, :]
            logits = self.W_out_seq(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)

            # Update
            h_S[:, t, :] = self.W_seq(S_t)
            S[:, t] = S_t

        # # Using h_S and Encoder information to predict cctop
        # h_cv = self.act(self.W_cv(h_V + h_S))
        # logits_cctop = self.W_out_cctop(h_cv)
        # B,N,C = logits_cctop.shape
        # # logits_cctop [B, N, C]
        # C = self.crf.decode_crf(logits_cctop,mask)
        return S

