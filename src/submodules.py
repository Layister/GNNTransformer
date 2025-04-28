#!/usr/bin/python
# -*- coding: utf-8 -*-
#
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# import src.network_utils as net_utils
from src.graph_agg import *
# from config import cfg



LEAKY_VALUE = 0.0
act = nn.LeakyReLU(LEAKY_VALUE,inplace=True)

# out_shape = (H-1)//stride + 1 # for dilation=1
def conv(in_channels, out_channels, kernel_size=3, act=True, stride=1, groups=1, bias=True):
    """
    Constructs a convolutional layer (1D) with optional LeakyReLU activation.
    It's a basic building block for neural network models, particularly useful for feature extraction from sequences or time-series data.
    """
    m = []
    m.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        padding=(kernel_size-1)//2, groups=groups, bias=bias))
    if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
    return nn.Sequential(*m)

# out_shape = H*stride + kernel - 2*padding - stride + out_padding # for dilation=1
# def upconv(in_channels, out_channels, stride=2, act=True, groups=1, bias=True):
#     m = []
#     kernel_size = 2 + stride
#     m.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#         padding=1, groups=groups, bias=bias))
#     if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
#     return nn.Sequential(*m)


class ResBlock(nn.Module):
    """
    Implements a residual block using 1D convolutional layers, commonly used in deep learning architectures to enable training of deeper networks by allowing better gradient flow.
    """
    def __init__(self, n_feats, kernel_size=3, res_scale=1, bias=True):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv1d(n_feats, n_feats, kernel_size, padding=(kernel_size-1)//2,bias=bias))
            if i == 0:
                m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range=1., rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1., 1., 1.), sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False
#
# def upsampler(in_channels, kernel_size=3, scale=2, act=False, bias=True):
#     m = []
#     if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#         for _ in range(int(math.log(scale, 2))):
#             m.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size, padding=1,bias=bias))
#             m.append(nn.PixelShuffle(2))
#             if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
#
#     elif scale == 3:
#         m.append(nn.Conv2d(in_channels, 9 * in_channels, kernel_size, padding=1, bias=bias))
#         m.append(nn.PixelShuffle(3))
#         if act: m.append(nn.LeakyReLU(LEAKY_VALUE,inplace=True))
#     else:
#         raise NotImplementedError
#
#     return nn.Sequential(*m)



# class PixelShuffle_Down(nn.Module):
#     def __init__(self, scale=2):
#         super(PixelShuffle_Down, self).__init__()
#         self.scale = scale
#     def forward(self, x):
#         # assert h%scale==0 and w%scale==0
#         b,c,h,w = x.size()
#         x = x[:,:,:int(h-h%self.scale), :int(w-w%self.scale)]
#         out_c = c*(self.scale**2)
#         out_h = h//self.scale
#         out_w = w//self.scale
#         outs = x.contiguous().view(b, c, out_h, self.scale, out_w, self.scale)
#         return outs.permute(0,1,3,5,2,4).contiguous().view(b, out_c, out_h, out_w)

# ----------GCNBlock---------- #
class Graph(nn.Module):
    r"""
    Designed for graph construction in neural networks. 
    It processes input features to construct a graph by determining node connections and weights, useful in graph-based learning tasks.
    """
    def __init__(self, scale, k=5, patchsize=3, stride=1, window_size=20, in_channels=256, embedcnn=None):
        r"""
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param window_size: size of matching window around each patch,
            i.e. the window_size x window_size patches around a query patch
            are used for matching
        :param in_channels: number of input channels
        :param embedcnn_opt: options for the embedding cnn
        """
        super(Graph, self).__init__()
        self.scale = scale
        self.k = k
        # self.vgg = embedcnn is not None

        # if embedcnn is None:
        #     embed_ch = 64
        #     embed_out = 8
        #     self.embedcnn = nn.Sequential(
        #         conv(in_channels, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_out, kernel_size=3)
        #     )
        # else:
        #     self.embedcnn = embedcnn

        # indexer = lambda xe_patch,ye_patch: index_neighbours(xe_patch, ye_patch, window_size, scale)

        self.graph_construct = GraphConstruct(scale=scale, k=k,
                patchsize=patchsize, stride=stride)


    def forward(self, x, y):
        # x: son features, y: father features

        # xe = self.embedcnn(x)
        # ye = self.embedcnn(y)

        score_k, idx_k, diff_patch = self.graph_construct(x, y)
        return score_k, idx_k, diff_patch

class GCNBlock(nn.Module):
    r"""
    Graph Aggregation. A Graph Convolutional Network block that aggregates features from graph nodes. 
    This class is key in learning high-level features in graph-structured data.
    """
    def __init__(self, scale, k=5, patchsize=3, stride=1, diff_n=192):
        r"""
        :param nplanes_in: number of input features
        :param scale: downsampling factor
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param diff_n: number of diff vector channels
        """        
        super(GCNBlock, self).__init__()
        # self.nplanes_in = nplanes_in
        self.scale = scale
        self.k = k
        RES_SCALE=0.1
        # self.pixelshuffle_down = PixelShuffle_Down(scale)

        self.graph_aggregate = GraphAggregation(scale=scale, k=k,
                patchsize=patchsize, stride=stride)

        # self.knn_downsample = nn.Sequential(
        #         conv(nplanes_in, nplanes_in, kernel_size=5, stride=scale),
        #         conv(nplanes_in, nplanes_in, kernel_size=3),
        #         conv(nplanes_in, nplanes_in, kernel_size=3, act=False)
        #     )

        # self.diff_downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)

        # self.weightnet_lr = nn.Sequential(
        #                 conv(diff_n, 64, kernel_size=1, act=False),
        #                 ResBlock(64, kernel_size=1, res_scale=NETWORK.RES_SCALE),
        #                 conv(64, 1, kernel_size=1, act=False)
        #             )
        self.weightnet_hr = nn.Sequential(
                        conv(diff_n, 64, kernel_size=1, act=False),
                        ResBlock(64, kernel_size=1, res_scale=RES_SCALE),
                        conv(64, 1, kernel_size=1, act=False)
                    )

 
    def weight_edge(self, knn_hr, diff_patch):
        b, c, h_hr  = knn_hr.shape
        b, ce, _ = diff_patch.shape
        # print(c,ce,h_hr,"w")
        h_lr = h_hr//self.scale
        # print(self.scale,self.k,"dddddddddd")
        knn_hr = knn_hr.view(b, self.k, c//self.k, h_hr)
        diff_patch = diff_patch.view(b, self.k, ce//self.k, h_hr)
        print(diff_patch.shape,"www1ww")
        knn_lr,weight_hr = [],[]
        for i in range(self.k):
            # knn_lr.append(self.knn_downsample(knn_hr[:,i]).view(b, 1, c//self.k, h_lr, w_lr))
            # diff_patch_lr = self.diff_downsample(diff_patch[:,i])
            # weight_lr.append(self.weightnet_lr(diff_patch_lr))
            weight_hr.append(self.weightnet_hr(diff_patch[:,i]))


        # weight_lr = torch.cat(weight_lr, dim=1)
        # weight_lr = weight_lr.view(b, self.k, 1, h_lr, w_lr)
        # weight_lr = F.softmax(weight_lr, dim=1)
     
        weight_hr = torch.cat(weight_hr, dim=1)
        weight_hr = weight_hr.view(b, self.k, 1, h_hr)
        weight_hr = F.softmax(weight_hr, dim=1)

        # knn_lr = torch.cat(knn_lr, dim=1)
        # knn_lr = torch.sum(knn_lr*weight_lr, dim=1, keepdim=False)
        knn_hr = torch.sum(knn_hr*weight_hr, dim=1, keepdim=False)
        knn_lr = knn_hr.view(b, c//self.k, self.scale,h_hr//self.scale)
        knn_lr=torch.mean(knn_lr, dim=2, keepdim=False)
        return knn_lr, knn_hr


    def forward(self, y,yd, idx_k, diff_patch):

        # graph_aggregate
        # yd = self.pixelshuffle_down(y)
        # print(y.shape,yd.shape,"eeeeeeeeee")
        # b k*c h*s w*s
        # print(yd.shape,'wwwwwwwwwwwww')
        knn_hr = self.graph_aggregate(y, yd, idx_k)
        # print(knn_hr.shape,"aaaaaaaaaa")
        # for diff socre
        knn_lr, knn_hr = self.weight_edge(knn_hr, diff_patch)

        return knn_lr, knn_hr


class Graph_spatial(nn.Module):
    r"""
    Similar to Graph, but tailored for spatial data. 
    It constructs a graph considering the spatial arrangement of features, which is critical in tasks like image processing or spatial genomics.
    """

    def __init__(self, scale, k=10, patchsize=3, stride=1, window_size=20, in_channels=256, embedcnn=None):
        r"""
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param window_size: size of matching window around each patch,
            i.e. the window_size x window_size patches around a query patch
            are used for matching
        :param in_channels: number of input channels
        :param embedcnn_opt: options for the embedding cnn
        """
        super(Graph_spatial, self).__init__()
        self.scale = scale
        self.k = k
        # self.vgg = embedcnn is not None

        # if embedcnn is None:
        #     embed_ch = 64
        #     embed_out = 8
        #     self.embedcnn = nn.Sequential(
        #         conv(in_channels, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_out, kernel_size=3)
        #     )
        # else:
        #     self.embedcnn = embedcnn

        # indexer = lambda xe_patch,ye_patch: index_neighbours(xe_patch, ye_patch, window_size, scale)

        self.graph_construct = GraphConstruct_spatial_gai(scale=scale, k=k,
                                              patchsize=patchsize, stride=stride)

    def forward(self, x, y,spatial):
        # x: son features, y: father features

        # xe = self.embedcnn(x)
        # ye = self.embedcnn(y)

        score_k, idx_k, diff_patch = self.graph_construct(x, y,spatial)
        return score_k, idx_k, diff_patch


class GCNBlock_spatial(nn.Module):
    r"""
    A spatial variant of the GCNBlock, focusing on aggregating spatially structured data. 
    It's particularly useful in scenarios where spatial relationships are crucial, such as in computer vision tasks.
    """

    def __init__(self, scale, k=5, patchsize=3, stride=1, diff_n=192):
        r"""
        :param nplanes_in: number of input features
        :param scale: downsampling factor
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param diff_n: number of diff vector channels
        """
        super(GCNBlock_spatial, self).__init__()
        # self.nplanes_in = nplanes_in
        self.scale = scale
        self.k = k
        RES_SCALE = 0.1
        # self.pixelshuffle_down = PixelShuffle_Down(scale)

        self.graph_aggregate = GraphAggregation_spatial_gai(scale=scale, k=k,
                                                patchsize=patchsize, stride=stride)

        # self.knn_downsample = nn.Sequential(
        #         conv(nplanes_in, nplanes_in, kernel_size=5, stride=scale),
        #         conv(nplanes_in, nplanes_in, kernel_size=3),
        #         conv(nplanes_in, nplanes_in, kernel_size=3, act=False)
        #     )

        # self.diff_downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)

        # self.weightnet_lr = nn.Sequential(
        #                 conv(diff_n, 64, kernel_size=1, act=False),
        #                 ResBlock(64, kernel_size=1, res_scale=NETWORK.RES_SCALE),
        #                 conv(64, 1, kernel_size=1, act=False)
        #             )
        self.weightnet_hr = nn.Sequential(
            conv(diff_n, 64, kernel_size=1, act=False),
            ResBlock(64, kernel_size=1, res_scale=RES_SCALE),
            conv(64, 1, kernel_size=1, act=False)
        )

    def weight_edge(self, knn_hr, diff_patch):
        b, c, h_hr = knn_hr.shape
        b, ce, _ = diff_patch.shape
        h_lr = h_hr // self.scale

        knn_hr = knn_hr.view(b, self.k, c // self.k, h_hr)
        diff_patch = diff_patch.view(b, self.k, ce // self.k, h_hr)

        knn_lr, weight_hr = [], []
        for i in range(self.k):
            # knn_lr.append(self.knn_downsample(knn_hr[:,i]).view(b, 1, c//self.k, h_lr, w_lr))
            # diff_patch_lr = self.diff_downsample(diff_patch[:,i])
            # weight_lr.append(self.weightnet_lr(diff_patch_lr))
            weight_hr.append(self.weightnet_hr(diff_patch[:, i]))

        # weight_lr = torch.cat(weight_lr, dim=1)
        # weight_lr = weight_lr.view(b, self.k, 1, h_lr, w_lr)
        # weight_lr = F.softmax(weight_lr, dim=1)

        weight_hr = torch.cat(weight_hr, dim=1)
        weight_hr = weight_hr.view(b, self.k, 1, h_hr)
        weight_hr = F.softmax(weight_hr, dim=1)

        # knn_lr = torch.cat(knn_lr, dim=1)
        # knn_lr = torch.sum(knn_lr*weight_lr, dim=1, keepdim=False)
        knn_hr = torch.sum(knn_hr * weight_hr, dim=1, keepdim=False)
        knn_lr = knn_hr.view(b, c // self.k, self.scale, h_hr // self.scale)
        knn_lr = torch.mean(knn_lr, dim=2, keepdim=False)
        return knn_lr, knn_hr

    def forward(self, y, yd, idx_k, diff_patch):
        # graph_aggregate
        # yd = self.pixelshuffle_down(y)
        # print(y.shape,yd.shape,"eeeeeeeeee")
        # b k*c h*s w*s
        knn_hr = self.graph_aggregate(y, yd, idx_k)
        print(diff_patch.shape,"[[[[[[")

        # for diff socre
        knn_lr, knn_hr= self.weight_edge(knn_hr, diff_patch)
        # knn_hr=torch.swapaxes(knn_hr, 1, 2)
        return knn_lr, knn_hr
class Graph_TransformerST(nn.Module):
    r"""
    Designed for integrating graph-based approaches with TransformerST models. 
    """

    def __init__(self, scale=1, k=2, patchsize=1, stride=1):
        r"""
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param window_size: size of matching window around each patch,
            i.e. the window_size x window_size patches around a query patch
            are used for matching
        """
        super(Graph_TransformerST, self).__init__()
        self.s = scale
        self.k = k
        # self.vgg = embedcnn is not None

        # if embedcnn is None:
        #     embed_ch = 64
        #     embed_out = 8
        #     self.embedcnn = nn.Sequential(
        #         conv(in_channels, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_ch, kernel_size=3),
        #         conv(embed_ch, embed_out, kernel_size=3)
        #     )
        # else:
        #     self.embedcnn = embedcnn

        # indexer = lambda xe_patch,ye_patch: index_neighbours(xe_patch, ye_patch, window_size, scale)

        self.graph_construct = GraphConstruct_TransformerST(scale=scale, k=k,
                                              patchsize=patchsize, stride=stride)

    def forward(self, raw, img,adata,gene,num):

        super_adata,dis= self.graph_construct(raw, img,adata,gene,num)
        return super_adata,dis
class GCNBlock_TransformerST(nn.Module):
    r"""
    Graph Aggregation. A Graph Convolutional Network block designed to work with TransformerST models. 
    """

    def __init__(self, k=5, patchsize=3, stride=1):
        super(GCNBlock_TransformerST, self).__init__()
        # self.nplanes_in = nplanes_in
        self.scale = 2
        self.k = k
        RES_SCALE = 0.1
        # self.pixelshuffle_down = PixelShuffle_Down(scale)

        self.graph_aggregate = GraphAggregation_TransformerST(k=k,
                                                patchsize=patchsize, stride=stride)

        # self.knn_downsample = nn.Sequential(
        #         conv(nplanes_in, nplanes_in, kernel_size=5, stride=scale),
        #         conv(nplanes_in, nplanes_in, kernel_size=3),
        #         conv(nplanes_in, nplanes_in, kernel_size=3, act=False)
        #     )

        # self.diff_downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)

        # self.weightnet_lr = nn.Sequential(
        #                 conv(diff_n, 64, kernel_size=1, act=False),
        #                 ResBlock(64, kernel_size=1, res_scale=NETWORK.RES_SCALE),
        #                 conv(64, 1, kernel_size=1, act=False)
        #             )
        # self.weightnet_hr = nn.Sequential(
        #     conv(diff_n, 64, kernel_size=1, act=False),
        #     ResBlock(64, kernel_size=1, res_scale=RES_SCALE),
        #     conv(64, 1, kernel_size=1, act=False)
        # )

    def weight_edge(self, knn_hr, diff_patch):

        b, c, h_hr = knn_hr.shape
        b, ce, _ = diff_patch.shape
        h_lr = h_hr // self.scale

        knn_hr = knn_hr.view(b, self.k, c // self.k, h_hr)
        diff_patch = diff_patch.view(b, self.k, ce // self.k, h_hr)

        knn_lr, weight_hr = [], []
        for i in range(self.k):
            # knn_lr.append(self.knn_downsample(knn_hr[:,i]).view(b, 1, c//self.k, h_lr, w_lr))
            # diff_patch_lr = self.diff_downsample(diff_patch[:,i])
            # weight_lr.append(self.weightnet_lr(diff_patch_lr))
            weight_hr.append(self.weightnet_hr(diff_patch[:, i]))

        # weight_lr = torch.cat(weight_lr, dim=1)
        # weight_lr = weight_lr.view(b, self.k, 1, h_lr, w_lr)
        # weight_lr = F.softmax(weight_lr, dim=1)

        weight_hr = torch.cat(weight_hr, dim=1)
        weight_hr = weight_hr.view(b, self.k, 1, h_hr)
        weight_hr = F.softmax(weight_hr, dim=1)

        # knn_lr = torch.cat(knn_lr, dim=1)
        # knn_lr = torch.sum(knn_lr*weight_lr, dim=1, keepdim=False)
        knn_hr = torch.sum(knn_hr * weight_hr, dim=1, keepdim=False)
        knn_lr = knn_hr.view(b, c // self.k, self.scale, h_hr // self.scale)
        knn_lr = torch.mean(knn_lr, dim=2, keepdim=False)
        return knn_lr, knn_hr

    def forward(self, adata,super_adata,dis,num):
        raw_feature=adata.X.toarray()
        for i in range(super_adata.shape[0]):
            index = super_adata.obs.index[i]
            dis_tmp = dis.loc[index, :].sort_values()
            nbs = dis_tmp[0:num]
            # print(nbs)
            k=2
            dis_tmp = (nbs.to_numpy() + 0.1) / np.min(nbs.to_numpy() + 0.1)  # avoid 0 distance
            if isinstance(k, int):
                weights = ((1 / (dis_tmp ** k)) / ((1 / (dis_tmp ** k)).sum()))
            else:
                weights = np.exp(-dis_tmp) / np.sum(np.exp(-dis_tmp))
            row_index = [adata.obs.index.get_loc(i) for i in nbs.index]
            # print(weights.shape,adata.X.shape)
            super_adata.X[i, :] =self.graph_aggregate(weights, raw_feature[row_index, :])
        return super_adata
