import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
import math
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset



class lung_finetune_flex(pl.LightningModule):
    """
    spatial transcriptomics task in lung tissue. This class defines a neural network architecture and its training, validation, and testing routines.
    """
    def __init__(self, patch_size=128, n_layers=5, n_genes=3000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.dim=1024
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)
        self.phase = "reconstruction"  # Set initial phase
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        # _, centers, _ = patches.size()
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        patches = self.patch_embedding(patches)

        x = patches + centers_x + centers_y
        h = self.vit(x)
        # print(h.shape,'shape')
        if self.phase == "reconstruction":
            gene_recon = self.gene_head(h)
            return gene_recon
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

    def one_hot_encode(self,labels, num_classes):
        return torch.eye(num_classes)[labels]

    def check_for_invalid_values(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values!")

    def training_step(self, batch, batch_idx):
        patch, centers, target_gene= batch
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('train_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def validation_step(self, batch, batch_idx):
        patch, centers,target_gene= batch  # assuming masks are the segmentation ground truth

        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('eval_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def test_step(self, batch, batch_idx):
        patch, centers, target_gene = batch  # assuming masks are the segmentation ground truth
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('test_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        return loss

    def reconstruction_parameters(self):
        return list(self.gene_head.parameters())

    def configure_optimizers(self):
        if self.phase == "reconstruction":
            optimizer = torch.optim.Adam(self.reconstruction_parameters(), lr=1e-3)
        return optimizer


class uni_finetune_flex(pl.LightningModule):
    """
    This class defines a neural network architecture and its training, validation, and testing routines.
    """
    def __init__(self, n_genes=3000, dim=1536, n_pos=128):
        super().__init__()
        self.dim = dim
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.phase = "reconstruction"  # Set initial phase
        self.model = self.get_uni(local_dir = "model")
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers, img):
        min_vals = img.amin(dim=(2, 3), keepdim=True)
        max_vals = img.amax(dim=(2, 3), keepdim=True)
        img_normalized = (img - min_vals) / (max_vals - min_vals)

        batch_size = 16
        dataset = TensorDataset(img_normalized.squeeze(dim=0))
        loader = DataLoader(dataset, batch_size=batch_size)
        features = []
        self.model.eval()
        with torch.inference_mode():
            for batch in loader:
                x = batch[0]
                feat = self.model(x)  # [batch_size, dim]
                features.append(feat)

        features = torch.stack(features).unsqueeze(dim=0)  # [1, n, embed_dim]
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])

        h = features + centers_x + centers_y
        # print(h.shape,'shape')
        if self.phase == "reconstruction":
            gene_recon = self.gene_head(h)
            return gene_recon
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

    def one_hot_encode(self,labels, num_classes):
        return torch.eye(num_classes)[labels]

    def check_for_invalid_values(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values!")

    def training_step(self, batch, batch_idx):
        patch, centers, target_gene, img = batch
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers,img)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('train_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def validation_step(self, batch, batch_idx):
        patch, centers, target_gene, img = batch  # assuming masks are the segmentation ground truth

        if self.phase == "reconstruction":
            pred_gene = self(patch,centers,img)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('eval_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def test_step(self, batch, batch_idx):
        patch, centers, target_gene, img = batch  # assuming masks are the segmentation ground truth
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers,img)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('test_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        return loss

    def reconstruction_parameters(self):
        return list(self.gene_head.parameters())

    def configure_optimizers(self):
        if self.phase == "reconstruction":
            optimizer = torch.optim.Adam(self.reconstruction_parameters(), lr=1e-3)
        return optimizer

    def get_uni(self, local_dir):
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',  # 使用 ViT-Giant 模型
            'img_size': 224,  # 输入图像尺寸
            'patch_size': 14,  # 图像分块大小（14x14像素为一个块）
            'depth': 24,  # Transformer 编码器层数
            'num_heads': 24,  # 多头注意力机制的头数
            'init_values': 1e-5,  # 初始化参数值
            'embed_dim': 1536,  # 特征嵌入维度（输出维度为1536）
            'mlp_ratio': 2.66667 * 2,  # MLP 扩展比例
            'num_classes': 0,  # 无分类头，仅提取特征
            'no_embed_class': True,  # 禁用类别标记 CLS 的嵌入过程
            'mlp_layer': timm.layers.SwiGLUPacked,  # 激活函数类型（SwiGLU）
            'act_layer': torch.nn.SiLU,  # 激活函数（SiLU）
            'reg_tokens': 8,  # 向输入序列中添加入 8 个注册令牌
            'dynamic_img_size': True  # 支持动态输入尺寸
        }
        model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        return model

if __name__ == '__main__':
    a = torch.rand(1,4000,3*128*128)
    p = torch.ones(1,4000,2).long()
    model = lung_finetune_flex()
    x = model(a,p)
    print(x)
