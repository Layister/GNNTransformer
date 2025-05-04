#!/user/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import lung_finetune_flex
from utils import *
from dataset import LUNG
from dataset import DLPFC
from PIL import Image

def main():
    fold = 1
    dataname = 'DLPFC'
    tag = '-vit_1_1_cv'
    # 加载训练好的模型
    checkpoint_path = f"model/{dataname}_last_train_" + tag + '_' + str(fold) + ".ckpt"
    model = lung_finetune_flex.load_from_checkpoint(checkpoint_path)
    model.phase = "reconstruction"
    model.eval()  # 设置模型为评估模式

    # 加载数据集
    dataset = DLPFC(train=False, fold=fold)  # 使用测试集或验证集进行特征提取
    data_loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)

    all_features = []
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        for batch in data_loader:
            patches, centers, _ = batch
            # 进行特征提取
            for patch, center in zip(patches, centers):
                patch = patch.unsqueeze(0)  # 添加一个维度以匹配模型输入
                center = center.unsqueeze(0)
                features = model(patch, center)
                all_features.append(features.squeeze())

    # 合并所有特征到一个特征矩阵
    feature_matrix = torch.stack(all_features, dim=0).squeeze()

    # 保存特征矩阵为.pt文件
    feature_save_path = f"model/{dataname}_features_matrix_" + tag + '_' + str(fold) + ".pt"
    torch.save(feature_matrix, feature_save_path)
    print(f"特征矩阵已保存到 {feature_save_path}")

if __name__ == "__main__":
    # torch.set_float32_matmul_precision("medium")
    # main()
    dataname = 'DLPFC'
    data = torch.load(f"model/{dataname}_features_matrix_-vit_1_1_cv_1.pt")
    print(data)
