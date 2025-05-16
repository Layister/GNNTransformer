#!/user/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import old_finetune_flex, uni_finetune_flex
from dataset import oldTranscriptomics, Transcriptomics


def predict_with_model(dataname='Lung', cell_feat_dim=3000, model_type='old_finetune_flex',
                       checkpoint_path='model/Lung_last_train-vit_1_1_cv.ckpt'):
    # 设置GPU设备
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 根据模型类型加载数据集
    if dataname == 'Lung':
        names = ['A1']
    elif dataname == 'DLPFC':
        names = ['151508']
    else:
        names = None

    if model_type == 'old_finetune_flex':
        dataset = oldTranscriptomics(names=names, train=False, cell_feat_dim=cell_feat_dim, loc_dir=f'../data/{dataname}/')
    elif model_type == 'uni_finetune_flex':
        dataset = Transcriptomics(names=names, train=True, cell_feat_dim=cell_feat_dim, loc_dir=f'../data/{dataname}/')
    else:
        dataset = None

    loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=False)

    # 根据模型类型加载模型
    if model_type == 'old_finetune_flex':
        model = old_finetune_flex.load_from_checkpoint(checkpoint_path, n_layers=5, n_genes=cell_feat_dim, learning_rate=1e-4)
    elif model_type == 'uni_finetune_flex':
        model = uni_finetune_flex.load_from_checkpoint(checkpoint_path, n_genes=cell_feat_dim)
    else:
        raise ValueError("Invalid model type. Choose between 'old_finetune_flex' and 'uni_finetune_flex'.")

    # 设置模型为评估模式
    model.to(device)
    model.eval()
    model.phase = "reconstruction"

    # 进行预测
    predictions = []
    with torch.no_grad():
        for batch in loader:
            if model_type == 'old_finetune_flex':
                patch, centers, _ = batch
                patch, centers = patch.to(device), centers.to(device)
                pred_gene = model(patch, centers)
            elif model_type == 'uni_finetune_flex':
                patch, centers, _, img = batch
                patch, centers, img = patch.to(device), centers.to(device), img.to(device)
                pred_gene = model(patch, centers, img)
            predictions.append(pred_gene.squeeze())
    return torch.stack(predictions, dim=0).squeeze()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dataname = 'DLPFC'
    tag = '-vit_1_1_cv'

    # 使用模型进行预测
    predictions = predict_with_model(dataname=dataname, model_type='old_finetune_flex',
                                     checkpoint_path=f'model/{dataname}_last_train{tag}.ckpt')
    print(predictions)

    save_path = f"model/{dataname}_features_matrix" + tag + ".pt"
    torch.save(predictions, save_path)
    print(f"特征矩阵已保存到 {save_path}")

    # data = torch.load(f"model/{dataname}_features_matrix-vit_1_1_cv.pt")
    # print(data)
