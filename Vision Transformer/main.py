import os
import torch
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import old_finetune_flex, uni_finetune_flex
from utils import *
from dataset import oldTranscriptomics
from dataset import Transcriptomics
from PIL import Image


def main():
    dataname = 'DLPFC'
    cell_feat_dim = 3000
    tag = '-vit_1_1_cv'

    # dataset = HER2ST(train=True, fold=fold)
    dataset = oldTranscriptomics(train=True, cell_feat_dim=cell_feat_dim, loc_dir=f'../data/{dataname}/')
    #dataset = Transcriptomics(train=True, cell_feat_dim=cell_feat_dim, loc_dir=f'../data/{dataname}/')

    train_loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)
    # model=STModel(n_genes=785,hidden_dim=1024,learning_rate=1e-5)
    model = old_finetune_flex(n_layers=5, n_genes=cell_feat_dim, learning_rate=1e-4)
    #model = uni_finetune_flex(n_genes=cell_feat_dim)
    model.phase = "reconstruction"
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=200)
    #trainer = pl.Trainer(gpus=1, max_epochs=200)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint(f"model/{dataname}_last_train" + tag + ".ckpt")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
