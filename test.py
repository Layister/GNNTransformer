
import torch
import numpy as np
import pandas as pd
from src.TransformerST_utils_func import load_ST_file


def get_true_labels(adata, labels_path, id):
    # 读取包含真实标签的tsv文件
    df = pd.read_csv(labels_path, sep='\t')
    true_labels_df = df[df['sample_name'] == id]
    true_labels = true_labels_df.set_index(true_labels_df.columns[0])[true_labels_df.columns[2]]

    # 检查索引是否匹配
    missing_index = set(adata.obs.index) - set(true_labels.index)
    if missing_index:
        print("存在缺失的索引:", missing_index)
        # 为缺失的索引创建默认值为 "WM" 的 Series
        default_labels = pd.Series(['WM'] * len(missing_index), index=list(missing_index))
        # 将默认值和真实标签合并
        combined_labels = pd.concat([true_labels, default_labels])
        # 重新排序以匹配 adata_h5.obs.index
        final_labels = combined_labels.loc[adata.obs.index]
    else:
        final_labels = true_labels

    # 添加真实标签
    adata.obs['true_labels'] = final_labels


if __name__ == '__main__':
    # # 读取空间转录组数据
    # adata_h5 = load_ST_file(file_fold='data/DLPFC/151508/outs/', load_images=False)  # 加载ST数据（不包括img）
    # adata_h5.var_names_make_unique()
    # print(adata_h5.obs)
    #
    # get_true_labels(adata_h5, 'data/DLPFC/barcode_level_layer_map.tsv', 151508)
    #
    # print(adata_h5.obs)

    print(torch.cuda.is_available())
