
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.TransformerST_graph_func import TransformerST_graph_construction,calculate_adj_matrix,search_l
from src.TransformerST_utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata
from src.TransformerST_train_adaptive import TransformerST_Train
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import cv2
from pytictoc import TicToc
from anndata import AnnData
from rpy2.robjects.packages import importr
from rpy2.robjects import r
# from scipy.io import savemat
# import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score


importr('mclust')
pandas2ri.activate()

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)


# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean', help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')#1000
parser.add_argument('--cell_feat_dim', type=int, default=3000, help='Dim of input genes')
parser.add_argument('--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=1, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=1, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.001, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.0001, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=7, help='DEC cluster number.')#10
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

params = parser.parse_args()
params.device = device

# ################ Path setting
data_root = './data/DLPFC'
# all DLPFC folder list
# proj_list = ['A1', 'A2', 'A3', 'A4']
proj_list = ['151508']
# set saving result path
save_root = './output/DLPFC_adaptive/'


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.1):#0.01
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

def combine_information_neighbors(img_fold, adata_h5):
    image = cv2.imread(img_fold)
    # print(img.shape)
    params.use_feature = 0
    spatial_loc = adata_h5.obsm['spatial']
    # print(type(spatial_loc[0]))
    x_pixel = spatial_loc[:, 2]
    y_pixel = spatial_loc[:, 3]
    x = spatial_loc[:, 0]
    y = spatial_loc[:, 1]
    # print(image.shape)

    beta = 24
    alpha = 1
    g = []
    # print(x_pixel.shape)
    ######################
    for i in range(spatial_loc.shape[0]):
        max_x = image.shape[0]
        max_y = image.shape[1]
        # print(x_pixel.shape,max(0, x_pixel[i] - beta),min(max_x, x_pixel[i] + beta + 1),max(0, y_pixel[i] - beta),min(max_y, y_pixel[i] + beta + 1))
        nbs = image[max(0, x_pixel[i] - beta):min(max_x, x_pixel[i] + beta + 1),
              max(0, y_pixel[i] - beta):min(max_y, y_pixel[i] + beta + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    # print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    c4 = (c3 - np.mean(c3)) / np.std(c3)
    z_scale = np.max([np.std(x), np.std(y)]) * alpha
    z = c4 * z_scale
    z = z.tolist()
    # print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
    X = np.array([x, y, z]).T.astype(np.float32)

    return X

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
    adata.obs['True_labels'] = final_labels

def evaluation(estimated_labels, true_labels, file_path):
    # 将真实标签转换为数字标签
    unique_true_labels = np.unique(true_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_true_labels)}
    true_labels_numeric = true_labels.map(label_mapping)

    # 计算调整兰德指数（Adjusted Rand Index）
    ari = adjusted_rand_score(true_labels_numeric, estimated_labels)
    # 计算归一化互信息（Normalized Mutual Information）
    nmi = normalized_mutual_info_score(true_labels_numeric, estimated_labels)
    # 计算同质性（Homogeneity）
    homogeneity = homogeneity_score(true_labels_numeric, estimated_labels)
    # 计算完整性（Completeness）
    completeness = completeness_score(true_labels_numeric, estimated_labels)
    # 计算 V 测度（V-measure）
    v_measure = v_measure_score(true_labels_numeric, estimated_labels)

    # 打印评估结果
    print(f"调整兰德指数 (ARI): {ari}")
    print(f"归一化互信息 (NMI): {nmi}")
    print(f"同质性: {homogeneity}")
    print(f"完整性: {completeness}")
    print(f"V 测度: {v_measure}")

    # 将结果写入文件
    with open(file_path, 'w') as file:
        file.write(f"调整兰德指数 (ARI): {ari}\n")
        file.write(f"归一化互信息 (NMI): {nmi}\n")
        file.write(f"同质性: {homogeneity}\n")
        file.write(f"完整性: {completeness}\n")
        file.write(f"V测度: {v_measure}\n")


t = TicToc()
for proj_idx in range(len(proj_list)):
    t.tic()
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + data_name)
    file_fold = f'{data_root}/{data_name}/outs/'
    img_fold = f'{data_root}/{data_name}/outs/spatial/full_image.tif'

    ################### Load data
    adata_h5 = load_ST_file(file_fold=file_fold,load_images=False)#load ST data without image
    adata_h5.var_names_make_unique()
    # print(adata_h5.obsm['spatial'])

    adata = adata_preprocess(AnnData(adata_h5), min_counts=5, min_cells=5, pca_n_comps=params.cell_feat_dim)
    adata_X=adata.X.toarray()

    n_clusters = 7 #the number of clusters

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    pre_resolution = res_search_fixed_clus(adata, n_clusters)
    sc.tl.leiden(adata, resolution=pre_resolution, key_added='expression_louvain_label')

    # combine information of neighbors(includeing X=[x,y,z]), and get the three-dimensional coordinates.
    X = combine_information_neighbors(img_fold, adata_h5)

    ################### Graph construction
    graph_dict,data_graph = TransformerST_graph_construction(X, adata_h5.shape[0],adata.obs['expression_louvain_label'], params)
    params.use_feature = 1
    graph_dict_prue, data_graph_prue = TransformerST_graph_construction(X, adata_h5.shape[0],adata.obs['expression_louvain_label'], params)
    params.save_path = mk_dir(f'{save_root}/{data_name}/TransformerST_adaptive')

    params.cell_num = adata_h5.shape[0]
    print('==== Graph Construction Finished')

    vit_path = 'Vision Transformer/model/DLPFC_features_matrix-vit_1_1_cv.pt'
    vision_transformer = True  # Set this to True or False depending on your specific case
    # Check if using a vision transformer
    if vision_transformer:
      # If you want to combine the image vision transformer embedding with the original gene expression,
      # concatenate the outputs. Otherwise, you might use only the gene expression data.
      # Load image vision transformer embedding from a file
      # If gene_pred is saved as a PyTorch tensor
      gene_pred = torch.load(vit_path)
      gene_pred = gene_pred.numpy()  # Convert it to numpy if it's a tensor

      # Check if the dimensions match for concatenation
      if adata_X.shape[1] == gene_pred.shape[1]:
        # Process for vision transformer
        # Include processing steps for vision transformer if necessary
        print("Data will be processed for image-gene expression corepresentation using a vision transformer.")
        # The following line is where you'd include your vision transformer processing
        # For now, it just concatenates the arrays
        adata_X = np.concatenate((adata_X, gene_pred), axis=1)
      else:
        raise ValueError('The number of columns (genes) in adata_X and gene_pred must be the same')
    else:
      # If not using a vision transformer, you may decide to use adata_X as is
      # or any other processing you may want to apply
      print("Using gene expression data without vision transformer processing.")

    ################### Model training
    TransformerST_net = TransformerST_Train(adata_X, graph_dict, data_graph, graph_dict_prue, data_graph_prue, adata_h5.obsm['spatial'], params)
    if params.using_dec:
        TransformerST_net.train_with_dec()
    else:
        TransformerST_net.train_without_dec()
    TransformerST_feat, _, _ = TransformerST_net.process()

    np.savez(f'{params.save_path}/TransformerST_result.npz', sedr_feat=TransformerST_feat, params=params)
    t.toc()

    ################### Result plot
    adata_TransformerST = anndata.AnnData(TransformerST_feat)
    adata_h5 = load_ST_file(file_fold=file_fold,load_images=True)
    adata_h5.var_names_make_unique()
    adata_TransformerST.uns['spatial'] = adata_h5.uns['spatial']
    adata_TransformerST.obsm['spatial'] = adata_h5.obsm['spatial']
    adata_TransformerST.obs.index = adata_h5.obs.index

    sc.pp.neighbors(adata_TransformerST, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_TransformerST)
    eval_resolution = res_search_fixed_clus(adata_TransformerST, n_clusters)

    matplotlib.use('Agg')
    sc.tl.leiden(adata_TransformerST, key_added="TransformerST_leiden", resolution=eval_resolution)
    sc.pl.spatial(adata_TransformerST, img_key="hires", color=['TransformerST_leiden'], show=False)
    plt.savefig(f'{params.save_path}/TransformerST_leiden_plot.jpg', bbox_inches='tight', dpi=150)

    if "DLPFC" in data_root:
        get_true_labels(adata_TransformerST, 'data/DLPFC/barcode_level_layer_map.tsv', int(data_name))
        sc.pl.spatial(adata_TransformerST, img_key="hires", color=['True_labels'], show=False)
        plt.savefig(f'{params.save_path}/True_labels_plot.jpg', bbox_inches='tight', dpi=150)

        evaluation(adata_TransformerST.obs['TransformerST_leiden'], adata_TransformerST.obs['True_labels'], f'{params.save_path}/cluster_results.txt')

    # df_meta = pd.read_csv(f'{data_root}/{data_name}/outs/metadata.tsv', sep='\t')
    # df_meta['TransformerST'] = adata_TransformerST.obs['TransformerST_leiden'].tolist()
    # df_meta.to_csv(f'{params.save_path}/metadata.tsv', sep='\t', index=False)

