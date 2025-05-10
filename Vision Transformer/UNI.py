
import os
import torch
from torchvision import transforms
import timm
from PIL import Image


local_dir = "model"
timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',  # 使用 ViT-Giant 模型
            'img_size': 224,                        # 输入图像尺寸
            'patch_size': 14,                       # 图像分块大小（14x14像素为一个块）
            'depth': 24,                            # Transformer 编码器层数
            'num_heads': 24,                        # 多头注意力机制的头数
            'init_values': 1e-5,                    # 初始化参数值
            'embed_dim': 1536,                      # 特征嵌入维度（输出维度为1536）
            'mlp_ratio': 2.66667*2,                 # MLP 扩展比例
            'num_classes': 0,                       # 无分类头，仅提取特征
            'no_embed_class': True,                 # 禁用类别标记 CLS 的嵌入过程
            'mlp_layer': timm.layers.SwiGLUPacked,  # 激活函数类型（SwiGLU）
            'act_layer': torch.nn.SiLU,             # 激活函数（SiLU）
            'reg_tokens': 8,                        # 向输入序列中添加入 8 个注册令牌
            'dynamic_img_size': True                # 支持动态输入尺寸
        }

model = timm.create_model(
    pretrained=False, **timm_kwargs
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()

image = Image.open("../data/DLPFC/151507/outs/spatial/full_image.tif")
image = transform(image).unsqueeze(
    dim=0)  # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
with torch.inference_mode():
    feature_emb = model(image)  # Extracted features (torch.Tensor) with shape [1,1536]
    print(feature_emb.shape)