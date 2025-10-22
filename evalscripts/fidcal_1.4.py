import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from scipy import linalg
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import transforms

# ============================================================
# 参数设置
# ============================================================
GEN_PATH = "../data/coco10k/sdv14"        # ⚠️ 你的生成图片路径
COCO_STATS_PATH = "coco_val2017_stats.npz"  # 之前保存的统计文件
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 特征提取模型
# ============================================================
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn as nn

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ 使用新API加载预训练权重（替代 deprecated 'pretrained' 参数）
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        # ✅ 只保留到 Mixed_7c + pool 层
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 得到 [B, 2048, 1, 1]
        )
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)  # 输出 shape: [B, 2048]
        return x


# ============================================================
# 图像预处理
# ============================================================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 提取特征
# ============================================================
def get_features(image_paths, model):
    features = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"❌ 跳过损坏图片: {path}, 错误: {e}")
        if len(imgs) == 0:
            continue
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            feats = model(imgs).cpu().numpy()
        features.append(feats)
    return np.concatenate(features, axis=0)

# ============================================================
# 计算均值与协方差
# ============================================================
def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

# ============================================================
# FID公式
# ============================================================
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# ============================================================
# 主函数
# ============================================================
def main():
    # 加载生成图片
    gen_images = sorted(glob(os.path.join(GEN_PATH, "*.png")))
    print(f"找到 {len(gen_images)} 张生成图片。")

    # 加载 COCO 统计
    coco_stats = np.load(COCO_STATS_PATH)
    mu_real, sigma_real = coco_stats["mu"], coco_stats["sigma"]

    # 提取生成图片特征
    model = InceptionFeatureExtractor().to(DEVICE)
    gen_feats = get_features(gen_images, model)
    mu_gen, sigma_gen = calculate_activation_statistics(gen_feats)

    # 计算 FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"\n✅ FID 分数: {fid:.2f}")
    print("越小表示生成图片越接近 COCO 分布。")

if __name__ == "__main__":
    main()