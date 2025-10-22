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

COCO_VAL_PATH = "../val2017/"    # ⚠️ 修改为你的 COCO val2017 图片路径
SAVE_PATH = "coco_val2017_stats.npz"     # 保存统计文件路径
BATCH_SIZE = 32                            # 每批处理的图片数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        # 只保留到 Mixed_7c
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 输出2048维
        )
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)  # [batch, 2048]
        return x

# ============================================================
# 3. 图像预处理
# ============================================================
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 4. 计算特征
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
                print(f"❌ 跳过损坏图像: {path}, 错误: {e}")
        if len(imgs) == 0:
            continue
        imgs = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            feats = model(imgs).cpu().numpy()
        features.append(feats)
    return np.concatenate(features, axis=0)

# ============================================================
# 5. 计算均值与协方差
# ============================================================
def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

# ============================================================
# 6. 主函数
# ============================================================
def main():
    print("COCO_VAL_PATH =", COCO_VAL_PATH)
    image_paths = sorted(glob(os.path.join(COCO_VAL_PATH, "*.jpg")))
    print(f"共找到 {len(image_paths)} 张 COCO val2017 图片")
    
    model = InceptionFeatureExtractor().to(DEVICE)
    feats = get_features(image_paths, model)
    mu, sigma = calculate_activation_statistics(feats)
    
    np.savez(SAVE_PATH, mu=mu, sigma=sigma)
    print(f"✅ 保存完成: {SAVE_PATH}")
    print(f"均值 shape: {mu.shape}, 协方差 shape: {sigma.shape}")
    print("你可以在后续FID计算时直接加载这个统计文件。")

if __name__ == "__main__":
    main()