import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import clip
import argparse

@torch.no_grad()
def calculate_clip_scores(csv_path, image_folder, output_path="clip_scores.csv", device=None):
    """
    计算每张图片与对应prompt的CLIP Score
    并输出整体平均分与标准差

    参数：
        csv_path: 包含 case_number 和 prompt 列的 CSV
        image_folder: 存放图片的目录 (文件名格式如 1_0.png)
        output_path: 保存结果的 CSV 文件名
        device: cuda 或 cpu
    """

    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 CLIP 模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ Loaded CLIP model on {device}")

    # 读取 CSV
    df = pd.read_csv(csv_path)
    results = []

    print(f"🔍 Found {len(df)} prompts, start evaluating...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_number = row["case_number"]
        prompt = str(row["prompt"])

        # 图片文件路径
        image_path = os.path.join(image_folder, f"{case_number}_0.png")
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        # 提取文本和图像特征
        text = clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 计算 CLIP score（余弦相似度）
        score = (image_features @ text_features.T).item()

        results.append({
            "case_number": case_number,
            "prompt": prompt,
            "image_name": f"{case_number}_0.png",
            "clip_score": score
        })

    # 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"✅ Saved individual results to {output_path}")

    # 计算整体平均值与标准差
    if len(out_df) > 0:
        mean_clip = out_df['clip_score'].mean()
        std_clip = out_df['clip_score'].std()
        print(f"\n📊 Overall CLIP Score Statistics:")
        print(f"   Mean CLIP Score: {mean_clip:.4f}")
        print(f"   Standard Deviation: {std_clip:.4f}")

        # 同时写入一个summary文件
        summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Mean CLIP Score: {mean_clip:.4f}\n")
            f.write(f"Standard Deviation: {std_clip:.4f}\n")
        print(f"📁 Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP Scores for generated images.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with case_number and prompt columns.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing generated images.")
    parser.add_argument("--output_path", type=str, default="clip_scores.csv", help="Where to save CLIP score results.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu).")
    args = parser.parse_args()

    calculate_clip_scores(args.csv_path, args.image_folder, args.output_path, args.device)
