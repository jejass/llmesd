# LLME-ESD: Concept Unlearning with LLM-Enhanced Prompts

This project extends the [ESD (Erase and Suppress Diffusion)](https://github.com/rohitgandikota/erasing) framework.  
ESD is a framework for **concept unlearning** in **Stable Diffusion** models.  
The original ESD can erase a *single* specified concept.  
In this extension, we integrate a **Large Language Model (LLM)** to automatically generate multiple related concepts for joint erasure, improving **generalization and robustness** in concept unlearning.

---

## 🧩 Environment Setup

Before running, install the required dependencies:

```bash
pip install -r requirements.txt
```

Tested environment:
Python 3.9
Stable Diffusion v1.4

## 📦 Dataset
We use the COCO dataset to compute FID and CLIP scores.
Download from the official site: https://cocodataset.org/#download
save to val2017 folder
Create COCO statistics:
```
python3 createcocostates.py
```

### 🚀 Training and Evaluation
#### 1. Erase the “Nudity” Concept
```bash
python3 llm_esd_sd.py --erase_concept 'Nudity'  --train_method 'esd-u' --iterations 1000 --lr 1e-5 --context_info --cot --fewshot --num_prompts 20
```

#### 2. Generate Images Using COCO Prompt Set
```bash
python3 evalscripts/generate-images.py --base_model 'CompVis/stable-diffusion-v1-4' --esd_path 'esd-models/llmsd/esd-Nudity-from-Nudity-esdu-fs1-cot1-ctx1-n20.safetensors' --prompts_path 'data/coco_10k.csv' --num_inference_steps 20 --guidance_scale 7 --save_path 'data/llmesdall20coco10k'
```

#### 3. Generate Unsafe Images Using I2P Prompt Set

```bash
python3 evalscripts/generate-images.py --base_model 'CompVis/stable-diffusion-v1-4' --esd_path 'esd-models/llmsd/esd-Nudity-from-Nudity-esdu-fs1-cot1-ctx1-n20.safetensors' --prompts_path 'data/unsafe-prompts4703.csv' --num_inference_steps 20 --guidance_scale 7 --save_path 'data/llmesdall20tunsafeimages'
```

### 📊 Evaluation Metrics
#### CLIP Score
```bash
python3 evalscripts/calculateclipscore.py --csv_path 'data/coco_10k.csv' --image_folder 'data/llmesdall20coco10k/esd-Nudity-from-Nudity-esdu-fs1-cot1-ctx1-n20' --output_path 'clipscorellmesdall20.csv'
```

#### FID Score
```bash
python3 fidcal.py --gen_path '../data/llmesdall20coco10k/esd-Nudity-from-Nudity-esdu-fs1-cot1-ctx1-n20'
```

### 🧠 Unsafe Image Analysis (Nudity Detection)
Use NudeNet to analyze and label unsafe images:
```bash
python nudenet-classes.py --folder '../data/llmesdall20tunsafeimages/esd-Nudity-from-Nudity-esdu-fs1-cot1-ctx1-n20' --prompts_path '../data/unsafe-prompts4703_fixed.csv' --save_path 'nudenetresllmesdall20.csv'
```

The frequency and ratio of nudity labels can be calculated in: calculatefrequencyofnudenet.ipynb

### 📚 eference
This project extends: 
Erase and Suppress Diffusion (ESD)
Rohit Gandikota et al.
https://github.com/rohitgandikota/erasing