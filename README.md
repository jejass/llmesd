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

## Dataset
We use the COCO dataset to compute FID and CLIP scores.
Download from the official site: https://cocodataset.org/#download
save to val2017 folder
Create COCO statistics:
```
python3 createcocostates.py
```
