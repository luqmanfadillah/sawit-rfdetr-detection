# 🌴 Palm Oil Tree Detection using RF-DETR Nano

Repositori ini berisi kode *inference* lokal, dokumentasi pelatihan, serta pipeline evaluasi model **RF-DETR Nano** untuk deteksi dan kalkulasi pohon kelapa sawit dari citra satelit/drone.

---

## 📊 Model Performance Metrics

Evaluasi model dilakukan pada *test set* dengan hasil sebagai berikut:

| Metric | Score |
| :--- | :--- |
| **mAP@50** | **98.9%** |
| **Precision** | **97.9%** |
| **Recall** | **97.8%** |
| **F1-Score** | **97.9%** |
| **Cloud API Latency** | **7.9 ms** |

---

## 📁 Repository Structure

```text
.
├── lfs-sawit-rf-detr.ipynb    # Google Colab Notebook (Training & Local Inference)
└── README.md                  # Project Documentation
```

---

## 📥 Model Weights (.pth)

Karena file bobot model (checkpoint weights) berukuran besar, file .pth disimpan secara terpisah di Google Drive. Kamu bisa mengunduhnya secara langsung untuk keperluan inference lokal/offline tanpa keterbatasan API cloud.   
* 🔗 Download Checkpoint Weights: [https://drive.google.com/open?id=1TmmWSmcXslY4ojZ1rxkms-X0Y1Jhbk5R](https://drive.google.com/drive/folders/1TmmWSmcXslY4ojZ1rxkms-X0Y1Jhbk5R?usp=sharing)
* File Name: checkpoint_best_total.pth
* Model Architecture: RF-DETR (Nano)

---

## 🚀 Quick Start (Local Inference)

### 1. Requirements & Installation

Pastikan kamu telah menginstall dependensi yang dibutuhkan:

```bash
pip install -q "rfdetr[train,loggers]<1.9.0" roboflow "supervision==0.29.1"
```
### 2. Run Local Inference in Python
```bash
import torch
from rfdetr import RFDETRNano
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load Local Model Weights (.pth)
WEIGHTS_PATH = "checkpoint_best_total.pth"
model_local = RFDETRNano(pretrain_weights=WEIGHTS_PATH)

# 2. Load Image
image = Image.open("path/to/your/palm_tree_image.jpg")

# 3. Predict / Inference
result = model_local.predict(image, confidence=0.3)

# 4. Annotate & Visualize Result
bbox_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_scale=0.35)
detections_labels = [f"{confidence*100:.0f}%" for confidence in result.confidence]

detections_image = image.copy()
detections_image = bbox_annotator.annotate(detections_image, result)
detections_image = label_annotator.annotate(detections_image, result, detections_labels)

plt.figure(figsize=(10, 10))
plt.imshow(detections_image)
plt.axis("off")
plt.show()
```
---

## ☁️ Roboflow Cloud Deployment

Model ini juga telah di-deploy secara aktif di Roboflow Cloud sebagai cadangan layanan API:
* **Workspace**: `new-workspace-4e9wd`
* **Project**: `sawit-8npnv`
* **Version**: `2` (`sawit-8npnv/2`)
