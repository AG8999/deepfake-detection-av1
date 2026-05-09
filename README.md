# Deepfake Detection in AV1 Compressed Videos
### EfficientNet-B0 + Stacked Bi-LSTM Ensemble Model

**MSc Research Project | Data Analytics | National College of Ireland**  
**Author:** Aniket Suryakant Ghadge (x23106786)  

---

## Overview

This research investigates the effect of **AV1 lossy compression** on deepfake video detection — a research gap not addressed by prior work focused on H.264/H.265 codecs. A novel ensemble model combining **EfficientNet-B0** (spatial feature extraction) and a **3-layered Bidirectional LSTM** (temporal inconsistency detection) is proposed and evaluated across three video formats:

- Raw (uncompressed)
- AV1 compressed at **250 kbps** (low bitrate)
- AV1 compressed at **1024 kbps** (high bitrate)

The model achieves **over 90% accuracy** across all three formats using the FaceForensics++ dataset.

---

## Key Results

| Format | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Raw Video | 90.18% | 0.9545 | 0.8235 | 0.8842 |
| AV1 @ 250 kbps | 91.96% | 0.9615 | 0.8772 | 0.9174 |
| AV1 @ 1024 kbps | 91.07% | 0.8852 | 0.9474 | 0.9153 |

The high-bitrate model achieved the best recall and fewest false positives/negatives. Low-bitrate compression increases model complexity by obscuring deepfake artefacts.

---

## Model Architecture

```
FaceForensics++ Videos
        |
   AV1 Compression (HandBrake)
        |
   Frame Extraction (10 frames/video)
        |
   Face Detection + Cropping (dlib)
        |
   Feature Extraction (scikit-image: noise, entropy, phase)
        |
   Resize to 224x224
        |
   EfficientNet-B0 (pretrained on ImageNet)  <-- Spatial features
        |
   3-Layer Stacked Bidirectional LSTM (1280-dim hidden, 0.5 dropout)
        |
   SoftMax Activation --> Real (1) / Fake (0)
```

---

## Repository Structure

```
deepfake-detection-av1/
|
|- data/                        # Dataset (see data/README.md for download)
|   |- raw/                     # Raw FaceForensics++ videos
|   |- compressed_250kbps/      # AV1 compressed at 250 kbps
|   |- compressed_1024kbps/     # AV1 compressed at 1024 kbps
|
|- src/
|   |- preprocessing/
|   |   |- compress_videos.py       # HandBrake/ffmpeg AV1 compression pipeline
|   |   |- label_videos.py          # Binary labelling (real=1, fake=0)
|   |   |- extract_frames.py        # Frame extraction using ffmpeg
|   |   |- face_crop.py             # Face detection + cropping using dlib
|   |   |- feature_extraction.py    # Noise, entropy, phase features (scikit-image)
|   |   |- resize_transform.py      # Resize to 224x224, augmentation
|   |
|   |- model/
|   |   |- efficientnet_bilstm.py   # Model architecture definition
|   |   |- train.py                 # Training loop with early stopping
|   |   |- predict.py               # Inference on new videos
|   |
|   |- evaluation/
|       |- evaluate.py              # Accuracy, precision, recall, F1, confusion matrix
|       |- plot_results.py          # Loss and accuracy curve plots
|
|- notebooks/
|   |- deepfake_detection_colab.ipynb   # Full pipeline (Google Colab, TPU runtime)
|
|- results/
|   |- figures/                     # Training/validation loss and accuracy plots
|   |- metrics/                     # Evaluation results per experiment
|
|- models/
|   |- saved/                       # Trained model weights (see models/README.md)
|
|- research_report.pdf              # Full MSc research paper
|- requirements.txt
|- .gitignore
|- README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended, TPU runtime used for training)
- [HandBrake](https://handbrake.fr/) for AV1 video compression

### Installation

```bash
git clone https://github.com/<your-username>/deepfake-detection-av1.git
cd deepfake-detection-av1
pip install -r requirements.txt
```

### Dataset and Model Weights

Large files (videos, model weights) are hosted on Google Drive:

**[Access Data and Model on Google Drive](https://drive.google.com/drive/folders/1EXU7wkOtbzYbdLN-9bqrF__GkmLLAj35?usp=sharing)**

Download and place files as described in `data/README.md` and `models/README.md`.

### Running the Pipeline

The full pipeline is available as a Google Colab notebook:

```
notebooks/deepfake_detection_colab.ipynb
```

To run individual stages locally:

```bash
# Step 1: Extract frames from videos
python src/preprocessing/extract_frames.py

# Step 2: Detect and crop faces
python src/preprocessing/face_crop.py

# Step 3: Feature extraction + resize
python src/preprocessing/feature_extraction.py
python src/preprocessing/resize_transform.py

# Step 4: Train model
python src/model/train.py

# Step 5: Evaluate
python src/evaluation/evaluate.py
```

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Framework | PyTorch + TorchVision |
| Runtime | Google Colab (TPU) |
| Optimizer | Adam (lr=1e-5, weight decay=1e-3) |
| Batch Size | 6 |
| Epochs | 25 (early stopping: patience=5) |
| EfficientNet Input | 224x224 RGB frames |
| LSTM Layers | 3 stacked bidirectional |
| LSTM Hidden Dim | 1280 |
| Dropout | 0.5 |
| Loss Function | Cross-Entropy |
| Train/Test Split | 80/20 |
| Dataset | FaceForensics++ (280 real + 280 fake videos) |

---

## Research Context

### Research Question
> How well does the fusion of EfficientNet-B0 and stacked Bi-LSTM models adapt to AV1 compressed videos at different bitrates, and how does this impact model performance compared to uncompressed videos?

### Key Findings
- AV1 low-bitrate compression hides deepfake artefacts, increasing false positives
- High-bitrate AV1 preserves enough signal for strong recall with balanced FP/FN
- The EfficientNet-B0 + Bi-LSTM fusion is robust across compression levels, consistently exceeding 90% accuracy
- Prior literature focused on H.264/H.265; this work addresses a clear gap for AV1

### Related Work
This work builds on and extends: Kuang et al. (2022), Guan et al. (2023), Wu et al. (2023), Chen et al. (2022), and K et al. (2023). Full references are in the research report.

---

## Citation

If you use this work, please cite:

```
Ghadge, A. S. (2024). Deepfake Detection in AV1 Compressed Videos with EfficientNet 
and Stacked Bi-LSTM Model. MSc Research Project, Data Analytics, National College 
of Ireland.
```

---

## License

This project is for academic and research purposes. The FaceForensics++ dataset is subject to its own [terms of use](https://github.com/ondyari/FaceForensics).

---

## Contact

**Aniket Suryakant Ghadge**  
MSc Data Analytics | National College of Ireland  
[LinkedIn](https://www.linkedin.com/in/aniket-ghadge) | [Portfolio](https://meetaniket.com)
