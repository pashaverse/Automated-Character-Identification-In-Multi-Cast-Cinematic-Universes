# Automated Character Identification in Multi-Cast Cinematic Universes

A deep learning pipeline for closed-set character identification in Stranger Things video frames.

**Group:** 2 (Project 2)  
**Members:** M. Fahad Pasha, M. Ali, M. Hadi  
**Course:** Programming for AI

## Overview

Five-class character classifier using MTCNN face detection and ResNet-50 transfer learning. Trained on self-collected S1–S5 dataset with 95.83% test accuracy.

**Target Characters:** Dustin (Gaten Matarazzo), Eleven (Millie Bobby Brown), Hopper (David Harbour), Lucas (Caleb McLaughlin), Steve (Joe Keery)

## Results

**Test Set Performance (clean single-face crops):**

```
Top-1 Accuracy:        95.83%
Macro F1-score:        95.73%
Per-class F1:
  dustin : 91.67%
  eleven : 94.87%
  hopper : 98.85%
  lucas  : 96.10%
  steve  : 97.14%
Uncertain predictions:  12 (threshold θ=0.70)
```

## Installation & Setup

```bash
cd "Automated-Character-Identification-In-Multi-Cast-Cinematic-Universes"
pip install -r requirements.txt
```

## Usage

**Full pipeline:**
```bash
python main.py --all
python main.py --all --dry-run  # Preview only
```

**Individual stages:**
```bash
python main.py --consolidate  # Merge raw data
python main.py --split         # Create train/val/test splits
python main.py --detect --season S4  # MTCNN preprocessing
python main.py --train         # Two-phase training
python main.py --evaluate      # Test evaluation + metrics
```

**Interactive demo:**
```bash
python app.py
# Open http://127.0.0.1:7860 in browser
```

Demo features: image upload, adjustable confidence threshold (0.3–0.9), multi-face toggle, bounding boxes (green=confident, red=uncertain), probability chart, pre-loaded test examples.

## Architecture

**ResNet-50 Transfer Learning:**
- ImageNet pretrained backbone
- Phase 1: Frozen backbone, train FC head only (5 epochs, lr=1e-3)
- Phase 2: Unfreeze layer3+layer4+FC, fine-tune (20 epochs, lr=1e-4, CosineAnnealingLR)
- Custom head: Linear(2048, 5) + Dropout(0.4)
- Early stopping: patience=5 on validation accuracy

**Data Processing:**
- Training: Resize(224), HorizontalFlip, Rotate(±15°), ColorJitter, GaussNoise, RandomResizedCrop(0.8-1.0), ImageNet Normalize
- Validation/Inference: Resize(224), ImageNet Normalize (no augmentation)
- MTCNN single-face detection with bbox clipping
- Confidence threshold θ=0.70 (adjustable in demo)

## File Structure

```
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── main.py                    # CLI orchestrator
├── app.py                     # Gradio demo
├── extract_faces_mtcnn.py     # MTCNN preprocessing
├── datasets.py                # PyTorch dataset
├── resnet50_model.py          # Model architecture
├── train_resnet50.py          # Training pipeline
├── evaluate.py                # Evaluation script
├── consolidate_dataset.py     # Data consolidation
├── create_splits.py           # Train/val/test split
├── src/                       # Package wrappers
├── dataset/                   # Raw input (S1-S5)
├── finalised_dataset/         # Consolidated images
├── finalised_dataset/splits/  # Train/val/test splits
├── outputs/                   # Metrics and plots
└── best_model.pth            # Trained checkpoint
```

## Limitations & Future Work

**Known Issues:**
- Multi-face scenes require face selection logic
- Domain-specific to Stranger Things; limited generalization
- Sensitive to lighting and occlusion
- Requires clean single-face crops for best performance

## References

He et al. (2016) — Deep Residual Learning for Image Recognition (ResNet)  
Zhang et al. (2016) — Joint Face Detection and Alignment using MTCNN  
Albumentations Documentation — Fast Image Augmentation  
PyTorch, facenet-pytorch, scikit-learn, Gradio documentation
