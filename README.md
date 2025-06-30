# MNIST Digit Classifier with CNN + Grad-CAM

This project implements a CNN model to classify handwritten digits from the MNIST dataset using PyTorch, and visualizes the prediction using Grad-CAM via Gradio interface.

## Features
- Custom CNN with 3 conv layers, BatchNorm, Dropout
- Training script (`train.py`)
- Live prediction + heatmap viewer (`demo_interface.py`)
- Achieves ~99% test accuracy
- Easy to demo via web interface (Gradio)

## Run the demo
```bash
python demo_interface.py
