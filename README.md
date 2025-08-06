# HokuMaker - Hokusai Style Image Generation

## Overview

This Jupyter notebook implements a **Diffusion Denoising Probabilistic Model (DDPM)** to generate images in the style of Katsushika Hokusai, the famous Japanese ukiyo-e artist. The model learns to create new artwork by training on a dataset of Hokusai's paintings.

## Features

- **DDPM Implementation**: Complete implementation of the diffusion model from the paper "Denoising Diffusion Probabilistic Models"
- **U-Net Architecture**: Time-conditioned U-Net with sinusoidal position embeddings
- **Image Generation**: Generate new Hokusai-style artwork from random noise
- **Interactive Training**: Real-time loss plotting and progress tracking
- **Google Colab Integration**: Designed to run on Google Colab with Google Drive

## Requirements

### Dependencies
```bash
pip install pycm livelossplot torchinfo torchvision
```

### Key Libraries
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations
- **PIL**: Image processing
- **tqdm**: Progress bars

## Setup

1. **Google Drive Mounting**: The notebook automatically mounts Google Drive to access the dataset
2. **Dataset Path**: Ensure your Hokusai images are in `/content/drive/MyDrive/HokuMaker/Katsushika_Hokusai/`
3. **GPU Support**: Automatically detects and uses CUDA if available

## Model Architecture

### Key Components

1. **Sinusoidal Position Embeddings**: Time embeddings for the diffusion process
2. **ConvBlock**: Time-conditioned convolutional blocks
3. **EncBlock/DecBlock**: U-Net encoder and decoder blocks
4. **Unet**: Complete U-Net architecture with skip connections

### Model Parameters
- **Input Channels**: 3 (RGB)
- **Embedding Dimension**: 128
- **Timesteps**: 1000
- **Image Size**: 256x256

## Training Process

### Data Preparation
- Images are resized to 256x256
- Random cropping and horizontal flipping for augmentation
- Color jittering for robustness
- Normalization to [-1, 1] range

### Training Loop
- **Epochs**: 50
- **Batch Size**: 8
- **Learning Rate**: 1e-4 with cosine annealing
- **Loss Function**: MSE between predicted and actual noise
- **Optimizer**: Adam with weight decay

### Forward Diffusion
- Adds noise to images progressively over 1000 timesteps
- Uses linear variance schedule (β₁ = 5e-5, βₜ = 0.01)

## Generation Process

### Reverse Diffusion
- Starts from random noise
- Iteratively denoises using the trained model
- Clamps values to [-1, 1] range
- Visualizes progress every 100 timesteps

### Sampling
- Generates 256x256 RGB images
- Final output is in [0, 1] range for display

## Usage

1. **Training**: Run cells 0-15 to train the model
2. **Saving**: Model is saved to Google Drive
3. **Loading**: Load pre-trained model for generation
4. **Generation**: Run cells 18-22 to generate new images

## File Structure

```
HokuMaker/
├── notebook.ipynb          # Main training and generation notebook
├── README.md              # This file
└── Katsushika_Hokusai/   # Dataset directory (on Google Drive)
    └── *.jpg             # Hokusai artwork images
```

## Model Files

- **second_model.pt**: Trained model weights (saved to Google Drive)

## Notes

- Designed for Google Colab environment
- Requires Google Drive with Hokusai dataset
- GPU recommended for faster training
- Model can be fine-tuned for different art styles

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (for sinusoidal embeddings)

## License

This project is for educational and research purposes. Please respect copyright when using generated images. 