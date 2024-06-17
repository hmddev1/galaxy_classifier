<!-- # Machine learning for morphological galaxy classification 

The *"Machine learning for morphological galaxy classification"* is a repository for classifying [Galaxy Zoo 2](GZ2) images into (1) Galaxy and Non-Galaxy, and (2) Galaxy in Spiral, Elliptical, and Odd objects. We used five classification models including Supprot Vector Machine (SVM) and ..... The SVM and classic CNN used the Zernike moments (ZMs) extracted from images, while CNN - transformer and resnet and vgg .... designed base on original images. 

The details of algorithms where explained in [Ghaderi, Alipour, and Safari](paperlink).

This repository includes two python notebooks of the classifiers for galaxy-non-galaxy classification (ntebook name) and galaxies classification (notebook name). -->


# Machine Learning for Morphological Galaxy Classification

Welcome to the **Machine Learning for Morphological Galaxy Classification** repository! This project focuses on classifying Galaxy Zoo 2 (GZ2) images into various categories using state-of-the-art machine learning models.

## Overview

In this repository, we classify GZ2 images into:
1. **Galaxy and Non-Galaxy**
2. **Galaxy Morphologies:** Spiral, Elliptical, and Odd objects

We employed five different classification models, including:

- Support Vector Machine (SVM)
- Classic Convolutional Neural Network (CNN)
- CNN - Transformer
- ResNet
- VGG

The SVM and classic CNN models utilized Zernike moments (ZMs) extracted from the images, whereas the CNN - Transformer, ResNet, and VGG models were designed based on the original images.

For more details on the algorithms, please refer to our paper: [Ghaderi, Alipour, and Safari](paperlink).

## Repository Structure

This repository includes two Jupyter notebooks:

- **Galaxy-Non-Galaxy Classification**: [galaxy_non_galaxy_classification.ipynb](path-to-notebook)
- **Galaxy Morphology Classification**: [galaxy_morphology_classification.ipynb](path-to-notebook)

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn tensorflow keras
