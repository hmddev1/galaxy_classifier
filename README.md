<!-- # Machine learning for morphological galaxy classification 

The *"Machine learning for morphological galaxy classification"* is a repository for classifying [Galaxy Zoo 2](GZ2) images into (1) Galaxy and Non-Galaxy, and (2) Galaxy in Spiral, Elliptical, and Odd objects. We used five classification models including Supprot Vector Machine (SVM) and ..... The SVM and classic CNN used the Zernike moments (ZMs) extracted from images, while CNN - transformer and resnet and vgg .... designed base on original images. 

The details of algorithms where explained in [Ghaderi, Alipour, and Safari](paperlink).

This repository includes two python notebooks of the classifiers for galaxy-non-galaxy classification (ntebook name) and galaxies classification (notebook name). -->


# Machine Learning for Morphological Galaxy Classification

The *"Machine learning for morphological galaxy classification"* is a repository for classifying [Galaxy Zoo 2](https://data.galaxyzoo.org/#section-7) (GZ2) images into **(1) Galaxy and Non-Galaxy**, and **(2) Galaxy in Spiral, Elliptical, and Odd objects** using the five state-of-the-art machine learning models.

## Overview

We employed five different classification models, including:

- Support Vector Machine (SVM)
- Classic 1D-Convolutional Neural Network (1D-CNN)
- CNN - Vision Transformer
- ResNet50 - Vision Transformer
- VGG16 - Vision Transformer

The SVM and classic 1D-CNN models utilized Zernike moments (ZMs) extracted from the images, while the CNN - Vision Transformer, ResNet50 - Vision Transformer, and VGG16 - Vision Transformer models were designed based on the original images.

For more details on the algorithms, please refer to our paper: [Ghaderi, Alipour, and Safari](paperlink).

## Repository Structure

This repository includes two main Jupyter notebooks:

- **Galaxy-Non-Galaxy Classification**: [galaxy_nongalaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_nongalaxy_classifiers.ipynb)
- **Galaxy Classification**: [galaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_classifier.ipynb)

The Data includes: 

galaxy ZMs
galaxy non-galaxy


## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn tensorflow keras
