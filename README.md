# Machine Learning for Morphological Galaxy Classification

The *"Machine learning for morphological galaxy classification"* is a repository for classifying [Galaxy Zoo 2](https://data.galaxyzoo.org/#section-7) (GZ2) images into **(1) Galaxy and Non-Galaxy**, and **(2) Galaxy in Spiral, Elliptical, and Odd objects** using the five state-of-the-art machine learning models.

## Overview

We employed five different classification models, including:

1. Support Vector Machine (SVM) with Zernike moments (ZMs)
2. 1D-Convolutional Neural Network (1D-CNN) with ZMs
3. 2D-CNN with Vision Transformer (ViT) and original images
4. ResNet50 with ViT and original images
5. VGG16 with ViT and original images

The SVM and 1D-CNN models utilized Zernike moments (ZMs) extracted from the images, while the 2D-CNN, ResNet5, and VGG16 with Vision Transformer (ViT) models were designed based on the original images.

For more details on the algorithms, please refer to our paper: [H. Ghaderi, N. Alipour, and H. Safari](https://arxiv.org/abs/2501.09816).

## Repository Structure

This repository includes two main Jupyter notebooks:

- **Galaxy-Non-Galaxy Classification**: [galaxy_nongalaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_nongalaxy_classifiers.ipynb)
- **Galaxy Classification**: [galaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_classifier.ipynb)

## Data
Please download the **Data** files from [this link](https://drive.google.com/drive/folders/1pwNk-8VJ-a_jUn84DyPYRYmhYSmki1dh?usp=sharing) that includes two categories:

1. **galaxy-nongalaxy**
2. **galaxy**

Each category contains two folders:

- **images**: This folder includes the original images for galaxy_nongalaxy and cropped images for galaxy classifiers.
- **ZMs**: This folder contains Zernike Moments (ZMs) data sets in CSV file format.

## Authors

[Hamed Ghaderi](https://scholar.google.com/citations?user=G1jGaYcAAAAJ&hl=en), [Nasibe Alipour](https://scholar.google.com/citations?user=PfzZOI0AAAAJ&hl=en), [Hossein Safari](https://scholar.google.com/citations?user=nCc1FV8AAAAJ&hl=en)

## License
[MIT](https://choosealicense.com/licenses/mit/)


