# Machine Learning for Morphological Galaxy Classification

The *"Machine learning for morphological galaxy classification"* is a repository for classifying [Galaxy Zoo 2](https://data.galaxyzoo.org/#section-7) (GZ2) images into **(1) Galaxy and Non-Galaxy**, and **(2) Galaxy in Spiral, Elliptical, and Odd objects** using the five state-of-the-art machine learning models.

## Overview

We employed five different classification models, including:

1. Support Vector Machine (SVM)
2. Classic 1D-Convolutional Neural Network (1D-CNN)
3. CNN - Vision Transformer
4. ResNet50 - Vision Transformer
5. VGG16 - Vision Transformer

The SVM and classic 1D-CNN models utilized Zernike moments (ZMs) extracted from the images, while the CNN - Vision Transformer, ResNet50 - Vision Transformer, and VGG16 - Vision Transformer models were designed based on the original images.

For more details on the algorithms, please refer to our paper: [Ghaderi, Alipour, and Safari](paperlink).

## Repository Structure

This repository includes two main Jupyter notebooks:

- **Galaxy-Non-Galaxy Classification**: [galaxy_nongalaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_nongalaxy_classifiers.ipynb)
- **Galaxy Classification**: [galaxy_classifiers.ipynb](https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification/blob/main/galaxy_classifier.ipynb)

## Data
Please download the **Data** files from [this link](https://drive.google.com/file/d/1wxmYQ8qpgaVDuD3kTeBrZlyny0IBA9wn/view?usp=drive_link) that includes two categories:

1. **galaxy-nongalaxy**
2. **galaxy**

Each category contains two folders:

- **image**: This folder includes the original images for galaxy_nongalaxy and cropped images for galaxy classifiers.
- **ZMs**: This folder contains Zernike Moments (ZMs) data sets in CSV file format.

## Authors

[Hamed Ghaderi](https://scholar.google.com/citations?user=G1jGaYcAAAAJ&hl=en), [Nasibe Alipour](https://scholar.google.com/citations?user=PfzZOI0AAAAJ&hl=en), [Hossein Safari](https://scholar.google.com/citations?user=nCc1FV8AAAAJ&hl=en)

## License
[MIT](https://choosealicense.com/licenses/mit/)


