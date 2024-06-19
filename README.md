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

## Authors

[Hamed Ghaderi](https://scholar.google.com/citations?user=G1jGaYcAAAAJ&hl=en), [Nasibe Alipour](https://scholar.google.com/citations?user=PfzZOI0AAAAJ&hl=en), [Hossein Safari](https://scholar.google.com/citations?user=nCc1FV8AAAAJ&hl=en)

## License
[MIT](https://choosealicense.com/licenses/mit/)


