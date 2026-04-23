# AML-2026-Miniproject

**Project:** Generate novel Animal Crossing villagers using a DCGAN

**Group members:** Emilia Victoria Helsted (ehel), Sebastian Cloos Hylander (sehy), Thorbjørn Peter Høgsbro Pedersen (tpep)

This repository contains our miniproject for the course **Advanced Machine Learning (Spring 2026)**. The goal of this project is to train a Deep Convolutional GAN (DCGAN) to generate images of new Animal Crossing villagers. We will explore how well the model learns the visual structure and style of the villagers and produce novel characters that resemble the original dataset in style and structure.

## Dataset

Our dataset consists of unlabelled images of Animal Crossing villagers collected from [kaggle](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers/data). A limitation of our dataset is its rather small size. Another issue is that the images have varying resolution. To solve the issue of varying resolution we are preprocessing of the images, such that they are padded with white pixels to make them square and then resized to be 64x64. The backgrounds of many of the images were transparent so we changed them from RGBA to RGB. Due to the images being full-body portraits, with many of them being symmetric, we did not think it made sense to do data augmentation through flipping and rotating the images.

- Image format: JPG
- Image resolution: Varies. Avg width: 216.67. Avg height: 348.82
- Dataset size: 392 images
