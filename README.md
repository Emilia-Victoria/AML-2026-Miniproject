# AML-2026-Miniproject

**Project:** Generating novel Animal Crossing villagers with GANs

**Group members:** Emilia Victoria Helsted (ehel), Sebastian Cloos Hylander (sehy), Thorbjørn Peter Høgsbro Pedersen (tpep)

This repository contains our miniproject for the course **Advanced Machine Learning (Spring 2026)**. This project explores generative adversarial networks (GANs) to generate novel Animal Crossing-style villager images. The goal is to learn stylistic features from existing characters and generate novel, coherent designs.

## Dataset

Our dataset consists of unlabelled images of Animal Crossing villagers collected from [kaggle](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers/data). A limitation of our dataset is its relatively small size (392 images), which makes training GANs challenging and prone to overfitting or mode collapse. Another issue is that the images have varying sizes. To address this all images were padded with white pixels to make when square and then resized to 64x64 pixels. Additionally, images with transparent backgrounds were converted from RGBA to RGB to make all backgrounds fully white.

Although data augmentation is often useful, we chose not to apply transformations such as flipping or rotation. Since the images are full-body portraits and many villagers are symmetric,such augmentations would introduce unrealistic samples and potentially harm training.

- Image format: JPG
- Image resolution: Varies. Avg width: 216.67. Avg height: 348.82
- Dataset size: 392 images

## Models and Architectures

The standard GAN consists of a disciminator and a generator, trained in an adversarial manner. The discriminator is trained to differentiate between real and synthetic input, while the generator tries to generate output that fools the discriminator.

### Models

We have trained the following models:

- DCGAN
- WGAN-GP

### DCGAN

Our deep convolutional GAN (DCGAN) follows a standard arhcitecture.

### WGAN-GP

For our Wasserstein GAN with gradient penalty (WGAN-GP) we changed the discriminator to a critic that outputs a 

## Training Setup

We have used the following parameters:

- Latent dimension: 100
- Batch size: _
- Optimizer: Adam
- Learning rate: _
- Training epochs: _

For the WGAN-GP, the discriminator (critic) was updated multiple times per generator update, and a gradient penalty term was added to stabilize training.

## Observations and Results

### Example Outputs

## Discussion

## Key Takeaways
