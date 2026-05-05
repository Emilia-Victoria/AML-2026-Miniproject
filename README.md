# AML-2026-Miniproject

**Project:** Generating novel Animal Crossing villagers with GANs

**Group members:** Emilia Victoria Helsted (ehel), Sebastian Cloos Hylander (sehy), Thorbjørn Peter Høgsbro Pedersen (tpep)

This repository contains our miniproject for the course **Advanced Machine Learning (Spring 2026)**. This project explores generative adversarial networks (GANs) to generate novel Animal Crossing-style villager images. The goal is to learn stylistic features from existing characters and generate novel, coherent designs.

## Dataset

Our dataset consists of unlabelled images of Animal Crossing villagers collected from [kaggle](https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers/data). A limitation of our dataset is its relatively small size (392 images), which makes training GANs challenging and prone to overfitting or mode collapse. Another issue is that the images have varying sizes. To address this all images were padded with white pixels to make when square and then resized to 64x64 pixels. Additionally, images with transparent backgrounds were converted from RGBA to RGB to make all backgrounds fully white.

- Image format: JPG
- Image resolution: Varies. Avg width: 216.67. Avg height: 348.82
- Dataset size: 392 images

**Insert pretty picture of selected images!!!**

We chose not to apply direct transformations to the dataset such as flipping or rotation. Since the images are full-body portraits and many villagers are symmetric, such augmentations would introduce unrealistic samples and potentially harm training. Instead we make use of differentiable augmentation.

### Differentiable Augmentation

Differentiable augmentation applies the same differentiable augmentation to both real and  fake images for training of the discriminator and generator.

**Insert pretty picture of the augmentation here!!!**

## Models and Architectures

The standard GAN consists of a disciminator and a generator, trained in an adversarial manner. The discriminator is trained to differentiate between real and synthetic input, while the generator tries to generate output that fools the discriminator.

### Models

We have trained the following models:

- DCGAN
- WGAN-GP

### DCGAN Architecture

Our deep convolutional GAN (DCGAN) follows a standard arhcitecture.

### WGAN-GP Architecture

For our Wasserstein GAN with gradient penalty (WGAN-GP), we changed the discriminator to a critic that outputs a real-valued score instead of a probability. To do this we have removed the batch normalization and the final Sigmoid activation from the discriminator.

## Training Setup

We have used the following parameters for both models:

- Latent dimension: 100
- Batch size: _
- Optimizer: Adam
- Learning rate: _
- Training epochs: _

For each epoch we train the discriminator first followed by training of the generator.

### DCGAN Training

- Loss function: binary cross-entropy loss
- Optimizer hyperparameters: _

### WGAN-GP Training

For the WGAN-GP, the discriminator (critic) was updated multiple times per epoch, and a gradient penalty term was introduced to stabilize training.

- Loss function: Wasserstein loss
- Optimizer hyperparameters: _
- Ratio for critic: 5
- Lambda for gradients penalty: 10

## Observations and Results

### Example Outputs

## Discussion

## Key Takeaways

## Relevant Litterature

- Ou, Xunxiong (2024).
  [*Deep Convolutional Generative Adversarial Networks (DCGAN)-Based Anime Face Generation*](https://www.atlantis-press.com/proceedings/iciaai-24/126004091)
- Zhao, Shengyu et al. (2020).  
  [*Differentiable Augmentation for Data-Efficient GAN Training*](https://arxiv.org/abs/2006.10738)
- [GAN — Wasserstein GAN & WGAN-GP](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490) (Medium article)
- [Tackling Mode Collapse in GANs: From DCGAN to WGAN-GP](https://aneelabashir425.medium.com/medium-article-tackling-mode-collapse-in-gans-from-dcgan-to-wgan-gp-0b31c7ac3692) (Medium article)
