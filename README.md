# NODE-AdvGAN

## Overview
NODE-AdvGAN:  Improving the transferability and perceptual similarity of adversarial examples by dynamic-system-driven adversarial generative model

## Requirements
- Python == 3.9
- matplotlib == 3.9.2
- piqa == 1.3.2
- torch == 1.12.0
- torchdiffeq == 0.2.5
- torchsummary == 1.5.1
- torchvision == 0.13.0

## Pretrained Models

Please download the pretrained classifier models from the following link: https://drive.google.com/drive/folders/15IrcLN9QHoFHi_u9_adYxka2fMevwxnr?usp=sharing.

After downloading, place the models into the corresponding directories:
- For CIFAR-10 models, place the files in: `src/cifar10_models/state_dicts`
- For FMNIST models, place the files in: `src/FMNIST_models/state_dicts`

## Running NODE-AdvGAN for CIFAR-10

To run NODE-AdvGAN for CIFAR-10, use the following command:
```bash
python NODE_AdvGAN_CIFAR10.py
```
For NODE-AdvGAN-T with CIFAR-10, run the command with the `--l_inf_bound_train` parameter:

```bash
python NODE_AdvGAN_CIFAR10.py --l_inf_bound_train 0.03921569
```

Alternatively, you can modify the `l_inf_bound_train` parameter directly in the `NODE_AdvGAN_CIFAR10.py` file and set it to `10/255`.


## Running NODE-AdvGAN for FMNIST

To run NODE-AdvGAN for FMNIST, use the following command:
```bash
python NODE_AdvGAN_FMNIST.py
```
For NODE-AdvGAN-T with CIFAR-10, run the command with the `--l_inf_bound_train` parameter:

```bash
python NODE_AdvGAN_FMNIST.py --l_inf_bound_train 0.03921569
```

Alternatively, you can modify the `l_inf_bound_train` parameter directly in the `NODE_AdvGAN_FMNIST.py` file and set it to `10/255`.