import torchvision.datasets

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import os, time, glob
import numpy as np
import torch
import torch.nn as nn


def init_params(args):
    if args.target == 'FMNIST':
        batch_size = args.batch_size

        n_labels = 10
        n_channels = 1

        if args.add_transform:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])

        train_dataset = torchvision.datasets.FashionMNIST('./datasets', train=True, transform=train_transform,
                                                     download=True)
        test_dataset = torchvision.datasets.FashionMNIST('./datasets', train=False, transform=test_transform,
                                                    download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    elif args.target == 'CIFAR10':
        batch_size = args.batch_size

        n_labels = 10
        n_channels = 3

        if args.add_transform:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transform,
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transforms.ToTensor(),
                                                    download=True)

        if hasattr(args, 'use_partial_data') and args.use_partial_data:
            np.random.seed(args.seed)
            num_train_samples = len(train_dataset)
            indices = list(range(num_train_samples))
            np.random.shuffle(indices)
            split = int(num_train_samples * args.p_samples)
            train_indices = indices[:split]
            train_dataset = Subset(train_dataset, train_indices)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            total_samples = 0
            for batch in train_dataloader:
                inputs, labels = batch
                total_samples += inputs.size(0)
            print(f'Total number of samples in train_dataloader: {total_samples}')
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        if hasattr(args, 'is_targeted_attack') and args.is_targeted_attack:
            indices = [i for i, label in enumerate(test_dataset.targets) if label != args.target_label]
            filtered_test_dataset = Subset(test_dataset, indices)
            test_dataloader = DataLoader(filtered_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    else:
        raise NotImplementedError('Unknown Dataset')

    return train_dataloader, test_dataloader, n_labels, n_channels, len(test_dataset)


def test_target_model_performance(args, dataloader, target_model, dataset_size):
    n_correct = 0
    dataset_size = 0
    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(args.device), true_label.to(args.device)
        pred_label = torch.argmax(target_model(img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)
        dataset_size += img.shape[0]
    accuracy = 100.0 * n_correct / dataset_size
    print(f'Accuracy of the model on the dataset: {accuracy:.2f}%, '
          f'n_correct: {n_correct} and dataset_size:{dataset_size}')

def test_targeted_attack_performance(args, dataloader, target_model):
    n_correct = 0
    dataset_size = 0
    for i, data in enumerate(dataloader, 0):
        img, _ = data
        labels = torch.full((img.shape[0],), args.target_label, dtype=torch.long)
        img, labels = img.to(args.device), labels.to(args.device)
        pred_label = torch.argmax(target_model(img), 1)
        n_correct += torch.sum(pred_label == labels, 0)
        dataset_size += img.shape[0]
    accuracy = 100.0 * n_correct / dataset_size
    print(f'Attack SR of the model on the dataset: {accuracy:.2f}%, '
          f'n_correct: {n_correct} and dataset_size:{dataset_size}')
    return accuracy.item()

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()

        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


def load_target_model(model_name, dataset = 'CIFAR10'):
    if dataset == 'FMNIST':
        from FMNIST_models.densenet import densenet121, densenet161, densenet169
        from FMNIST_models.googlenet import googlenet
        from FMNIST_models.inception import inception_v3
        from FMNIST_models.mobilenetv2 import mobilenet_v2
        from FMNIST_models.resnet import resnet18, resnet34, resnet50
        from FMNIST_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

        all_classifiers = {
            "vgg11_bn": vgg11_bn(),
            "vgg13_bn": vgg13_bn(),
            "vgg16_bn": vgg16_bn(),
            "vgg19_bn": vgg19_bn(),
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "densenet121": densenet121(),
            "densenet161": densenet161(),
            "densenet169": densenet169(),
            "mobilenet_v2": mobilenet_v2(),
            "googlenet": googlenet(),
            "inception_v3": inception_v3(),
        }

        model = all_classifiers[model_name]
        state_dict = os.path.join(
            "FMNIST_models", "state_dicts", model_name + ".pt"
        )
        model.load_state_dict(torch.load(state_dict))
        mean = [0.286]
        std = [0.353]

    elif dataset == 'CIFAR10':
        from cifar10_models.densenet import densenet121, densenet161, densenet169
        from cifar10_models.googlenet import googlenet
        from cifar10_models.inception import inception_v3
        from cifar10_models.mobilenetv2 import mobilenet_v2
        from cifar10_models.resnet import resnet18, resnet34, resnet50
        from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

        all_classifiers = {
            "vgg11_bn": vgg11_bn(),
            "vgg13_bn": vgg13_bn(),
            "vgg16_bn": vgg16_bn(),
            "vgg19_bn": vgg19_bn(),
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "densenet121": densenet121(),
            "densenet161": densenet161(),
            "densenet169": densenet169(),
            "mobilenet_v2": mobilenet_v2(),
            "googlenet": googlenet(),
            "inception_v3": inception_v3(),
        }

        model = all_classifiers[model_name]
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", model_name + ".pt"
        )
        model.load_state_dict(torch.load(state_dict))
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
    else:
        raise NotImplementedError('Unknown Dataset')
    model = nn.Sequential(Normalize(mean=mean, std=std),
                          model)
    model.eval()
    return model

