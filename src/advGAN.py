from utility import load_target_model
import time

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import wandb
from piqa import SSIM, MS_SSIM, PSNR

def init_weights(m):
    '''
        Custom weights initialization called on G and D
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class AdvGAN_Attack:
    def __init__(
            self,
            device,
            model,
            n_labels,
            n_channels,
            args
    ):
        self.args = args
        self.device = device
        self.n_labels = n_labels
        self.model = model
        self.epochs = args.epochs
        self.target = args.target

        self.lr = args.lr

        self.l_inf_bound_train = args.l_inf_bound_train

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.kappa = args.kappa
        self.c = args.c
        self.n_steps_D = args.n_steps_D
        self.n_steps_G = args.n_steps_G

        if args.target == 'CIFAR10':
            self.ssim = SSIM().to(device)
            self.ms_ssim = MS_SSIM().to(device)
            self.psnr = PSNR().to(device)
        elif args.target == 'FMNIST':
            self.ssim = SSIM(n_channels=1).to(device)
            self.ms_ssim = MS_SSIM(n_channels=1).to(device)
            self.psnr = PSNR().to(device)

        if 'NODE' in args.G_model:
            G_model_zoo = {
                "NODE_AdvGAN": models.NODE_AdvGAN(args),
            }
            self.G = G_model_zoo[args.G_model].to(device)
            print(f'we are choosing {args.G_model} model as generator now.')

        self.D = models.Discriminator(n_channels).to(device)
        self.D.apply(init_weights)

        # initialize optimizers

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        if args.lr_halve:
            self.lr_scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G,
                                                                  step_size=args.lr_h_n_steps, gamma=args.lr_h_rate)
            self.lr_scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D,
                                                                  step_size=args.lr_h_n_steps, gamma=args.lr_h_rate)

        self.save_dir = args.save_dir
        self.models_path = self.save_dir + 'checkpoints/'
        self.losses_path = self.save_dir + f'loss_results/'
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

        if not os.path.exists(self.losses_path):
            os.makedirs(self.losses_path)

    def train_gan_batch(self, x, labels, epoch):
        # optimize D

        perturbation = self.G(x)

        if self.args.training_clamp:
            perturbation = torch.clamp(perturbation, -self.l_inf_bound_train, self.l_inf_bound_train)
            adv_images = torch.clamp(perturbation + x, 0, 1)
        else:
            adv_images = perturbation + x
            adv_images = torch.clamp(adv_images, 0, 1)

        logits_real, pred_real = self.D(x)
        logits_fake, pred_fake = self.D(adv_images.detach())

        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D = 1 / 2 * (loss_D_fake + loss_D_real)

        if (epoch - 1) % self.n_steps_D == 0:
            loss_D.backward()
            self.optimizer_D.step()

        self.G.zero_grad()
        # the Hinge Loss part of L
        perturbation_norm_per_sample = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1)
        loss_hinge_per_sample = torch.max(torch.zeros_like(perturbation_norm_per_sample),
                                          perturbation_norm_per_sample - self.c)
        loss_hinge = torch.mean(loss_hinge_per_sample)
        # C&W loss
        if hasattr(self.args, 'is_targeted_attack') and self.args.is_targeted_attack:
            target_labels = torch.full((labels.shape[0],), self.args.target_label, dtype=torch.long).to(self.device)
            logits_model = self.model(adv_images)
            target_labels_one_hot = F.one_hot(target_labels, num_classes=self.n_labels)
            target_logits = torch.sum(target_labels_one_hot * logits_model, dim=1)
            max_other_logits, _ = torch.max((1 - target_labels_one_hot) * logits_model - target_labels_one_hot * 10000,
                                           dim=1)
            loss_adv_ = torch.max(max_other_logits - target_logits, self.kappa * torch.ones_like(target_logits))
            loss_adv = torch.mean(loss_adv_)
        else:
            logits_model = self.model(adv_images)
            true_labels_one_hot = F.one_hot(labels, num_classes=self.n_labels)
            real_logits = torch.sum(true_labels_one_hot * logits_model, dim=1)
            target_class_logits, _ = torch.max((1 - true_labels_one_hot) * logits_model - true_labels_one_hot * 10000,
                                               dim=1)
            loss_adv_ = torch.max(real_logits - target_class_logits, self.kappa * torch.ones_like(real_logits))
            loss_adv = torch.mean(loss_adv_)
        #######################################
        # the GAN Loss part of L
        logits_fake, pred_fake = self.D(adv_images)
        loss_G_gan = 1 / 2 * F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_G = self.gamma * loss_adv + self.alpha * loss_G_gan + self.beta * loss_hinge
        if (epoch - 1) % self.n_steps_G == 0:
            loss_G.backward()
            self.optimizer_G.step()
        return loss_D.item(), loss_G.item(), loss_G_gan.item(), loss_hinge.item(), loss_adv.item()

    def train(self, train_dataloader, val_dataloader):
        log = {}
        for epoch in range(1, self.epochs + 1):
            self.G.train()
            stats = ['loss_D', 'loss_G', 'loss_G_gan', 'loss_hinge', 'loss_adv']
            meters_trn = {stat: AverageMeter() for stat in stats}
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                loss_D, loss_G, loss_G_gan, loss_hinge, loss_adv = self.train_gan_batch(
                            images, labels, epoch)
                for s in stats:
                    v = locals()[s]
                    meters_trn[s].update(v, self.args.batch_size)
            if self.args.lr_halve:
                self.lr_scheduler_G.step()
                self.lr_scheduler_D.step()
            # print statistics
            batch_size = len(train_dataloader)
            print(f'Epoch {epoch}:')
            for s in stats:
                print(f'{s}: {meters_trn[s].avg}')
                log[s] = meters_trn[s].avg
            current_lr_G = self.optimizer_G.param_groups[0]['lr']
            current_lr_D = self.optimizer_D.param_groups[0]['lr']
            print("Current learning rate for Generator: ", current_lr_G)
            print("Current learning rate for Discriminator: ", current_lr_D)
            self.G.eval()  # Set the model to evaluation
            correct = 0
            total = 0
            with torch.no_grad():
                correct, total, ssim_values, psnr_values = self.evaluate_target_model_and_similarity(
                    correct, total, self.model, val_dataloader, self.G, self.args)
            print(f'ssim_values: {ssim_values.avg}')
            print(f'psnr_values: {psnr_values.avg}')
            log['ssim_values'] = ssim_values.avg
            log['psnr_values'] = psnr_values.avg

            if hasattr(self.args, 'is_targeted_attack') and self.args.is_targeted_attack:
                print(f'Attack success rate for evaluation dataset: {100 * (correct / total)} %')
                log['Attack_success_rate'] = correct / total
            else:
                print(f'Attack success rate for evaluation dataset: {100 * (1 - correct / total)} %')
                log['Attack_success_rate'] = 1 - correct / total

            if self.args.test_transferability:
                if epoch % self.args.test_transferability_per_times == 0:
                    for transfer_model_name in self.args.transfer_model_names:
                        with torch.no_grad():
                            transfer_model = load_target_model(transfer_model_name, dataset = self.args.target).to(self.device)
                            correct, total = 0, 0
                            correct, total = self.evaluate_target_model(correct, total, transfer_model,
                                                                        val_dataloader, self.G, self.args)
                        if hasattr(self.args, 'is_targeted_attack') and self.args.is_targeted_attack:
                            attack_success_rate = 100 * (correct / total)
                        else:
                            attack_success_rate = 100 * (1 - correct / total)
                        log[transfer_model_name] = attack_success_rate
                        print(
                            f'Attack success rate for {transfer_model_name} in evaluation dataset: {attack_success_rate} %')

            if self.args.use_wandb:
                wandb.log(log)

            # save generator
            g_filename = '{}G_epoch_{}.pth'.format(self.models_path, str(epoch))
            torch.save(self.G.state_dict(), g_filename)
            d_filename = '{}D_epoch_{}.pth'.format(self.models_path, str(epoch))
            torch.save(self.D.state_dict(), d_filename)

    def evaluate_target_model(self, correct, total, model, val_dataloader, G_model, args, l_inf_bound = None):
        if l_inf_bound is None:
            l_inf_bound = args.l_inf_bound
        if hasattr(self.args, 'is_targeted_attack') and self.args.is_targeted_attack:
            for data in val_dataloader:
                images, _ = data
                labels = torch.full((images.shape[0],), self.args.target_label, dtype=torch.long)
                images = images.to(self.device)
                labels = labels.to(self.device)

                perturbation = G_model(images)
                adv_images = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + images
                adv_images = torch.clamp(adv_images, 0, 1)
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        else:
            for data in val_dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                perturbation = G_model(images)
                adv_images = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + images
                adv_images = torch.clamp(adv_images, 0, 1)
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct, total

    def evaluate_target_model_and_similarity(self,
                                             correct, total, model, val_dataloader, G_model, args, l_inf_bound = None):
        if l_inf_bound is None:
            l_inf_bound = args.l_inf_bound
        if hasattr(self.args, 'is_targeted_attack') and self.args.is_targeted_attack:
            ssim_values = AverageMeter()
            psnr_values = AverageMeter()
            for data in val_dataloader:
                images, _ = data
                labels = torch.full((images.shape[0],), self.args.target_label, dtype=torch.long)
                images = images.to(self.device)
                labels = labels.to(self.device)

                perturbation = G_model(images)
                adv_images = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + images
                adv_images = torch.clamp(adv_images, 0, 1)
                ssim_value = self.ssim(images, adv_images).mean()
                psnr_value = self.psnr(images, adv_images).mean()
                ssim_values.update(ssim_value.item(), images.size(0))
                psnr_values.update(psnr_value.item(), images.size(0))
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        else:
            times_per_iteration = []
            ssim_values = AverageMeter()
            psnr_values = AverageMeter()
            for data in val_dataloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                start_time = time.time()

                perturbation = G_model(images)
                end_time = time.time()  # Capture the end time for this iteration
                elapsed_time = end_time - start_time  # Calculate the elapsed time for this iteration
                times_per_iteration.append(elapsed_time)

                adv_images = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + images
                adv_images = torch.clamp(adv_images, 0, 1)
                ssim_value = self.ssim(images, adv_images).mean()
                psnr_value = self.psnr(images, adv_images).mean()
                ssim_values.update(ssim_value.item(), images.size(0))
                psnr_values.update(psnr_value.item(), images.size(0))
                outputs = model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            total_time = sum(times_per_iteration)
            print(f"Total time for all iterations: {total_time} seconds")
        return correct, total, ssim_values, psnr_values

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

