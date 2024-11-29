import torch
import os
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

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

class NODE_AdvGAN(nn.Module):
    '''
    The generic CNN module for learning with Neural ODEs.

    This class wraps the `_F` field that acts like an RNN-Cell. This method simply initialises the hidden dynamics
    and computes the updated hidden dynamics through a call to `ode_int` using the `_F` as the function that
    computes the update.
    '''

    def __init__(self, args):
        super().__init__()
        self.odeint_adjoint = args.odeint_adjoint
        self.t = args.t
        self.N_t = args.N_t
        self.solver = args.solver
        self.ODE_vector_field = globals()[args.ODE_vector_field](args)
        self.dev = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, image: torch.Tensor):
        if self.odeint_adjoint:
            odeint_solver = odeint_adjoint
        else:
            odeint_solver = odeint
        timesteps = torch.linspace(0, self.t, self.N_t + 1).to(self.dev)
        out = odeint_solver(func=self.ODE_vector_field, y0=image, t=timesteps, method=self.solver)[-1]
        return out - image


class _VectorField(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Conv2D_6to128 = nn.Conv2d(in_channels=args.num_channel + 1, out_channels=32, kernel_size=3, stride=1,
                                       padding=1)
        self.BatchNorm2d_128_1 = nn.BatchNorm2d(32, eps=1e-3)
        self.Conv2D_128to128_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3,
                                           dilation=3)
        self.BatchNorm2d_128_2 = nn.BatchNorm2d(64, eps=1e-3)
        self.Conv2D_128to128_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3,
                                           dilation=3)
        self.BatchNorm2d_128_3 = nn.BatchNorm2d(64, eps=1e-3)
        self.Conv2D_128to128_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3,
                                           dilation=3)
        self.BatchNorm2d_128_7 = nn.BatchNorm2d(64, eps=1e-3)
        self.Conv2D_128to128_7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,  padding=3,
                                           dilation=3)
        self.BatchNorm2d_128_8 = nn.BatchNorm2d(32, eps=1e-3)
        self.Conv2D_128to3 = nn.Conv2d(in_channels=32, out_channels=args.num_channel, kernel_size=3, stride=1,
                                       padding=1)
        if hasattr(args, 'is_initialize') and args.is_initialize:
            self._initialize_weights()
        self.dev = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.dev)

    def forward(self, t, h):
        # 1st layer, Conv+relu
        input = self.concat_t(h, t)  # 6
        output = self.Conv2D_6to128(input)
        output = F.relu(self.BatchNorm2d_128_1(output))
        output = self.Conv2D_128to128_1(output)
        output = F.relu(self.BatchNorm2d_128_2(output))
        output = self.Conv2D_128to128_2(output)
        output = F.relu(self.BatchNorm2d_128_3(output))
        output = self.Conv2D_128to128_3(output)
        output = F.relu(self.BatchNorm2d_128_7(output))
        output = self.Conv2D_128to128_7(output)
        output = F.relu(self.BatchNorm2d_128_8(output))
        output = self.Conv2D_128to3(output)
        return output

    def concat_t(self, h: torch.Tensor, t):
        h_shape = h.shape
        tt = torch.ones(h_shape[0], 1, h_shape[2], h_shape[3]).to(self.dev) * t
        out_ = torch.cat((h, tt), dim=1).float()  # shape =[bach_size, features+1, N_x, N_y]
        return out_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # self.apply(init_weights)


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            # 8*16*16
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*8*8
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32*4*4
            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0),
            # 1*1*1
        ]

        self.model = nn.Sequential(*model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.model(x)
        output = self.sigmoid(logits).squeeze()
        return logits, output
