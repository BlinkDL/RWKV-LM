########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc
import torchvision as vision
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_msssim import SSIM

class To2Bin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x + torch.empty_like(x).uniform_(0, 1))
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

class RWKV_IMG(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.e0b0 = nn.BatchNorm2d(12)
        self.e0w0 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.e0b1 = nn.BatchNorm2d(12)
        self.e0w1 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.e0b2 = nn.BatchNorm2d(12)
        self.e0w2 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.e0b3 = nn.BatchNorm2d(12)
        self.e0w3 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)

        self.e1b0 = nn.BatchNorm2d(48)
        self.e1w0 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.e1b1 = nn.BatchNorm2d(48)
        self.e1w1 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.e1b2 = nn.BatchNorm2d(48)
        self.e1w2 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.e1b3 = nn.BatchNorm2d(48)
        self.e1w3 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)

        self.e2b0 = nn.BatchNorm2d(192)
        self.e2w0 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.e2b1 = nn.BatchNorm2d(192)
        self.e2w1 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.e2b2 = nn.BatchNorm2d(192)
        self.e2w2 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.e2b3 = nn.BatchNorm2d(192)
        self.e2w3 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)

        self.ewww = nn.Conv2d(192, 8, kernel_size = 3, stride = 1, padding = 1)

        self.dwww = nn.Conv2d(8, 192, kernel_size = 3, stride = 1, padding = 1)

        self.d0b0 = nn.BatchNorm2d(192)
        self.d0w0 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.d0b1 = nn.BatchNorm2d(192)
        self.d0w1 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.d0b2 = nn.BatchNorm2d(192)
        self.d0w2 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)
        self.d0b3 = nn.BatchNorm2d(192)
        self.d0w3 = nn.Conv2d(192, 192, kernel_size = 3, stride = 1, padding = 1)

        self.d1b0 = nn.BatchNorm2d(48)
        self.d1w0 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.d1b1 = nn.BatchNorm2d(48)
        self.d1w1 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.d1b2 = nn.BatchNorm2d(48)
        self.d1w2 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)
        self.d1b3 = nn.BatchNorm2d(48)
        self.d1w3 = nn.Conv2d(48, 48, kernel_size = 3, stride = 1, padding = 1)

        self.d2b0 = nn.BatchNorm2d(12)
        self.d2w0 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.d2b1 = nn.BatchNorm2d(12)
        self.d2w1 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.d2b2 = nn.BatchNorm2d(12)
        self.d2w2 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)
        self.d2b3 = nn.BatchNorm2d(12)
        self.d2w3 = nn.Conv2d(12, 12, kernel_size = 3, stride = 1, padding = 1)

        self.SSIM = SSIM(data_range=1, size_average=True, channel=3)

    def configure_optimizers(self):
        args = self.args
        optim_groups = [
            {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},
        ]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def forward(self, img):
        x = img

        x = F.pixel_unshuffle(x, 2)
        x = x + self.e0w1(F.mish(self.e0b1(self.e0w0(F.mish(self.e0b0(x))))))
        x = x + self.e0w3(F.mish(self.e0b3(self.e0w2(F.mish(self.e0b2(x))))))

        x = F.pixel_unshuffle(x, 2)
        x = x + self.e1w1(F.mish(self.e1b1(self.e1w0(F.mish(self.e1b0(x))))))
        x = x + self.e1w3(F.mish(self.e1b3(self.e1w2(F.mish(self.e1b2(x))))))

        x = F.pixel_unshuffle(x, 2)
        x = x + self.e2w1(F.mish(self.e2b1(self.e2w0(F.mish(self.e2b0(x))))))
        x = x + self.e2w3(F.mish(self.e2b3(self.e2w2(F.mish(self.e2b2(x))))))

        x = self.ewww(x)

        x = To2Bin.apply(torch.sigmoid(x))
        # print(x.shape, x)

        x = self.dwww(x)

        x = x + self.d0w1(F.mish(self.d0b1(self.d0w0(F.mish(self.d0b0(x))))))
        x = x + self.d0w3(F.mish(self.d0b3(self.d0w2(F.mish(self.d0b2(x))))))
        x = F.pixel_shuffle(x, 2)

        x = x + self.d1w1(F.mish(self.d1b1(self.d1w0(F.mish(self.d1b0(x))))))
        x = x + self.d1w3(F.mish(self.d1b3(self.d1w2(F.mish(self.d1b2(x))))))
        x = F.pixel_shuffle(x, 2)

        x = x + self.d2w1(F.mish(self.d2b1(self.d2w0(F.mish(self.d2b0(x))))))
        x = x + self.d2w3(F.mish(self.d2b3(self.d2w2(F.mish(self.d2b2(x))))))
        x = F.pixel_shuffle(x, 2)

        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        args = self.args
        img, txt = batch
        out = self(img)
        if self.trainer.is_global_zero:
            if (self.trainer.global_step+1) % (100 * int(args.devices)) == 0:
                vision.utils.save_image(img[:4], f"test/image_model/{self.trainer.global_step}-src.jpg")
                vision.utils.save_image(out[:4], f"test/image_model/{self.trainer.global_step}-out.jpg")

        return 1 - self.SSIM(out.float(), img.float())

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape
            
            m[n] = p

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

        gc.collect()
        torch.cuda.empty_cache()
        return m
