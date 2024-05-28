########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import torch, types, os
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
import torchvision as vision
import torchvision.transforms as transforms
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print(f'loading...')

########################################################################################################

model_prefix = 'test/image_trained/out-v7c_d8_256-224-13bit-OB32x0.5-201'
input_img = 'test/img_ae_test/test0.png'

########################################################################################################

class ToBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x + 0.5) # no need for noise when we have plenty of data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() # pass-through

class R_ENCODER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.Bxx = nn.BatchNorm2d(dd*64)

        self.CIN = nn.Conv2d(3, dd, kernel_size=3, padding=1)
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)

        self.B00 = nn.BatchNorm2d(dd*4)
        self.C00 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)

        self.B10 = nn.BatchNorm2d(dd*16)
        self.C10 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)

        self.B20 = nn.BatchNorm2d(dd*64)
        self.C20 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)

        self.COUT = nn.Conv2d(dd*64, args.my_img_bit, kernel_size=3, padding=1)

    def forward(self, img):
        ACT = F.mish

        x = self.CIN(img)
        xx = self.Bxx(F.pixel_unshuffle(x, 8))
        x = x + self.Cx1(ACT(self.Cx0(x)))

        x = F.pixel_unshuffle(x, 2)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))

        x = F.pixel_unshuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))

        x = F.pixel_unshuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))

        x = self.COUT(x + xx)
        return torch.sigmoid(x)

class R_DECODER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        dd = 8
        self.CIN = nn.Conv2d(args.my_img_bit, dd*64, kernel_size=3, padding=1)

        self.B00 = nn.BatchNorm2d(dd*64)
        self.C00 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)

        self.B10 = nn.BatchNorm2d(dd*16)
        self.C10 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)

        self.B20 = nn.BatchNorm2d(dd*4)
        self.C20 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)

        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)
        self.COUT = nn.Conv2d(dd, 3, kernel_size=3, padding=1)

    def forward(self, code):
        ACT = F.mish
        x = self.CIN(code)

        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))
        x = x + self.C03(ACT(self.C02(x)))
        x = F.pixel_shuffle(x, 2)

        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))
        x = x + self.C13(ACT(self.C12(x)))
        x = F.pixel_shuffle(x, 2)

        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))
        x = x + self.C23(ACT(self.C22(x)))
        x = F.pixel_shuffle(x, 2)

        x = x + self.Cx1(ACT(self.Cx0(x)))
        x = self.COUT(x)
        
        return torch.sigmoid(x)

########################################################################################################

print(f'building model...')
args = types.SimpleNamespace()
args.my_img_bit = 13
encoder = R_ENCODER(args).eval().cuda()
decoder = R_DECODER(args).eval().cuda()

zpow = torch.tensor([2**i for i in range(0,13)]).reshape(13,1,1).cuda().long()

encoder.load_state_dict(torch.load(f'{model_prefix}-E.pth'))
decoder.load_state_dict(torch.load(f'{model_prefix}-D.pth'))

########################################################################################################

print(f'test image...')
img_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((224, 224))
])

with torch.no_grad():
    img = img_transform(Image.open(input_img)).unsqueeze(0).cuda()
    z = encoder(img)
    z = ToBinary.apply(z)

    zz = torch.sum(z.squeeze().long() * zpow, dim=0)
    print(f'Code shape = {zz.shape}\n{zz.cpu().numpy()}\n')
    
    out = decoder(z)
    vision.utils.save_image(out, f"{input_img.split('.')[0]}-out-13bit.jpg")
