import os
import sys
sys.path.insert(0, 'U-2-Net')

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

from data_loader import RescaleT
from data_loader import ToTensorLab

from model import U2NET  # full size version 173.6 MB
# from model import U2NETP # small version u2net 4.7 MB

model_dir = os.path.join(os.getcwd(), 'U-2-Net',
                         'saved_models', 'u2net', 'u2net.pth')

print("Loading U-2-Net...")
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_dir, map_location='cpu'))

if torch.cuda.is_available():
    print("Cuda is available")
    net.cuda()
else:
    print("Cuda is NOT available")

net.eval()
print("U-2-Net is ready")


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    #print('Start image preprocess')
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3

    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({
        'imidx': np.array([0]),
        'image': image,
        'label': label
    })
    #print('Preprocess completed')

    return sample


def run(img):
    torch.cuda.empty_cache()

    sample = preprocess(img)
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    #print('Run U^2-Net')
    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

    # Normalization.
    # print('Normalization')
    pred = d1[:, 0, :, :]
    predict = normPRED(pred)

    # Formatting data
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # Cleanup.
    del d1, d2, d3, d4, d5, d6, d7
    torch.cuda.empty_cache()

    # Return the mask as numpy array with values within
    # the range [0.0, 1.0] with format float32
    return predict_np
