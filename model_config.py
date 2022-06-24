import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from efficientnet_pytorch import EfficientNet

# In[2]:


from torch.autograd import Variable


pretrained_size = (448,448)
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]


def get_model():

    # pretrained_model = models.resnet34(pretrained = True)
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
    #model = model.to(device)
    model.load_state_dict(torch.load('model_effb1.pt'))
    return model

def get_model_8class():

    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=8)
    model.load_state_dict(torch.load('model_effb4_8class.pt'))
    return model
