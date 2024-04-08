import os
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image


class Resnet50():
    def __init__(self, config):
        self.newid2imgnum = config['newid2imgnum']
        # self.model = models.resnet50(pretrained=True).cuda()
        # self.model.fc = nn.Linear(2048, 300).cuda()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 300)
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/src - 20240125/dataset/pheme/pheme_image/pheme_images_jpg/'
        self.trans = self.img_trans()

    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def forward(self, xtid):
        img_path = []
        img_list = []
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        # batch_img = torch.cat(img_list, dim=0).cuda()
        batch_img = torch.cat(img_list, dim=0)
        img_output = self.model(batch_img).cuda()
        return img_output