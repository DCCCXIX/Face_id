import os
import cv2
import numpy as np
import torch
import json
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize, ToPILImage
from torchvision import utils
from torchvision import datasets, models, transforms
from tqdm import tqdm

class DataCollector():
    def __init__(self):

        self.data_dir = './images'
        self.image_size = 160

        self.data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomApply([AddGaussianNoise(0.5, 0.5)], p=0.5)
            ])
        
        self.facedict = {}
        self.training_data = []
        
        #build facedict from folders and save it for training
        for dirs in next(os.walk(self.data_dir))[1]:
            self.facedict[next(os.walk(self.data_dir))[1].index(str(dirs))] = dirs

        with open('facedict.json', 'w') as fp:
            json.dump(self.facedict, fp)

    def collect_data(self):
        for key in self.facedict:
            for f in tqdm(os.listdir(os.path.join(self.data_dir, self.facedict[key]))):
                try:
                    path = os.path.join(self.data_dir, self.facedict[key], f)
                    img = cv2.imread(path, 1)
                    img = self.data_transforms(img)
                    self.training_data.append([np.array(img), int(key)])
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('Done!')

class AddGaussianNoise(object):
    def __init__(self, mean=0.5, std=0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == '__main__':
    dc = DataCollector()
    dc.collect_data()
