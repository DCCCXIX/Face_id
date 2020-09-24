import os
import cv2
import numpy as np
import time
import copy
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torchvision import datasets
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision import utils
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split
from data_collector import DataCollector

class FaceRecognitionClassifier():
    def __init__(self):
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')            
        else:
            self.device = torch.device('cpu')
        print(f'Recognition module is running on {self.device}')

        with open('facedict.json') as json_file:
            self.facedict = json.load(json_file)
            
        self.labels_count = 3
        self.pretrained_model = InceptionResnetV1(pretrained='vggface2')
        self.model = None
        
    def get_model(self):
        model = ModifiedResnet(pretrained_model=self.pretrained_model, labels_count=len(self.facedict))
        model.load_state_dict(torch.load('mod_resnet.pth'))
        model.eval()
        model.to(self.device)
        self.model = model
        return model    
    
    def train_classifier(self, num_epochs=50):
        try:
            training_data = np.load('training_data.npy', allow_pickle=True)
        except:
            dc = DataCollector
            dc.collect_data()
            training_data = np.load('training_data.npy', allow_pickle=True)

        images = torch.Tensor([i[0] for i in training_data])
        labels = torch.Tensor([i[1] for i in training_data])

        dataset = TensorDataset(images, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': train_size, 'val': val_size}
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        modified_resnet = ModifiedResnet(pretrained_model=self.pretrained_model, labels_count=len(self.facedict))
        modified_resnet.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(modified_resnet.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()

        best_model_wts = copy.deepcopy(modified_resnet.state_dict())
        best_acc = 0.0
        best_loss = 1.0

        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch}/{num_epochs - 1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    modified_resnet.train()
                else:
                    modified_resnet.eval()

                running_loss = 0.0
                running_corrects = 0

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.long().to(self.device)
                    #print(labels)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = modified_resnet(inputs)
                        value, preds = torch.max(outputs, 1)  #
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    # print(preds)
                    # print(labels)
                    running_corrects += torch.sum(preds == labels)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(modified_resnet.state_dict())
                    torch.save(best_model_wts, 'mod_resnet.pth')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print(f'Best val Acc: {best_acc}')

        # load best model weights
        modified_resnet.load_state_dict(best_model_wts)
        return modified_resnet

    def recognize(self, face):
        predictions = self.model(face.unsqueeze(0).to(self.device))
        predictions = predictions.cpu().detach().numpy()[0]
        recognized_person = self.facedict[str(np.argmax(predictions))]

        return recognized_person, predictions[np.argmax(predictions)]

class Data(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = int(self.data[idx][1])
        return torch.tensor(image), torch.tensor(label)

class ModifiedResnet(nn.Module):
    def __init__(self, pretrained_model, labels_count):
        super(ModifiedResnet, self).__init__()
        self.pretrained_model = pretrained_model
        self.out_layer = nn.Linear(512, labels_count)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.out_layer(x)
        x = self.softmax(x)
        return x

frc = FaceRecognitionClassifier()

try:
    frc.get_model()
except:
    print('Untrained model weights')
    frc.train_classifier(10)

if __name__ == '__main__':
    frc.train_classifier(10)