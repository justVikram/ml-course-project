import copy
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
import os

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = r'C:\Users\RISHABJAIN\OneDrive\Documents\dataset'

sets = ['train', 'val']

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}

datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_name = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch{epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_corrects.double() / datasets_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc{epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        time_elapsed = time.time() - since
        print(f'training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'best val acc : {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model


model = models.squeezenet1_0(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_channels
model.classifier[1] = nn.Conv2d(num_ftrs, 3, kernel_size=(1, 1), stride=(1, 1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7,
                                        gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)
