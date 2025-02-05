import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split

BATCH_SIZE = 64
NUM_WORKERS = 2

classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3}
img_per_class = 1000

# transforms
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])


# load data
dataset = torchvision.datasets.CIFAR10(root='./data',
                                          train=True,
                                          download=True,
                                          transform=test_transform)

# class indices
class_indices = {k:[] for k in classes.values()}

for idx, (img, label) in enumerate(dataset):
    if label in class_indices and len(class_indices[label]) < img_per_class:
        class_indices[label].append(idx)

    if all(len(num_indices)== img_per_class for num_indices in class_indices.values()):
        break

# create subset
total_indices = sum(class_indices.values(),[])
subset_dataset = Subset(dataset, total_indices)

# train and test split
train_size = int(0.8 * len(subset_dataset))
test_size = len(subset_dataset) - train_size
train_data, test_data = random_split(subset_dataset, [train_size, test_size])

# apply transform to test set separately
train_data.dataset.transform = train_transform


train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
