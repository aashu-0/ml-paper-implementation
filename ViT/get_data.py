from torchvision import transforms
from torch.utils.data import DataLoader


BATCH_SIZE = 64
NUM_WORKERS = 4

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


train_data = torchvision.datasets.CIFAR10(root='./data',
                                          train=True,
                                          download=True,
                                          transform=train_transform,)
test_data = torchvision.datasets.CIFAR10(root='./data',
                                         train = False,
                                         download = True,
                                         transform = test_transform)

                                            
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

