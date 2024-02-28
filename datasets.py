from torchvision import datasets,transforms
from torch.utils.data import Dataset
DATASETS = ['mnist_2cl','mnist','cifar10']
def get_dataset(dataset_name,split):
    split_b = split == 'train'
    if dataset_name == 'mnist_2cl':
        data_set  = datasets.MNIST('./data',download=True, train= split_b,transform=transforms.Compose([transforms.ToTensor()
        ]))
        data_set.targets[data_set.targets<=4] = 0
        data_set.targets[data_set.targets>4] = 1
    elif dataset_name == 'mnist':
         data_set  = datasets.MNIST('./data',download=True, train= split_b,transform=transforms.Compose([transforms.ToTensor()
        ]))
    elif dataset_name == 'cifar10':
         data_set  = datasets.CIFAR10('./data',download=True, train= split_b,transform=transforms.Compose([transforms.ToTensor()
        ]))

   
    return data_set

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

    
