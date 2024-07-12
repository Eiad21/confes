import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, Evaluator
import numpy as np

# Download TissueMNIST dataset
class TissueMNISTDataset(medmnist.TissueMNIST):
    def __init__(self, split, size=None, transform=None, download=True, n=None):
        super().__init__(split=split, size=size, transform=transform, download=download)

        self.targets = np.array(self.labels.flatten(), dtype=int)
        
        if split == "train":
            self.balance_classes()

        if n is not None and n < len(self.imgs):
            self.imgs = self.imgs[:n]
            self.targets = self.targets[:n]

    def balance_classes(self):
        label_counts = np.bincount(self.targets)
        min_count = np.min(label_counts)
        
        indices_to_keep = []
        for label in np.unique(self.labels):
            label_indices = np.where(self.targets == label)[0]
            if len(label_indices) > min_count:
                label_indices = np.random.choice(label_indices, min_count, replace=False)
            indices_to_keep.extend(label_indices)
        
        self.imgs = self.imgs[indices_to_keep]
        self.targets = self.targets[indices_to_keep]

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_set = TissueMNISTDataset(split='train', transform=transform, download=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=128)

    for image, label, index in train_dataloader:
        print(label[0])
        break

    # train_set.balance_classes()
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    # # Iterate through the dataset
    for i, (image, label, _) in enumerate(train_set):
        # Get the label for the current sample
        label_counts[label] += 1

    print(label_counts)