import os
import numpy as np
import config
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class myImageDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.class_names = os.listdir(root_path)
        self.data = []
        
        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_path, name))
            self.data += list(zip(files, [index]*len(files)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_path, self.class_names[label])

        image = Image.open(os.path.join(root_and_dir, img_file))
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]) 
        image = transform(image)

        return image, label
    
def data_loader(dataset, batch_size, shuffle):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=config.NUM_WORKERS)
    return loader
