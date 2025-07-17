from torchvision import transforms
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale = (0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def make_datapath_list(phase='train'):
    rootpath = './data/hymenoptera_data'
    target_path = os.path.join(rootpath, phase, '**', '*.jpg')
    print(target_path)
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list


class HymeopteraDataset(Dataset):
    def __init__(self, file_list, transform, phase='train'):
        self.file_list = file_list  
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img, self.phase)    
        
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]
            
        
        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1
            
        return img, label