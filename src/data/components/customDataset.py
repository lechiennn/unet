from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torch import squeeze

class CustomDataset(Dataset):
    def __init__(self, imageDir='data/train_hq', maskDir='data/train_masks', transform=None) -> None:
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.transform = transform

        self.listName = [os.path.splitext(file)[0] for file in os.listdir(imageDir)]

    def __len__(self):
        return len(self.listName)
    
    def __getitem__(self, idx):
        
        name = self.listName[idx]

        image = Image.open(os.path.join(self.imageDir, name) + '.jpg')
        mask = Image.open(os.path.join(self.maskDir, name) + '_mask.gif').convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = squeeze(mask)

        sample = {'image': image, 'mask': mask}
        return sample
        

def main():
    dataset = CustomDataset()
    sample = dataset[0]
    # sample['image'].show()
    # sample['mask'].show()
    print(len(dataset)) # 5088

if __name__ == '__main__':
    main()