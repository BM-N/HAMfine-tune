import os
# import sys

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# _CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# _PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, '..'))

# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)

from models.transforms import get_transforms

class HAM10kDS(Dataset):
    def __init__(self, img_dir1, img_dir2, csv_file, transform=None, target_transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.folder_paths=[img_dir1, img_dir2]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df['image_id'].iloc[idx]
        for folder_path in self.folder_paths:
            image_path = os.path.join(folder_path, image_name) + '.jpg'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                break
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Could not find image {image_path}")
        label = self.df['label'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_dataloader(img_dir1, img_dir2, csv_file, transform, bs=32, shuffle=False, *args, **kwargs):
    ds = HAM10kDS(img_dir1, img_dir2, csv_file, transform)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, **kwargs)

# passar isso para o train.py

# img_dir1 = "HAM10000_images_part_1"
# img_dir2 = "HAM10000_images_part_2"
# train_file = "train_set.csv"
# val_file = "val_set.csv"
# test_file = "test_set.csv"
# train_transform = get_transforms()
# val_transform = get_transforms(train=False)

# train_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=train_file, transform=train_transform, shuffle=True)
# val_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=test_file, transform=val_transform)
# test_dls = get_dataloader(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=test_file, transform=val_transform)


# train_dataset = HAM10kDS(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=train_file, transform=train_transform)
# val_dataset = HAM10kDS(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=val_file, transform=val_transform)
# test_dataset = HAM10kDS(img_dir1=img_dir1, img_dir2=img_dir2, csv_file=test_file, transform=val_transform)
#  
# train_dls = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# val_dls = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
# test_dls = DataLoader(dataset=test_dataset,batch_size=32, shuffle=False)