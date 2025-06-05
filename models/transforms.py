import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Important: crop then resize
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),                     # New: Vertical Flip
#     transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10), # More aggressive affine
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),      # More aggressive ColorJitter
#     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # New: Gaussian Blur
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

def get_transforms(train: bool=True):
    if not train:
        transform = val_transform
    else:
        transform = train_transform
    return transform

# transform = get_transforms()
# print(transform)