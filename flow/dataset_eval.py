import os
import glob
from PIL import Image
from torchvision.datasets import ImageFolder, LSUNClass
from torchvision.transforms import v2
from torch.utils.data import Dataset

root = "/extra/ucibdl0/shared/data"

class NoLabelDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = glob.glob(os.path.join(root, "*.png"))
        self.image_paths += glob.glob(os.path.join(root, "*.jpeg"))
        self.image_paths += glob.glob(os.path.join(root, "*.JPG"))
        self.image_paths += glob.glob(os.path.join(root, "*.jpg"))
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, 0
    
    def __len__(self):
        return len(self.image_paths)
    
class NoLabelLSUNClass(LSUNClass):
    def __getitem__(self, index):
        img, target = super(NoLabelLSUNClass, self).__getitem__(index)
        return img, 0

def get_dataset(args):
    transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToTensor(),
    ])
    if args.dataset == "celebahq":
        if args.validation:
            dataset = ImageFolder(root=os.path.join(root, "celeba_hq/val"), transform=transforms)
        else:
            dataset = ImageFolder(root=os.path.join(root, "celeba_hq/train"), transform=transforms)
    elif args.dataset  == "afhq":
        if args.validation:
            if args.label in ["dog", "cat", "wild"]:
                dataset = NoLabelDataset(root=os.path.join(root, "afhq/val", args.label), transform=transforms)
            else:
                dataset = ImageFolder(root=os.path.join(root, "afhq/val"), transform=transforms)
        else:
            if args.label in ["dog", "cat", "wild"]:
                dataset = NoLabelDataset(root=os.path.join(root, "afhq/train", args.label), transform=transforms)
            else:
                dataset = ImageFolder(root=os.path.join(root, "afhq/train"), transform=transforms)
    elif args.dataset == "lsunbedroom":
        if args.validation:
            dataset = NoLabelLSUNClass(root=os.path.join(root, "LSUN/bedroom_val_lmdb"), transform=transforms)
        else:
            raise NotImplementedError
    elif args.dataset == "lsunchurch":
        if args.validation:
            dataset = NoLabelLSUNClass(root=os.path.join(root, "LSUN/church_outdoor_val_lmdb"), transform=transforms)
        else:
            raise NotImplementedError
    else:

        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    return dataset