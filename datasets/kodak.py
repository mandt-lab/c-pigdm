import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class KodakDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = T.Compose(
            [
                T.CenterCrop(512),
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )
        self.images = []

        for img in os.listdir(self.root):
            if img.endswith(".png") or img.endswith(".jpg"):
                self.images.append(os.path.join(self.root, img))

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transform(img)
        return img, 0, {"index": index}

    def __len__(self):
        return len(self.images)


def get_kodak_dataset(root, split="val", transform="default", subset=-1, **kwargs):
    dset = KodakDataset(root)
    return dset
