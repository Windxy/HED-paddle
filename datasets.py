import os
import cv2
import numpy as np
from PIL import Image
import paddle
from paddle.io import Dataset

class HED_Dataset(Dataset):
    def __init__(self, root='data/HED-BSDS', split='train', transform=False):
        super(HED_Dataset, self).__init__()
        self.root = root
        self.split = split
        if transform is not False:
            self.transform = transform
        if self.split == 'train':
            self.filelist = os.path.join(self.root, 'train_pair.lst')
        elif self.split == 'test':
            self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __getitem__(self, index):
        if self.split == "train":
            img_file, label_file = self.filelist[index].split()
            label = np.array(Image.open(os.path.join(self.root, label_file)), dtype=np.float32)

            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            assert label.ndim == 2

            label = label[np.newaxis, :, :]
            label[label < 128] = 0.0
            label[label >= 128] = 1.0

            img = np.array(cv2.imread(os.path.join(self.root, img_file)), dtype=np.float32)
            img = self.transform(img)
            return img, label
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(cv2.imread(os.path.join(self.root, img_file)), dtype=np.float32)
            img = self.transform(img)
            return img

    def __len__(self):
        return len(self.filelist)

if __name__ == '__main__':
    HED_Dataset = HED_Dataset()

    train_loader = paddle.io.DataLoader(HED_Dataset,batch_size=4,shuffle=True)
    for batch_id,data in train_loader:
        x = data[0]
        y = data[1]
        print(x.shape,y.shape)
