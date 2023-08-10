import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import json


class PDBDataset(Dataset):
    def __init__(self, root: str, train: bool, partition_idx: int, transforms=None):
        super(PDBDataset, self).__init__()
        #self.flag = "training" if train else "test"
        self.partition_file = os.path.join(root, '3996.json')
        if train:
            self.partition_idx = [i for i in range(4) if i != partition_idx]
        else:
            self.partition_idx = [partition_idx]
        name_list = list(map(lambda x:'_'.join(x.split("_")[:2]), self.read_partition()))
        
        data_root = os.path.join(root, "Dataset_PDB_CV")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "CMAP")) if i.endswith(".png") and '_'.join(i.split("_")[:2]) in name_list]
        self.img_list = [os.path.join(data_root, "CMAP", i) for i in img_names]
        self.manual = [os.path.join(data_root, "GT", '_'.join(i.split("_")[:2]) + "_gt.png")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx]).convert('L')

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask
 

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

    def read_partition(self):
        infile=open(self.partition_file,'r')
        partition_dict = json.load(infile)
        infile.close()
        name_list = list()
        for i in self.partition_idx:
            name_list.extend(partition_dict['Partition_'+str(i)])
        return name_list





def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



