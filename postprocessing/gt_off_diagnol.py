
import torch.nn.functional as F
from train_utils.dice_coefficient_loss import multiclass_dice_coeff, build_target
import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from src import UNet

def data_path_list(partition_idx,root):
    partition_idx = [int(partition_idx)]
    partition_file = os.path.join(root, '3996.json')
    infile=open(partition_file,'r')
    partition_dict = json.load(infile)
    infile.close()
    name_list = list()
    for i in partition_idx:
        name_list.extend(partition_dict['Partition_'+str(i)])
    name_list = list(map(lambda x:'_'.join(x.split("_")[:2]), name_list))
    data_root = os.path.join(root, "Dataset_PDB_CV")
    img_names = [i for i in os.listdir(os.path.join(data_root, "CMAP")) if i.endswith(".png") and '_'.join(i.split("_")[:2]) in name_list]
    img_list = [os.path.join(data_root, "CMAP", i) for i in img_names]
    mask_list  = [os.path.join(data_root, "GT", '_'.join(i.split("_")[:2]) + "_gt.png")
                       for i in img_names]    
    return img_list, mask_list

def main():
    # get devices
    img_list, mask_list = data_path_list(4, './')
    #combine = torch.tensor([])
    for i in range(len(img_list)):
        pro_name = '_'.join(img_list[i].split('/')[-1].split('_')[:2])
        print(pro_name)
        mask_path = mask_list[i]


        mask = np.asarray(Image.open(mask_path))

        #print(final_pred[0].size)
        for i in range(mask[0].size):
            for j in range(mask[0].size):
                if i > j:
                    if mask[i][j] == 3:
                        pred = 'A'
                    elif mask[i][j] == 4:
                        pred = 'P'
                    else:
                        pred = '-'
                    with open('./GroundTruth_test_top/'+ pro_name + "_gt.txt", 'a') as f:
                        f.write(pro_name + ',' + str(i) + '|' + str(j) +  ',' + pred + '\n')
                if i < j:
                    if mask[i][j] == 3:
                        pred = 'A'
                    elif mask[i][j] == 4:
                        pred = 'P'
                    else:
                        pred = '-'
                    with open('./GroundTruth_test_bottom/'+ pro_name + "_gt.txt", 'a') as f:
                        f.write(pro_name + ',' + str(i) + '|' + str(j) +  ',' + pred + '\n')

if __name__ == '__main__':
    main()
