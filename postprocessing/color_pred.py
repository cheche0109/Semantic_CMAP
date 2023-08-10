
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

from src import UNet

class DiceCoefficient_Pred(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # compute the Dice score, ignoring background
        # argmax针对每一个像素找到属于概率最大的类别，然后转成one hot编码
        # print('InClass: ', pred.argmax(dim=1))
        # print('InClass: ', pred)
        pred = F.one_hot(pred.argmax(1), self.num_classes).permute(0, 3, 1, 2).float()
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        # pred从1开始，因为chanel0是背景
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index)
        
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count# 累计的样本个数count得到dice的均值

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        torch.distributed.all_reduce(self.count)


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


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    combine = torch.tensor([])
        #img_path = img_list[i]
    for j in range(4):
      classes = 4  # exclude background
      weights_path = "./save_weights/best_model_"+ str(j) + ".pth"
          # create model
      img_path = "./Dataset_PDB_CV/CMAP/1QKS_A_origin.png"
      mask_path = "./Dataset_PDB_CV/GT/1QKS_A_gt.png"
      model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

          # load weights
      model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
      model.to(device)

          # load image
      original_img = Image.open(img_path).convert('RGB')
      mask = np.asarray(Image.open(mask_path))
      mask = torch.from_numpy(mask)
         # mask  = torch.Tensor(list(mask))
      mask = torch.unsqueeze(mask, dim=0)
      original_size = original_img.size[0]

          # from pil image to tensor and normalize
      data_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)])
      img = data_transform(original_img)
      
          #print('after trans b4 expand batch dim: ', img)
          #expand batch dimension
      img = torch.unsqueeze(img, dim=0)
        
    
      model.eval()  # 进入验证模式
      dice = DiceCoefficient_Pred(num_classes=classes+1, ignore_index=255)
      with torch.no_grad():
          # init model
          img_height, img_width = img.shape[-2:]
          init_img = torch.zeros((1, 3, img_height, img_width), device=device)
          model(init_img)
          t_start = time_synchronized()
          output = model(img.to(device))
          #dice.update(output, gt.to(device))
          #print('dice score: ', dice.value.item())

          t_end = time_synchronized()
          #print("inference time: {}".format(t_end - t_start))
          prediction = output['out'].to("cpu")
          #exit()
          pred = prediction.argmax(1).squeeze(0).numpy().astype(np.uint8)
          color = [[0,0,0],[255,0,0], [0,255,0], [0,0,255], [255,255,255]]
          color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype='uint8') 
          for i in range(0, 5):
              color_mask[np.where(pred==i)] = color[i]

          mask_color = Image.fromarray(color_mask)
          mask_color.save("1QKS_Model"+str(j)+".png")   
          dice.update(prediction, mask)
          combine = torch.cat((combine, prediction), 0)
          print(dice.value.item(), end = ',')
          final_pred = torch.mean(combine, 0, keepdim=True)
          final = final_pred.argmax(1).squeeze(0).numpy().astype(np.uint8)
          color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype='uint8')
          for i in range(0, 5):
              color_mask[np.where(final==i)] = color[i]

          mask_color = Image.fromarray(color_mask)
          mask_color.save("1QKS_Ensemble.png")

          dice = DiceCoefficient_Pred(num_classes=classes+1, ignore_index=255)
          dice.update(final_pred, mask)
          print(dice.value.item())
if __name__ == '__main__':
    main()
