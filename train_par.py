import os
import time
import datetime

import torch

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset_par import PDBDataset
import transforms as T
import early_stop as S

class SegmentationPresetTrain:
    def __init__(self, size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
	    T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    crop_size = 100

    if train:
        return SegmentationPresetTrain(size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(size, mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

#def create_model(num_classes):
    #pre_backbone = backbone.load_state_dict(torch.load("./mobilenet_v3_large.pth", map_location='cpu'))
    #model = MobileV3Unet(num_classes=num_classes, pretrain_backbone=True)
    #return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    partition_idx = args.partition_idx
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    size = args.size
    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    
    early_stopping = S.EarlyStopping(patience=60)
 
    #results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = args.OutputFile

    train_dataset = PDBDataset(args.data_path,
                                 train=True,
                                 partition_idx=partition_idx,
                                 transforms=get_transform(train=True, size=size,mean=mean, std=std))

    val_dataset = PDBDataset(args.data_path,
                               train=False,
                               partition_idx=partition_idx,
                               transforms=get_transform(train=False, size=size, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]


    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(
        #params_to_optimize,
        #lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    #)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

 
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr, dice_train  = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice, loss_eval = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.7f}")
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.7f}\n" \
			 f"eval_loss: {loss_eval:.7f}\n" \
                         f"lr: {lr:.7f}\n" \
                         f"dice_train: {dice_train:.7f}\n" \
                         f"dice coefficient: {dice:.7f}\n"

            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        
        early_stopping(dice)

        if early_stopping.save_model:
            torch.save(save_file, "/home/projects/vaccine/people/cheche/thesis/Model/U_Net/"+model_dir+"/save_weights/best_model_"+str(partition_idx)+".pth")

        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DATASET root")
    # exclude background
    parser.add_argument("--num-classes", default=4, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    #parser.add_argument("-e", "--patience-earlystopping", default=500, type=int)
    parser.add_argument("-p", "--partition-idx", default=0, type=int)
    parser.add_argument('-o', dest='OutputFile',type=str, required=True, help='Output file')
    parser.add_argument('-dir', action='store', dest='ModelDirectory', type=str, required=True, help='Directory: best.model')
    parser.add_argument("--epochs", default=500, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument("--size", default=200, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    
    model_dir = args.ModelDirectory
    if not os.path.exists("/home/projects/vaccine/people/cheche/thesis/Model/U_Net/"+ model_dir):
        os.mkdir("/home/projects/vaccine/people/cheche/thesis/Model/U_Net/"+ model_dir)
        os.mkdir("/home/projects/vaccine/people/cheche/thesis/Model/U_Net/"+ model_dir + "/save_weights")

    main(args)
