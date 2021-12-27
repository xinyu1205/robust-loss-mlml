import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, COCO_missing_dataset, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss, Hill, SPLC
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--dataset', help='select dataset', default='./dataset/coco_train_0.4left.txt')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='resnet50',choices=['resnet50','resnet101'])
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.5, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--loss', default='Hill', type=str,
                    help='select loss function', choices=['BCE','Focal','Asl','Hill','SPLC'])


def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    print('done\n')

    if torch.cuda.device_count() > 1:
        device_id = range(torch.cuda.device_count())      
        model = nn.DataParallel(model, device_ids = device_id)

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    instances_path_train = args.dataset
    
    data_path_val   = f'{args.data}/val2014'    # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = COCO_missing_dataset(data_path_train,
                                instances_path_train,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    CutoutPIL(cutout_factor=0.5),
                                    RandAugment(),
                                    transforms.ToTensor(),
                                    # normalize,
                                ]),class_num=args.num_classes)
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(args, model, train_loader, val_loader, args.lr)


def train_multi_label_coco(args, model, train_loader, val_loader, lr):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 30
    weight_decay = 1e-4
    if args.loss == 'BCE':
        criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)
    elif args.loss == 'Focal':
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)
    elif args.loss == 'Asl':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    elif args.loss == 'Hill':
        criterion = Hill()
    elif args.loss == 'SPLC':
        criterion = SPLC()
    else:
        raise ValueError("Loss function dose not exist.")
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target ) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            # target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            if args.loss == 'SPLC' :
                loss = criterion(output, target, epoch)
            else:
                loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()
        mAP_score, if_ema_better = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            best_epoch = epoch
            try:
                if if_ema_better:
                    torch.save(ema.module.state_dict(), os.path.join(
                            'models/', 'model-highest.ckpt'))
                else:
                    torch.save(model.state_dict(), os.path.join(
                            'models/', 'model-highest.ckpt'))
            except:
                print('store failed')
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}, best_epoch={}\n'.format(mAP_score, highest_mAP, best_epoch))


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    mAP_max = max(mAP_score_regular, mAP_score_ema)
    if mAP_score_ema >= mAP_score_regular:
        if_ema_better = True
    else:
        if_ema_better = False

    return mAP_max, if_ema_better


if __name__ == '__main__':
    main()
