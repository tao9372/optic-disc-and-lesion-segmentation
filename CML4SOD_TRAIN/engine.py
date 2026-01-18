import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from IoU import iou
import cv2
import matplotlib.pyplot as plt


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets,img_name,msk_name,fovea_cor = data

        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        fovea_cor = fovea_cor / 255
        fovea_cor = fovea_cor.cuda(non_blocking=True).float()

        out = model(images,fovea_cor)
        loss = criterion(out, targets)


        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(val_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    img_save_path,img_save_path_msk):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    iou_mean=0
    dice_mean=0

    all_img_num=0

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk,img_name,msk_name,fovea_cor = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            fovea_cor = fovea_cor / 255
            fovea_cor = fovea_cor.cuda(non_blocking=True).float()

            out = model(img, fovea_cor)

            loss = criterion(out, msk)


            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            msk_pred = np.where(out > 0.5, 1, 0)
            msk = msk.squeeze(1).cpu().detach().numpy()
            msk = np.where(msk > 0.5, 1, 0)

            msk_pred_3dim_iou = torch.tensor(msk_pred)
            msk_3dim_iou = torch.tensor(msk)


            iou_in_batch, dice_in_bath = iou(msk_pred_3dim_iou, msk_3dim_iou)


            iou_mean = iou_mean + iou_in_batch.sum().item()
            dice_mean = dice_mean + dice_in_bath.sum().item()
            all_img_num = all_img_num + img.size(0)


            if epoch % 5 == 0:

                for img_num in range(img.size(0)):
                    img2 = img[img_num].permute(1, 2, 0).detach().cpu().numpy()
                    img2 = img2 / 255. if img2.max() > 1.1 else img2

                    plt.figure(figsize=(7, 15))
                    plt.subplot(3, 1, 1)
                    plt.imshow(img2)
                    plt.axis('off')

                    plt.subplot(3, 1, 2)
                    plt.imshow(msk[img_num], cmap='gray')
                    plt.axis('off')

                    plt.subplot(3, 1, 3)
                    plt.imshow(msk_pred[img_num], cmap='gray')
                    plt.axis('off')

                    plt.savefig(img_save_path + img_name[img_num])
                    plt.close()

                    pred_msk_array = (msk_pred[img_num] * 255).astype(np.uint8)

                    cv2.imwrite(img_save_path_msk + img_name[img_num], pred_msk_array)


    iou_mean = iou_mean / all_img_num
    dice_mean = dice_mean / all_img_num


    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
    print(log_info)
    logger.info(log_info)
    
    # return np.mean(loss_list)
    return np.mean(loss_list),iou_mean,dice_mean


def test_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  img_save_path,img_save_path_msk):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []


    iou_mean = 0
    dice_mean = 0

    all_img_num = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk, img_name, msk_name, fovea_cor = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            fovea_cor = fovea_cor / 255
            fovea_cor = fovea_cor.cuda(non_blocking=True).float()

            out = model(img, fovea_cor)

            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            msk_pred = np.where(out > 0.5, 1, 0)
            msk = msk.squeeze(1).cpu().detach().numpy()
            msk = np.where(msk > 0.5, 1, 0)

            msk_pred_3dim_iou = torch.tensor(msk_pred)
            msk_3dim_iou = torch.tensor(msk)

            iou_in_batch, dice_in_bath = iou(msk_pred_3dim_iou, msk_3dim_iou)


            iou_mean = iou_mean + iou_in_batch.sum().item()
            dice_mean = dice_mean + dice_in_bath.sum().item()
            all_img_num = all_img_num + img.size(0)

            if epoch % 5 == 0:

                for img_num in range(img.size(0)):
                    img2 = img[img_num].permute(1, 2, 0).detach().cpu().numpy()
                    img2 = img2 / 255. if img2.max() > 1.1 else img2

                    plt.figure(figsize=(7, 15))
                    plt.subplot(3, 1, 1)
                    plt.imshow(img2)
                    plt.axis('off')

                    plt.subplot(3, 1, 2)
                    plt.imshow(msk[img_num], cmap='gray')
                    plt.axis('off')

                    plt.subplot(3, 1, 3)
                    plt.imshow(msk_pred[img_num], cmap='gray')
                    plt.axis('off')

                    plt.savefig(img_save_path + img_name[img_num])
                    plt.close()

                    pred_msk_array = (msk_pred[img_num] * 255).astype(np.uint8)

                    cv2.imwrite(img_save_path_msk + img_name[img_num], pred_msk_array)


    iou_mean = iou_mean / all_img_num
    dice_mean = dice_mean / all_img_num

    log_info = f'test epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list),iou_mean,dice_mean

