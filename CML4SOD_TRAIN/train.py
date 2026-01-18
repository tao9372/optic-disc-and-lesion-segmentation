import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.OD_seg_network import CML4SOD
from engine import *
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)


    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path_train_img,config.data_path_train_msk,config.data_path_train_fovea, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.train_batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)

    val_dataset = NPY_datasets(config.data_path_val_img,config.data_path_val_msk,config.data_path_val_fovea, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=config.val_batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=False)

    test_dataset = NPY_datasets(config.data_path_test_img, config.data_path_test_msk, config.data_path_test_fovea,
                               config, train=False)
    test_loader = DataLoader(test_dataset,
                            batch_size=config.test_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=False)




    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'CML4SOD':
        model = CML4SOD(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )


    else: raise Exception('network in not right!')
    model = model.cuda()




    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)



    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    val_min_loss = 999




    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')

    val_max_iou_mean=0
    val_max_dice_mean = 0
    test_max_iou_mean = 0
    test_max_dice_mean = 0

    iou_dice_txt_save_path='iou-dice-results.txt'


    for epoch in range(start_epoch, config.epochs + 1):


        img_save_path_test = 'test_result/'+str(epoch)+'/'
        if not os.path.exists(img_save_path_test):
            os.makedirs(img_save_path_test)

        img_save_path_msk_test = 'test_result_msk/'+str(epoch)+'/'
        if not os.path.exists(img_save_path_msk_test):
            os.makedirs(img_save_path_msk_test)

        img_save_path_val = 'val_result/'+str(epoch)+'/'
        if not os.path.exists(img_save_path_val):
            os.makedirs(img_save_path_val)

        img_save_path_msk_val = 'val_result_msk/'+str(epoch)+'/'
        if not os.path.exists(img_save_path_msk_val):
            os.makedirs(img_save_path_msk_val)



        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        val_loss,val_iou_mean,val_dice_mean= val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                n_classes=1,
                img_save_path=img_save_path_val,
                img_save_path_msk=img_save_path_msk_val
            )

        loss, test_iou_mean, test_dice_mean = test_one_epoch(
            test_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            n_classes=1,
            img_save_path=img_save_path_test,
            img_save_path_msk=img_save_path_msk_test
        )

        if val_iou_mean >= val_max_iou_mean:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_iou.pth'))
            val_max_iou_mean = val_iou_mean

        if val_dice_mean >= val_max_dice_mean:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_dice.pth'))
            val_max_dice_mean = val_dice_mean

        if val_loss <= val_min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_loss.pth'))
            val_min_loss = val_loss

        if test_iou_mean >= test_max_iou_mean:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_test_iou.pth'))
            test_max_iou_mean = test_iou_mean

        if test_dice_mean >= test_max_dice_mean:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_test_dice.pth'))
            test_max_dice_mean = test_dice_mean

        if loss <= min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_test_loss.pth'))
            min_loss = loss
            min_epoch = epoch


        torch.save(
            {
                'epoch': epoch,
                'test_min_loss': min_loss,
                'test_min_loss_epoch': min_epoch,
                'test_loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))


        file_iou_dice_txt = open(iou_dice_txt_save_path, 'a')
        file_iou_dice_txt.write(
            str(epoch) + '  iou(val): ' + str(val_iou_mean)  + '  dice(val): ' + str(
                val_dice_mean) +  '  iou(test): ' + str(test_iou_mean)  +  '  dice(test): ' + str(
                test_dice_mean)  + '  loss(val): ' + str(val_loss) + '  loss(test): ' + str(loss))
        file_iou_dice_txt.write('\n')
        file_iou_dice_txt.close()




if __name__ == '__main__':
    config = setting_config
    main(config)