import logging
import os
import random
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, val_single_volume,val_single_volume1
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts
from dataset.dataset import dataset,  RandomGenerator



def inference(args, model, best_performance):

    db_test = dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        # h, w = sampled_batch["image"].size()[2:]
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     case=case_name, device=args.device)
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)
    #metric_list = metric_list / len(testloader)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_datasets(args, model, datashot_path):

    logging.basicConfig(filename=datashot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr

    num_classes = args.num_classes

    batch_size = args.batch_size * args.n_gpu
    db_train = dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train",
                            transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of the training set: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss()

    dice_loss = DiceLoss(num_classes)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005, amsgrad=False)
    #optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001, betas=(0.9, 0.999), amsgrad=False)

    # lr_ = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    lr_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6, last_epoch=-1)
    writer = SummaryWriter(datashot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    # max_epoch = max_iterations // len(trainloader) + 1
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} The number of iterations in each round. {} Maximum number of iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=100)
    basename = os.path.basename(os.path.normpath(datashot_path))
    f_loss_all = open(f"./loss/loss_{basename}.csv", "w", encoding='utf-8')
    writer_1 = csv.writer(f_loss_all)

    for epoch_num in iterator:
        loss_ce_all = 0.0
        loss_dice_all = 0.0
        loss_all = 0.0
        num = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image']
            label_batch = sampled_batch['label']

            image_batch = image_batch.to(args.device)
            label_batch = label_batch.to(args.device)

            outputs = model(image_batch).to(args.device)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            loss_all = loss_all + loss
            loss_ce_all = loss_ce_all + loss_ce
            loss_dice_all = loss_dice_all + loss_dice

            num = num + 1
            iter_num = iter_num + 1
            logging.info('Number of iterations %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
        print("The learning rate of the %d epoch：%f" % (epoch_num, optimizer.param_groups[0]['lr']))
        lr_.step()

        writer_1.writerow([epoch_num, float(loss_all / num)])
        f_loss_all.flush()
        writer.add_scalar('info/total_loss', loss_all / num, epoch_num)

        # save_mode_path = os.path.join(datashot_path, 'last.pth')

        # torch.save(model.state_dict(), save_mode_path)


        performance = inference(args, model, best_performance)
        save_interval = 50


        if (best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(datashot_path, 'best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # if (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(datashot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(datashot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    f_loss_all.close()
    writer.close()
    logging.info(f"The best dice is:{best_performance}")
    return "---The training is over!---"