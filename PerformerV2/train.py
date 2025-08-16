import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


from trainer import trainer_datasets
from lib.model import PerformerV2
from lib.PerformerV2 import performerV2



parser = argparse.ArgumentParser()

parser.add_argument('--train_root_path', type=str,
                    default='/home/xzp/dataset/Synapse/train', help='train root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='/home/xzp/dataset/Synapse/test', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/home/xzp/dataset/Synapse/list', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='train_log channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:2', help='gpu device')

args = parser.parse_args()
# config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.is_pretrain = False
    args.exp = 'PerformerV2_' + dataset_name + str(args.img_size)
    datashot_path = "checkpoint/{}".format(args.exp)
    datashot_path = datashot_path + '_pretrain' if args.is_pretrain else datashot_path
    datashot_path = datashot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else datashot_path
    datashot_path = datashot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else datashot_path
    datashot_path = datashot_path+'_bs'+str(args.batch_size)
    datashot_path = datashot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else datashot_path
    datashot_path = datashot_path + '_'+str(args.img_size)
    datashot_path = datashot_path + '_s'+str(args.seed) if args.seed != 1234 else datashot_path

    if not os.path.exists(datashot_path):
        os.makedirs(datashot_path)

    net = performerV2(num_classes=args.num_classes).to(args.device)

    trainer = {'Synapse': trainer_datasets}
    #
    trainer[dataset_name](args, net, datashot_path)