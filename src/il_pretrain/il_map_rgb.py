import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from il_pretrain.demo_dataset import HumanDemoDatasetRGBMAP
from models.ppo_nets import PPONetsMapRGB
from utils.color_print import *
from utils.common import imagenet_rgb_preprocess


def save_model(net_model, global_iter, global_step, model_dir, is_best):
    ckpt_file = os.path.join(model_dir,
                             'ckpt_{:08d}.pth'.format(global_iter))
    print_yellow('Saving checkpoint: %s' % ckpt_file)
    data_to_save = {
        'global_iter': global_iter,
        'global_step': global_step,
        'state_dict': net_model.state_dict(),
    }
    torch.save(data_to_save, ckpt_file)
    if is_best:
        shutil.copyfile(ckpt_file, os.path.join(model_dir, 'model_best.pth'))


def load_model(net_model, model_dir, device, step=None):
    if step is None:
        ckpt_file = os.path.join(model_dir, 'model_best.pth')
    else:
        ckpt_file = os.path.join(model_dir, 'ckpt_{:08d}.pth'.format(step))
    if not os.path.isfile(ckpt_file):
        raise ValueError("No checkpoint found at '{}'".format(ckpt_file))
    print_yellow('Loading checkpoint {}'.format(ckpt_file))
    checkpoint = torch.load(ckpt_file)
    model_dict = net_model.state_dict()
    requires_grad_ori = {}
    for name, param in net_model.named_parameters():
        requires_grad_ori[name] = param.requires_grad
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net_model.load_state_dict(model_dict)
    global_iter = checkpoint['global_iter']
    global_step = checkpoint['global_step']
    net_model.to(device)
    print('========= requires_grad =========')
    for name, param in net_model.named_parameters():
        param.requires_grad = requires_grad_ori[name]
        print(name, param.requires_grad)
    print('=================================')
    print_yellow('Checkpoint loaded...')
    print_yellow('          ' + ckpt_file)
    return global_iter, global_step


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--disable_small_map', action='store_true',
                        help='do not use small map')
    parser.add_argument('--demo_dir', type=str, default='../../human_demo',
                        help='directory of human demonstration data')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu for rendering')
    parser.add_argument('--traj_len', type=int, default=100,
                        help='truncated trajectory length')
    parser.add_argument('--lr', dest='lr', type=float, default=0.00025)
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.000,
                        help='weight_decay')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save model every n iterations')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='logging every n steps')
    parser.add_argument('--max_iters', type=int, default=1000000,
                        help='maximum number of episodes/iterations')
    parser.add_argument('--fix_cnn', action='store_true',
                        help='fix cnn(resnet18) weights')
    parser.add_argument('--rnn_seq_len', type=int, default=20,
                        help='sequence length in training rnn')
    parser.add_argument('--rnn_hidden_dim', type=int, default=128,
                        help='rnn hidden layer dimension')
    parser.add_argument('--rnn_type', help='lstm or gru',
                        default='gru', choices=['gru', 'lstm'], type=str)
    parser.add_argument('--rnn_num', help='number of rnn layers',
                        default=1, type=int)
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='maximum gradient norm')
    parser.add_argument('--test', action='store_true', help='test the trained model')
    parser.add_argument('--save_dir', type=str, default='./il_pretrain_rgb_map')
    parser.add_argument('--resume', '-rt', action='store_true', help='resume training')
    parser.add_argument('--resume_step', '-rs', type=int, default=None)
    return parser.parse_args()


def sample_human_demo(dataset, idx):
    return dataset[idx]


def main():
    args = parse_arguments()
    print_green('Program starts at: \033[92m %s '
                '\033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    args.device = None
    args.device = torch.device("cuda:0" if torch.cuda.is_available()
                                           and not args.disable_cuda
                               else "cpu")
    config = vars(args)
    train_mode = not args.test
    log_dir = os.path.join(args.save_dir, 'logs')
    model_dir = os.path.join(args.save_dir, 'model')
    if train_mode and not args.resume:
        dele = input("Do you wanna recreate ckpt and log folders? (y/n)")
        if dele == 'y':
            if os.path.exists(args.save_dir):
                shutil.rmtree(args.save_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    if train_mode:
        with open(os.path.join(args.save_dir, 'hyperparams.json'), 'w') as f:
            hps = {key: val for key, val in config.items() if key != 'device'}
            json.dump(hps, f, indent=2)
    best_val_acc = -np.inf
    net_model = PPONetsMapRGB(act_dim=6,
                              device=args.device,
                              fix_cnn=args.fix_cnn,
                              rnn_type=args.rnn_type,
                              rnn_hidden_dim=args.rnn_hidden_dim,
                              rnn_num=args.rnn_num,
                              use_rgb=True)
    net_model.to(args.device)
    if args.resume or args.test:
        global_iter, global_step = load_model(net_model,
                                              model_dir,
                                              args.device,
                                              step=args.resume_step)
    else:
        global_iter = 0
        global_step = 0
    optimizer = optim.Adam(net_model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_criterion = nn.CrossEntropyLoss().to(args.device)
    train_dataset = HumanDemoDatasetRGBMAP(root_dir=args.demo_dir,
                                           seq_len=args.traj_len, train=True)
    eval_dataset = HumanDemoDatasetRGBMAP(root_dir=args.demo_dir,
                                          seq_len=args.traj_len, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, drop_last=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=8, drop_last=False)
    for iter in tqdm(range(global_iter, args.max_iters), desc='iter'):
        global_iter = iter
        net_model.train()
        for i_batch, (rgbs, l_maps, s_maps, actions) in enumerate(tqdm(train_dataloader,
                                                                       desc='inner_batch')):
            global_step += 1
            ep_rgbs = rgbs.float().permute(1, 0, 2, 3, 4)
            ep_rgbs = imagenet_rgb_preprocess(ep_rgbs[:-1, :, ...])

            ep_l_maps = l_maps.float().permute(1, 0, 2, 3, 4)
            ep_l_maps = imagenet_rgb_preprocess(ep_l_maps[:-1, :, ...])

            ep_s_maps = s_maps.float().permute(1, 0, 2, 3, 4)
            ep_s_maps = imagenet_rgb_preprocess(ep_s_maps[:-1, :, ...])

            tgt_actions = actions.permute(1, 0).long()
            hidden_state = net_model.init_hidden(batch_size=ep_l_maps.shape[1])
            iter_loss = []
            for start in range(0, args.traj_len, args.rnn_seq_len):
                end = start + args.rnn_seq_len
                it_rgbs = ep_rgbs[start: end].to(args.device)
                it_l_maps = ep_l_maps[start: end].to(args.device)
                it_s_maps = ep_s_maps[start: end].to(args.device)
                it_actions = tgt_actions[start:end].to(args.device)
                _, _, _, _, hidden_state, pred_actions = net_model(it_l_maps,
                                                                   it_s_maps,
                                                                   it_rgbs,
                                                                   hidden_state)
                pred_actions = pred_actions.permute(0, 2, 1)
                loss = loss_criterion(pred_actions, it_actions)
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(),
                                                   args.max_grad_norm)
                optimizer.step()
                iter_loss.append(loss.item())
            iter_loss = np.mean(iter_loss)
            if global_step % args.log_interval == 0:
                writer.add_scalar('train/loss', iter_loss, global_step)
        if global_iter % args.save_interval == 0:
            val_loss = 0
            val_correct = 0
            val_total = 0
            net_model.eval()
            with torch.no_grad():
                for i_batch, (rgbs, l_maps, s_maps, actions) in enumerate(eval_dataloader):
                    ep_rgbs = rgbs.float().permute(1, 0, 2, 3, 4)
                    ep_rgbs = imagenet_rgb_preprocess(ep_rgbs[:-1, :, ...])

                    ep_l_maps = l_maps.float().permute(1, 0, 2, 3, 4)
                    ep_l_maps = imagenet_rgb_preprocess(ep_l_maps[:-1, :, ...])

                    ep_s_maps = s_maps.float().permute(1, 0, 2, 3, 4)
                    ep_s_maps = imagenet_rgb_preprocess(ep_s_maps[:-1, :, ...])

                    tgt_actions = actions.permute(1, 0).long()
                    hidden_state = net_model.init_hidden(batch_size=ep_l_maps.shape[1])
                    iter_loss = []
                    for start in range(0, args.traj_len, args.rnn_seq_len):
                        end = start + args.rnn_seq_len
                        it_rgbs = ep_rgbs[start: end].to(args.device)
                        it_l_maps = ep_l_maps[start: end].to(args.device)
                        it_s_maps = ep_s_maps[start: end].to(args.device)
                        it_actions = tgt_actions[start:end].to(args.device)
                        _, _, _, _, hidden_state, pred_actions = net_model(it_l_maps,
                                                                           it_s_maps,
                                                                           it_rgbs,
                                                                           hidden_state)
                        pred_actions = pred_actions.permute(0, 2, 1)
                        pred = pred_actions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        val_correct += pred.eq(it_actions.view_as(pred)).sum().item()
                        val_total += np.prod(list(it_actions.size()))
                        loss = loss_criterion(pred_actions, it_actions)
                        iter_loss.append(loss.item())
                    iter_loss = np.mean(iter_loss)
                    val_loss += iter_loss
            val_acc = val_correct / float(val_total)
            writer.add_scalar('test/loss', val_loss, global_step)
            writer.add_scalar('test/acc', val_acc, global_step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True
            else:
                is_best = False
            save_model(net_model=net_model, global_iter=global_iter,
                       global_step=global_step,
                       model_dir=model_dir, is_best=is_best)


if __name__ == '__main__':
    main()
