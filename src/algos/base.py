import json
import os
import shutil

import numpy as np
import torch
from torch import optim

from configs.house3d_config import get_configs
from utils import color_print
from utils import logger


class BaseAgent:
    def __init__(self, env, args, val_env=None):
        self.config = vars(args)
        self.house_config = get_configs()
        self.env = env
        self.val_env = val_env
        self.total_envs_per_rollout = env.num_envs * args.train_rollout_repeat
        self.train_mode = not args.test
        self.num_steps = args.num_steps
        self.noptepochs = args.noptepochs
        self.batch_size = args.batch_size
        self.rnn_seq_len = args.rnn_seq_len
        self.device = args.device
        self.global_iter = 0
        self.best_avg_reward = -np.inf
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.max_iters = args.max_iters
        nenvs = env.num_envs
        self.nbatch = nenvs * args.num_steps

        self.log_dir = os.path.join(args.save_dir, 'logs')
        self.model_dir = os.path.join(args.save_dir, 'model')
        if self.train_mode and not args.resume:
            dele = input("Do you wanna recreate ckpt and log folders? (y/n)")
            if dele == 'y':
                if os.path.exists(args.save_dir):
                    shutil.rmtree(args.save_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
        if not args.test:
            logger.configure(dir=self.log_dir,
                             format_strs=['tensorboard', 'csv'])

        self.net_model = None
        self.optimizer = None
        if self.train_mode:
            if not args.resume:
                with open(os.path.join(args.save_dir,
                                       'hyperparams.json'), 'w') as f:
                    hps = {key: val for key, val in self.config.items()
                           if key != 'device'}
                    json.dump(hps, f, indent=2)
            else:
                with open(os.path.join(args.save_dir,
                                       'hyperparams_%d.json' % self.global_iter),
                          'w') as f:
                    hps = {key: val for key, val in self.config.items()
                           if key != 'device'}
                    json.dump(hps, f, indent=2)

    def save_model(self, is_best, step=None):
        if step is None:
            step = self.global_iter
        ckpt_file = os.path.join(self.model_dir,
                                 'ckpt_{:08d}.pth'.format(step))
        color_print.print_yellow('Saving checkpoint: %s' % ckpt_file)
        data_to_save = {
            'ckpt_step': step,
            'global_iter': self.global_iter,
            'state_dict': self.net_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(data_to_save, ckpt_file)
        if is_best:
            shutil.copyfile(ckpt_file, os.path.join(self.model_dir,
                                                    'model_best.pth'))

    def load_model(self, step=None):
        if self.config['il_pretrain'] and not self.config['test'] and not self.config['resume']:
            ckpt_file = os.path.join(self.config['pretrain_dir'], 'model_best.pth')
        else:
            if step is None:
                ckpt_file = os.path.join(self.model_dir, 'model_best.pth')
            else:
                ckpt_file = os.path.join(self.model_dir, 'ckpt_{:08d}.pth'.format(step))
        if not os.path.isfile(ckpt_file):
            raise ValueError("No checkpoint found at '{}'".format(ckpt_file))
        color_print.print_yellow('Loading checkpoint {}'.format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        model_dict = self.net_model.state_dict()
        requires_grad_ori = {}
        for name, param in self.net_model.named_parameters():
            requires_grad_ori[name] = param.requires_grad
        pretrained_dict = {}
        unused_params = []
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict:
                pretrained_dict[k] = v
            elif k.split('.')[0] in ['small_large_map_merge_fc', 'rgb_map_merge_fc']:
                new_k = k.replace(k.split('.')[0], 'merge_fc')
                pretrained_dict[new_k] = v
            else:
                unused_params.append(k)

        model_dict.update(pretrained_dict)
        self.net_model.load_state_dict(model_dict)
        if self.config['il_pretrain'] and not self.config['resume']:
            self.global_iter = 0
        else:
            self.global_iter = checkpoint['global_iter']
        self.net_model.to(self.device)
        self.optimizer = optim.Adam(self.net_model.parameters(),
                                    lr=self.config['lr'],
                                    weight_decay=self.config['weight_decay'])
        print('========= requires_grad =========')
        for name, param in self.net_model.named_parameters():
            param.requires_grad = requires_grad_ori[name]
            print(name, param.requires_grad)
        print('=================================')
        color_print.print_yellow('Checkpoint loaded...')
        color_print.print_yellow('          ' + ckpt_file)
        if len(unused_params) > 0:
            print('==================================')
            print('Unused parameter from loaded model:')
            print(unused_params)
            print('==================================')
