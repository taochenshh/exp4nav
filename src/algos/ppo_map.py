import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch import optim

from algos.base import BaseAgent
from models.ppo_nets import PPONetsMapRGB
from utils import logger
from utils.color_print import *
from utils.common import imagenet_rgb_preprocess, safemean


class PPOAgentMap(BaseAgent):
    def __init__(self, env, args, val_env=None):
        super().__init__(env, args, val_env)
        assert self.num_steps % self.rnn_seq_len == 0
        self.net_model = PPONetsMapRGB(act_dim=env.action_space[0],
                                       device=args.device,
                                       fix_cnn=args.fix_cnn,
                                       rnn_type=args.rnn_type,
                                       rnn_hidden_dim=args.rnn_hidden_dim,
                                       rnn_num=args.rnn_num,
                                       use_rgb=args.use_rgb_with_map)

        self.net_model.to(self.device)

        self.optimizer = optim.Adam(self.net_model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
        self.val_loss_criterion = nn.SmoothL1Loss().to(args.device)
        if args.resume or args.test or args.il_pretrain:
            self.load_model(step=args.resume_step)

    def train(self):
        epinfobuf = deque(maxlen=10)
        t_trainstart = time.time()
        for iter in range(self.global_iter, self.max_iters):
            self.global_iter = iter
            t_iterstart = time.time()
            if iter % self.config['save_interval'] == 0 and logger.get_dir():
                with torch.no_grad():
                    res = self.rollout(val=True)
                    obs, actions, returns, values, advs, log_probs, epinfos = res
                avg_reward = safemean([epinfo['reward'] for epinfo in epinfos])
                avg_area = safemean([epinfo['seen_area'] for epinfo in epinfos])
                logger.logkv("iter", iter)
                logger.logkv("test/total_timesteps", iter * self.nbatch)
                logger.logkv('test/avg_area', avg_area)
                logger.logkv('test/avg_reward', avg_reward)
                logger.dumpkvs()
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    is_best = True
                else:
                    is_best = False
                self.save_model(is_best=is_best, step=iter)
            with torch.no_grad():
                obs, actions, returns, values, advs, log_probs, epinfos = self.n_rollout(
                    repeat_num=self.config['train_rollout_repeat'])
            if epinfos:
                epinfobuf.extend(epinfos)
            lossvals = {'policy_loss': [],
                        'value_loss': [],
                        'policy_entropy': [],
                        'approxkl': [],
                        'clipfrac': []}
            opt_start_t = time.time()
            noptepochs = self.noptepochs
            for _ in range(noptepochs):
                num_batches = int(np.ceil(actions.shape[1] / self.config['batch_size']))
                for x in range(num_batches):
                    b_start = x * self.config['batch_size']
                    b_end = min(b_start + self.config['batch_size'], actions.shape[1])
                    if self.config['use_rgb_with_map']:
                        rgbs, large_maps, small_maps = obs
                        b_rgbs, b_large_maps, b_small_maps = map(lambda p: p[:, b_start:b_end],
                                                                 (rgbs, large_maps, small_maps))
                    else:
                        large_maps, small_maps = obs
                        b_large_maps, b_small_maps = map(lambda p: p[:, b_start:b_end],
                                                         (large_maps, small_maps))
                    b_actions, b_returns, b_advs, b_log_probs = map(lambda p: p[:, b_start:b_end],
                                                                    (actions, returns, advs, log_probs))
                    hidden_state = self.net_model.init_hidden(batch_size=b_end - b_start)
                    for start in range(0, actions.shape[0], self.rnn_seq_len):
                        end = start + self.rnn_seq_len
                        slices = (arr[start: end] for arr in
                                  (b_large_maps, b_small_maps, b_actions,
                                   b_returns, b_advs, b_log_probs))

                        if self.config['use_rgb_with_map']:
                            info, hidden_state = self.net_train(*slices,
                                                                hidden_state=hidden_state,
                                                                rgbs=b_rgbs[start: end])
                        else:
                            info, hidden_state = self.net_train(*slices,
                                                                hidden_state=hidden_state)
                        lossvals['policy_loss'].append(info['pg_loss'])
                        lossvals['value_loss'].append(info['vf_loss'])
                        lossvals['policy_entropy'].append(info['entropy'])
                        lossvals['approxkl'].append(info['approxkl'])
                        lossvals['clipfrac'].append(info['clipfrac'])
            tnow = time.time()
            int_t_per_epo = (tnow - opt_start_t) / float(self.noptepochs)
            print_cyan('Net training time per epoch: {0:.4f}s'.format(int_t_per_epo))
            fps = int(self.nbatch / (tnow - t_iterstart))
            if iter % self.config['log_interval'] == 0:
                logger.logkv("Learning rate", self.optimizer.param_groups[0]['lr'])
                logger.logkv("per_env_timesteps", iter * self.num_steps)
                logger.logkv("iter", iter)
                logger.logkv("total_timesteps", iter * self.nbatch)
                logger.logkv("fps", fps)
                logger.logkv('ep_rew_mean', safemean([epinfo['reward']
                                                      for epinfo in epinfobuf]))
                logger.logkv('ep_area_mean', safemean([epinfo['seen_area']
                                                       for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - t_trainstart)
                for name, value in lossvals.items():
                    logger.logkv(name, np.mean(value))
                logger.dumpkvs()

    def test(self, render, val_id=0):
        self.net_model.eval()
        if self.env.num_envs != 1:
            raise ValueError('please use 1 env for testing and visualization')

        cum_rewards = []
        cum_seen_area = []
        cum_actions = []
        obs_uint8 = self.env.reset([self.house_config['fixed_test_set'][val_id]])
        if self.config['use_rgb_with_map']:
            print('RGB as input')
        hidden_state = self.net_model.init_hidden(batch_size=self.env.num_envs)
        obs_order = [1, 2, 0]
        for idx in range(self.num_steps):
            if render:
                self.env.render()
            if self.config['use_rgb_with_map']:
                obs = list(map(lambda p, device=self.device:
                               imagenet_rgb_preprocess(p, device), obs_uint8))
                obs = [obs[i] for i in obs_order]
            else:
                obs = list(map(lambda p, device=self.device:
                               imagenet_rgb_preprocess(p, device), obs_uint8[1:]))
            res = self.net_model(*obs, hidden_state=hidden_state, deterministic=False)
            actions, log_probs, entropy, vals, hidden_state, act_logits = res

            cum_actions.append(actions.cpu().data.numpy()[0])
            obs_uint8, rewards, dones, infos = self.env.step(actions.cpu().data.numpy().flatten())
            cum_rewards.append(infos[0]['reward_so_far'])
            cum_seen_area.append(infos[0]['seen_area'])
            if idx == self.num_steps - 1:
                print('Seen area:', infos[0]['seen_area'])
                print('Total reward:', infos[0]['reward_so_far'])
                print('Collisions:', infos[0]['collisions'])
                print('Start pose:')
                start_pose = infos[0]['start_pose']
                print('x: {0:.3f}  z:{1:.3f}  yaw:{2:.3f}'.format(start_pose[0],
                                                                  start_pose[1],
                                                                  start_pose[2]))
        if render:
            while True:
                self.env.render()
        return cum_seen_area, cum_rewards, cum_actions

    def net_train(self, large_maps, small_maps, actions, returns, advs,
                  old_log_probs, hidden_state, rgbs=None):
        self.net_model.train()
        actions = torch.from_numpy(actions).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        advs = torch.from_numpy(advs).float().to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        large_maps = imagenet_rgb_preprocess(large_maps, device=self.device)
        small_maps = imagenet_rgb_preprocess(small_maps, device=self.device)

        if self.config['use_rgb_with_map']:
            rgbs = imagenet_rgb_preprocess(rgbs, device=self.device)
            res = self.net_model(large_maps=large_maps,
                                 small_maps=small_maps,
                                 rgb_ims=rgbs,
                                 hidden_state=hidden_state,
                                 action=actions)
            _, log_probs, entropy, vals_pred, hidden_state, act_logits = res
        else:
            res = self.net_model(large_maps=large_maps,
                                 small_maps=small_maps,
                                 hidden_state=hidden_state,
                                 action=actions)
            _, log_probs, entropy, vals_pred, hidden_state, act_logits = res
        vals_pred = torch.squeeze(vals_pred)
        vf_loss = 0.5 * self.val_loss_criterion(vals_pred, returns)

        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss1 = -advs * ratio
        pg_loss2 = -advs * torch.clamp(ratio,
                                       1 - self.config['cliprange'],
                                       1 + self.config['cliprange'])
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))
        loss = pg_loss - torch.mean(entropy) * self.config['ent_coef'] + \
               vf_loss * self.config['vf_coef']

        approxkl = 0.5 * torch.mean(torch.pow(old_log_probs - log_probs, 2))
        clipfrac = np.mean(np.abs(ratio.cpu().data.numpy() - 1.0) > self.config['cliprange'])

        self.optimizer.zero_grad()
        loss.backward()
        if self.config['max_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.net_model.parameters(),
                                           self.config['max_grad_norm'])
        self.optimizer.step()
        info = {'pg_loss': pg_loss.cpu().data.numpy(),
                'vf_loss': vf_loss.cpu().data.numpy(),
                'entropy': torch.mean(entropy).cpu().data.numpy(),
                'approxkl': approxkl.cpu().data.numpy(),
                'clipfrac': clipfrac}
        return info, hidden_state

    def n_rollout(self, repeat_num=1):
        n_large_maps, n_small_maps, nacts = [], [], []
        nret, nvals, nadvs = [], [], []
        if self.config['use_rgb_with_map']:
            n_rgbs = []
        nlog_probs, nepinfos = [], []

        house_ids = [None for i in range(repeat_num)]
        for j in range(repeat_num):
            res = self.rollout(house_ids=house_ids[j])
            obs, acts, ret, vals, advs, log_probs, epinfos = res
            if self.config['use_rgb_with_map']:
                n_rgbs.append(obs[0])
                n_large_maps.append(obs[1])
                n_small_maps.append(obs[2])
            else:
                n_large_maps.append(obs[0])
                n_small_maps.append(obs[1])
            nacts.append(acts)
            nret.append(ret)
            nvals.append(vals)
            nadvs.append(advs)
            nlog_probs.append(log_probs)
            nepinfos.extend(epinfos)
        comb = map(lambda p: np.concatenate(p, axis=1),
                   (n_large_maps, n_small_maps, nacts, nret, nvals, nadvs, nlog_probs))
        n_large_maps, n_small_maps, nacts, nret, nvals, nadvs, nlog_probs = comb
        if self.config['use_rgb_with_map']:
            n_rgbs = np.concatenate(n_rgbs, axis=1)
            return (n_rgbs, n_large_maps, n_small_maps), nacts, nret, nvals, \
                   nadvs, nlog_probs, nepinfos
        else:
            return (n_large_maps, n_small_maps), nacts, nret, nvals, nadvs, nlog_probs, nepinfos

    def rollout(self, val=False, house_ids=None):
        self.net_model.eval()
        mb_rgb, mb_large_maps, mb_small_maps, mb_rewards, mb_actions = [], [], [], [], []
        mb_values, mb_log_probs = [], []
        if val:
            env = self.val_env
            obs_uint8 = env.reset(self.house_config['fixed_val_set'])
        else:
            env = self.env
            obs_uint8 = env.reset(house_ids)
        epinfos = [{'reward': 0.0, 'seen_area': 0.0} for i in range(obs_uint8[0].shape[0])]
        hidden_state = self.net_model.init_hidden(batch_size=obs_uint8[0].shape[0])
        obs_order = [1, 2, 0]
        for idx in range(self.num_steps):
            if self.config['use_rgb_with_map']:
                obs = list(map(lambda p, device=self.device:
                               imagenet_rgb_preprocess(p, device), obs_uint8))

                mb_rgb.append(obs_uint8[0])
                obs = [obs[i] for i in obs_order]
            else:
                obs = list(map(lambda p, device=self.device:
                               imagenet_rgb_preprocess(p, device), obs_uint8[1:]))
            res = self.net_model(*obs, hidden_state=hidden_state)
            actions, log_probs, entropy, vals, hidden_state, act_logits = res
            mb_large_maps.append(obs_uint8[1])
            mb_small_maps.append(obs_uint8[2])

            mb_actions.append(np.squeeze(actions.cpu().data.numpy(), axis=0))
            mb_values.append(np.squeeze(vals.cpu().data.numpy(), axis=(0, -1)))
            mb_log_probs.append(np.squeeze(log_probs.cpu().data.numpy(), axis=0))
            obs_uint8, rewards, dones, infos = env.step(actions.cpu().data.numpy().flatten())
            mb_rewards.append(rewards)
            if idx == self.num_steps - 1:
                for k, info in enumerate(infos):
                    epinfos[k]['reward'] = info['reward_so_far']
                    epinfos[k]['seen_area'] = info['seen_area']
        if self.config['use_rgb_with_map']:
            mb_rgb = np.stack(mb_rgb, axis=0)
        mb_large_maps = np.stack(mb_large_maps, axis=0)
        mb_small_maps = np.stack(mb_small_maps, axis=0)
        mb_rewards = np.stack(mb_rewards, axis=0)
        mb_actions = np.stack(mb_actions, axis=0)

        mb_values = np.stack(mb_values, axis=0)
        mb_log_probs = np.stack(mb_log_probs, axis=0)

        if self.config['use_rgb_with_map']:
            obs = list(map(lambda p, device=self.device:
                           imagenet_rgb_preprocess(p, device), obs_uint8))
            obs = [obs[i] for i in obs_order]
        else:
            obs = list(map(lambda p, device=self.device:
                           imagenet_rgb_preprocess(p, device), obs_uint8[1:]))
        res = self.net_model(*obs, hidden_state=hidden_state)
        actions, log_probs, entropy, vals, hidden_state, act_logits = res

        last_values = np.squeeze(vals.cpu().data.numpy(), axis=-1)

        mb_advs, mb_returns = self.get_gae(mb_rewards, mb_values, last_values)
        if self.config['use_rgb_with_map']:
            return (mb_rgb, mb_large_maps, mb_small_maps), mb_actions, mb_returns, \
                   mb_values, mb_advs, mb_log_probs, epinfos
        else:
            return (mb_large_maps, mb_small_maps), mb_actions, mb_returns, mb_values, \
                   mb_advs, mb_log_probs, epinfos

    def get_gae(self, rewards, value_estimates, value_next):
        mb_advs = np.zeros_like(rewards)
        lastgaelam = 0
        value_estimates = np.concatenate((value_estimates, value_next), axis=0)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.config['gamma'] * value_estimates[t + 1] - value_estimates[t]
            mb_advs[t] = lastgaelam = delta + self.config['gamma'] * self.config['lam'] * lastgaelam
        mb_returns = mb_advs + value_estimates[:-1, ...]
        return mb_advs, mb_returns
