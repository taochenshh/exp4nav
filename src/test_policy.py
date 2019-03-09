import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from algos.ppo_map import PPOAgentMap
from configs.parse_args import parse_arguments
from envs.house3dRGBD import House3DRGBD
from utils.color_print import *
from utils.subproc_vec_env import SubprocVecEnv


def make_n_envs(num,
                train_mode=True,
                area_reward_scale=1,
                collision_penalty=0.1,
                step_penalty=0.0005,
                max_depth=10.0,
                render_door=False,
                start_indoor=False,
                ob_dilation_kernel=5,
                large_map_size=80):
    param_dict = {
        "train_mode": train_mode,
        "area_reward_scale": area_reward_scale,
        "collision_penalty": collision_penalty,
        "step_penalty": step_penalty,
        "max_depth": max_depth,
        "render_door": render_door,
        "start_indoor": start_indoor,
        "ob_dilation_kernel": ob_dilation_kernel,
        "large_map_size": large_map_size,
    }

    def make_env(**kwargs):
        def _thunk():
            env = House3DRGBD(**kwargs)
            return env

        return _thunk

    return SubprocVecEnv([make_env(**param_dict)
                          for i in range(num)])


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = parse_arguments()
    print_green('Program starts at: \033[92m %s '
                '\033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    args.device = None
    args.device = torch.device("cuda:0" if torch.cuda.is_available()
                                           and not args.disable_cuda
                               else "cpu")
    args.test = True
    args.num_envs = 1
    args.max_depth = 3
    set_random_seed(args.seed)
    set_random_seed(args.seed)
    param_dict = {
        "train_mode": True,
        "area_reward_scale": args.area_reward_scale,
        "collision_penalty": args.collision_penalty,
        "step_penalty": args.step_penalty,
        "max_depth": args.max_depth,
        "render_door": args.render_door,
        "start_indoor": args.start_indoor,
        "ob_dilation_kernel": args.ob_dilation_kernel,
        "large_map_size": args.large_map_size
    }
    env = make_n_envs(args.num_envs, **param_dict)

    val_env = None

    agent = PPOAgentMap(env=env, args=args, val_env=val_env)
    env_cum_reward = []
    env_cum_area = []
    env_actions = []

    for i in tqdm(range(100), desc='Env id'):
        cum_area, cum_reward, cum_actions = agent.test(render=args.render, val_id=i)
        move_area, move_reward, move_actions = [], [], []
        for area, reward, action in zip(cum_area, cum_reward, cum_actions):
            move_area.append(float(area))
            move_reward.append(float(reward))
            move_actions.append(int(action[0]))
        env_cum_area.append(move_area)
        env_cum_reward.append(move_reward)
        env_actions.append(move_actions)
    agent.env.close()
    print('valid envs:', len(env_cum_reward))

    with open(os.path.join(args.save_dir, 'hyperparams.json'), 'r') as f:
        hp = json.load(f)
    data_to_save = {'hyperparams': hp,
                    'cum_reward': env_cum_reward,
                    'cum_area': env_cum_area,
                    'actions': env_actions}
    save_dir = args.save_dir
    with open(os.path.join(save_dir, 'data_%d.json' % args.num_steps), 'w') as f:
        json.dump(data_to_save, f, indent=2)


if __name__ == "__main__":
    main()
