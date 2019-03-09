import random
from datetime import datetime

import numpy as np
import torch

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
    set_random_seed(args.seed)
    param_dict = {
        "train_mode": not args.test,
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
    if args.test:
        val_env = None
    else:
        val_env = make_n_envs(min(args.num_envs, 8), **param_dict)

    agent = PPOAgentMap(env=env, args=args, val_env=val_env)

    if args.test:
        for i in range(0, 100):
            agent.test(render=args.render, val_id=i)
    else:
        agent.train()
    env.close()


if __name__ == "__main__":
    main()
