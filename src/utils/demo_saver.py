import argparse
import json
import os
import shutil
from multiprocessing import Pool

import cv2
import numpy as np
from IPython import embed

from envs.house3dRGBD import House3DRGBD


def save_img(imgs, save_dir, idx):
    l_map_save_dir = os.path.join(save_dir, 'large_map')
    s_map_save_dir = os.path.join(save_dir, 'small_map')
    rgb_save_dir = os.path.join(save_dir, 'rgb')
    rgb_path = os.path.join(rgb_save_dir, '%06d.png' % idx)
    l_map_path = os.path.join(l_map_save_dir, '%06d.png' % idx)
    s_map_path = os.path.join(s_map_save_dir, '%06d.png' % idx)
    rgb = imgs[0]
    l_map = imgs[1]
    s_map = imgs[2]
    rgb = rgb[:, :, ::-1]
    cv2.imwrite(rgb_path, rgb)
    l_map = l_map[:, :, ::-1]
    cv2.imwrite(l_map_path, l_map)
    s_map = s_map[:, :, ::-1]
    cv2.imwrite(s_map_path, s_map)


def get_pose(loc, k):
    x = loc[k][0]
    y = loc[k][2]
    yaw = loc[k][-1]
    return np.array([x, y, yaw])


def save_demo(idx, demos, root_dir):
    env = House3DRGBD(train_mode=True,
                      area_reward_scale=0.00005,
                      collision_penalty=0.1,
                      step_penalty=0.001)
    # 0: Forward
    # 1: Turn Left
    # 2: Turn Right
    # 3: Strafe Left
    # 4: Strafe Right
    # 5: Backward
    act_encoding = {'fwd': 0, 'left': 1, 'right': 2,
                    'sleft': 3, 'sright': 4, 'bwd': 5}
    save_dir = os.path.join(root_dir, '%06d' % idx)
    if os.path.exists(save_dir):
        num = len(os.listdir(save_dir))
        if num < 4:
            shutil.rmtree(save_dir)
    rgb_save_dir = os.path.join(save_dir, 'rgb')
    l_map_save_dir = os.path.join(save_dir, 'large_map')
    s_map_save_dir = os.path.join(save_dir, 'small_map')
    if os.path.exists(rgb_save_dir):
        rgb_imgs = os.listdir(rgb_save_dir)
        if len(rgb_imgs) < 90:
            print('=====idx:', idx)
            embed()
        else:
            return
    for folder in [save_dir, rgb_save_dir, l_map_save_dir, s_map_save_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    loc = demos[idx]['loc']
    actions = demos[idx]['actions']
    traj_len = len(loc)
    x = loc[1][0]
    y = loc[1][2]
    yaw = loc[1][-1]
    prev_pose = get_pose(loc, 1)
    house_id = demos[idx]['house_id']

    obs_uint8 = env.reset(house_id=house_id, x=x, y=y, yaw=yaw)
    act_to_save = []
    img_idx = 0
    save_img(obs_uint8, save_dir, img_idx)
    for k in range(2, traj_len):
        t_act = act_encoding[actions[k - 1]]
        obs_uint8, rewards, dones, infos = env.step(t_act)
        cam_pos = np.array([env.env.cam.pos.x,
                            env.env.cam.pos.z,
                            env.env.cam.yaw])
        demo_pos = get_pose(loc, k)

        act_to_save.append(t_act)
        img_idx += 1
        prev_pose = demo_pos
        save_img(obs_uint8, save_dir, img_idx)
    print('saved states:', img_idx + 1)
    print('saved_actions:', len(act_to_save))
    info_data = {'house_id': house_id,
                 'act': act_to_save}
    with open(os.path.join(save_dir, 'info.json'), 'w') as f:
        json.dump(info_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file',
                        default='../path_data/cleaned_human_demo_data.json',
                        type=str)
    parser.add_argument('--save_dir', default='../../human_demo', type=str)
    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        demos = json.load(f)
    num_trajs = len(demos)
    root_dir = args.save_dir
    print('root_dir:', os.path.abspath(root_dir))
    print('num_trajs:', num_trajs)

    p = Pool(processes=8)
    args_in = [(i, demos, root_dir) for i in range(num_trajs)]
    p.starmap(save_demo, args_in)


if __name__ == '__main__':
    main()
