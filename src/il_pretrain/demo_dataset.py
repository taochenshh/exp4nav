import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


class HumanDemoDatasetRGBMAP(Dataset):
    def __init__(self, root_dir, seq_len=100,
                 train=True, train_ratio=0.95):
        self.root_dir = root_dir
        self.seq_len = seq_len
        img_folders = os.listdir(self.root_dir)
        img_folders.sort()
        img_folders = [os.path.join(self.root_dir, x)
                       for x in img_folders]
        train_num = int(train_ratio * len(img_folders))

        if train:
            self.img_folders = img_folders[:train_num]
        else:
            self.img_folders = img_folders[train_num:]

    def __len__(self):
        return len(self.img_folders)

    def __getitem__(self, idx):
        img_dir = self.img_folders[idx]
        with open(os.path.join(img_dir, 'info.json'), 'r') as f:
            info_data = json.load(f)
        traj_actions = info_data['act']
        traj_len = len(traj_actions)
        try:
            assert traj_len >= self.seq_len
        except AssertionError:
            from IPython import embed
            embed()
        start_idx = random.randint(0, traj_len - self.seq_len)
        rgbs = []
        l_maps = []
        s_maps = []
        for k in range(start_idx, start_idx + self.seq_len + 1):
            rgb_path = os.path.join(img_dir, 'rgb', '%06d.png' % k)
            l_map_path = os.path.join(img_dir, 'large_map', '%06d.png' % k)
            s_map_path = os.path.join(img_dir, 'small_map', '%06d.png' % k)
            rgb = cv2.imread(rgb_path)
            l_map = cv2.imread(l_map_path)
            s_map = cv2.imread(s_map_path)
            rgb = rgb[:, :, ::-1]
            l_map = l_map[:, :, ::-1]
            s_map = s_map[:, :, ::-1]
            rgbs.append(rgb)
            l_maps.append(l_map)
            s_maps.append(s_map)

        actions = traj_actions[start_idx: start_idx + self.seq_len]

        rgbs, l_maps, s_maps, actions = map(lambda p: np.array(p), (rgbs, l_maps, s_maps, actions))
        actions = actions.astype(np.int32)

        return rgbs, l_maps, s_maps, actions
