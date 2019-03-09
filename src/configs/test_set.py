import argparse
import json
import os

from tqdm import tqdm

from configs.house3d_config import get_configs
from envs.house3dRGBD import House3DRGBD


def main():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--eqa_file', type=str, default='../../path_data/eqa_v1.json')
    parser.add_argument('--num_houses', type=int, default=20)
    parser.add_argument('--ins_per_house', type=int, default=5)
    parser.add_argument('--suncg_dataset', type=str, default='../../../suncg_data')
    args = parser.parse_args()
    configs = get_configs()
    test_houses = configs['val_houses']
    env = House3DRGBD(train_mode=False,
                      area_reward_scale=0.0005,
                      collision_penalty=0.006,
                      step_penalty=0,
                      max_depth=3,
                      render_door=False,
                      start_indoor=True)
    test_houses_poses = []
    for i in tqdm(range(len(test_houses))):
        for j in range(args.ins_per_house):
            try:
                env.reset(house_id=test_houses[i])
            except IndexError:
                break
            print(env.start_pos)
            house_data = {'x': env.start_pos[0],
                          'y': env.start_pos[1],
                          'yaw': env.start_pos[2],
                          'house_id': test_houses[i]}
            test_houses_poses.append(house_data)
        if len(test_houses_poses) >= args.num_houses * args.ins_per_house:
            break
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'test_set.json'), 'w') as f:
        json.dump(test_houses_poses, f, indent=2)


if __name__ == '__main__':
    main()
