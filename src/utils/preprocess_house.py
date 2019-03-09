# generate the occupancy map etc.
import argparse
import json
import os

from House3D import objrender, Environment, load_config
from tqdm import tqdm

from configs.house3d_config import get_configs


def gen_cache_files(ids, skip_file):
    configs = get_configs()
    config = load_config(configs['path'],
                         prefix=configs['par_path'])
    render_height = configs['render_height']
    render_width = configs['render_width']
    with open(skip_file, 'r') as f:
        skip_houses = json.load(f)
    for idx in tqdm(ids):
        if idx in skip_houses:
            continue
        print(idx)
        api = objrender.RenderAPI(render_width, render_height, device=0)
        try:
            env = Environment(api, idx, config,
                              GridDet=configs['GridDet'])
        except AssertionError:
            skip_houses.append(idx)
    return skip_houses


def main():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--eqa_file', type=str, default='../path_data/eqa_v1.json')
    parser.add_argument('--skip_file', type=str, default='../path_data/bad_houses.json')
    args = parser.parse_args()
    with open(args.eqa_file, 'r') as f:
        eqa_dataset = json.load(f)
    train_ids = eqa_dataset['splits']['train']
    val_ids = eqa_dataset['splits']['val']
    test_ids = eqa_dataset['splits']['test']
    total_ids = train_ids + val_ids + test_ids
    print('train num:', len(train_ids))
    print('test num:', len(test_ids))
    print('val num:', len(val_ids))
    print('total num:', len(total_ids))
    bad_houses = gen_cache_files(total_ids, args.skip_file)
    if len(bad_houses) > 0:
        save_file = os.path.join(args.skip_file)
        with open(save_file, 'w') as f:
            json.dump(bad_houses, f, indent=2)


if __name__ == '__main__':
    main()
