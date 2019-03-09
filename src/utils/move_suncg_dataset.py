import argparse
import os
import shutil

from tqdm import tqdm

from configs.house3d_config import get_configs


def copy_files_with_id(ids, suncg_dataset, root_dest_dir):
    for idx in tqdm(ids):
        src_dir = os.path.join(suncg_dataset, 'house', idx)
        dest_dir = os.path.join(root_dest_dir, 'house', idx)
        if not os.path.exists(dest_dir):
            shutil.copytree(src_dir, dest_dir)


def main():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--suncg_dataset', type=str, default='../../../suncg_data')
    parser.add_argument('--dest_dir', type=str, default='../../../nav_data')
    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)

    configs = get_configs()
    val_sets = configs['fixed_val_set']
    val_house_ids = []
    for val_house in val_sets:
        val_house_ids.append(val_house['house_id'])

    total_ids = list(set(configs['train_houses'] + val_house_ids + configs['fix_test_house_ids']))
    print('total num:', len(total_ids))
    copy_files_with_id(total_ids, args.suncg_dataset, args.dest_dir)


if __name__ == '__main__':
    main()
