import argparse
import csv
import json
import os
import pickle
from collections import OrderedDict

from House3D import load_config
from tqdm import tqdm

from configs.house3d_config import get_configs


def get_door_ids(modelCategoryFile):
    return get_class_ids(['door', 'fence', 'arch'],
                         modelCategoryFile)


def get_class_ids(names, modelCategoryFile):
    # load all the doors
    target_match_class = 'nyuv2_40class'
    ids = set()
    with open(modelCategoryFile) as csvfile:
        reader = csv.DictReader(csvfile)
        cls = []
        for row in reader:
            cls.append(row[target_match_class])
            if row[target_match_class] in names:
                ids.add(row['model_id'])
    return ids


def write_json(input_json_file, output_json_file, door_ids):
    with open(input_json_file) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    level = data['levels'][0]
    level['nodes'] = [node for node in level['nodes']
                      if node['modelId'] not in door_ids]
    with open(output_json_file, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


def save_variables(pickle_file_name, var, info, overwrite=False):
    if os.path.exists(pickle_file_name) and not overwrite:
        raise Exception('{:s} exists and over write is false.'
                        ''.format(pickle_file_name))
    # Construct the dictionary
    assert (type(var) == list);
    assert (type(info) == list);
    d = {}
    for i in range(len(var)):
        d[info[i]] = var[i]
    with open(pickle_file_name, 'w') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='Remove components in House3D')
    parser.add_argument('--eqa_file', type=str,
                        default='../../path_data/eqa_v1.json')
    args = parser.parse_args()
    with open(args.eqa_file, 'r') as f:
        eqa_dataset = json.load(f)
    train_ids = eqa_dataset['splits']['train']
    val_ids = eqa_dataset['splits']['val']
    test_ids = eqa_dataset['splits']['test']
    total_ids = train_ids + val_ids + test_ids

    configs = get_configs()
    config = load_config(configs['path'],
                         prefix=configs['par_path'])
    door_ids = get_door_ids(config['modelCategoryFile'])
    for idx in tqdm(total_ids):
        src_folder = os.path.join(config['prefix'], idx)
        input_json_file = os.path.join(src_folder, 'house.json')
        output_json_file = os.path.join(src_folder, 'house-no-doors.json')
        write_json(input_json_file, output_json_file, door_ids)


if __name__ == '__main__':
    main()
