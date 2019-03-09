import json
import os


def filter_houses(all_houses, bad_houses):
    return [x for x in all_houses if x not in bad_houses]


def get_configs():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_path, 'paths.json')
    house_file = os.path.join(cur_path, '../../path_data/eqa_v1.json')
    with open(house_file, 'r') as f:
        house_data = json.load(f)
    bad_house_file = os.path.join(os.path.dirname(house_file),
                                  'bad_houses.json')
    with open(bad_house_file, 'r') as f:
        bad_house_ids = json.load(f)
    train_houses = filter_houses(house_data['splits']['train'],
                                 bad_house_ids)
    val_houses = filter_houses(house_data['splits']['val'],
                               bad_house_ids)
    test_houses = filter_houses(house_data['splits']['test'],
                                bad_house_ids)
    fix_test_house_ids = []
    try:
        with open(os.path.join(cur_path, 'test_set.json'), 'r') as f:
            fixed_test_set = json.load(f)
            for val_house in fixed_test_set:
                fix_test_house_ids.append(val_house['house_id'])
    except FileNotFoundError:
        fixed_test_set = []

    val_set = [{'x': 41, 'y': 42, 'yaw': 45,
                'house_id': '0f93cf8b5f6375521bfb84bf8a6f86b8'},
               {'x': 37.88, 'y': 42.98, 'yaw': -29.87,
                'house_id': '75a0d79f6c9b3d5ce058f218b30a35de'},
               {'x': 37.275, 'y': 42.075, 'yaw': -90,
                'house_id': '1e6902285d93d085bcee81befff27044'},
               {'x': 40.33, 'y': 40.85, 'yaw': -70.7,
                'house_id': 'e813a59797612b21448108f1d7023a39'},
               {'x': 31.93, 'y': 36.12, 'yaw': 106.94,
                'house_id': 'e12192d1e4a602a2e63d04b92f3cd7b3'},
               {'x': 32.79, 'y': 41.86, 'yaw': 19.69,
                'house_id': '78cbb14b35d75bff69d672df957dd190'},
               {'x': 37.8, 'y': 35.16, 'yaw': 69.3,
                'house_id': '8356de0efb8a5f994210e0318c21d977'},
               {'x': 45.38, 'y': 41.03, 'yaw': 108.69,
                'house_id': 'd800d4dc09e45f408f7f37802dfda765'}
               ]
    fix_val_house_ids = []
    for val_house in val_set:
        fix_val_house_ids.append(val_house['house_id'])
    configs = {
        'render_height': 224,  # height and width for opengl rendering
        'render_width': 224,
        'output_height': 80,  # height and width for rgb image to the agent/network
        'output_width': 80,
        'par_path': cur_path,
        'path': config_path,
        'move_sensitivity': 0.25,  # forward movement distance in one step
        'rot_sensitivity': 9,  # rotation angle in one step
        'GridDet': 0.05,  # resolution for collision map (number of cells in a row/column)
        'fixed_val_set': val_set,  # selected houses from val_houses
        'fixed_test_set': fixed_test_set,
        'fix_test_house_ids': fix_test_house_ids,
        'fixed_val_houses': fix_val_house_ids,
        'train_houses': train_houses[:20],
        'test_houses': test_houses,
        'val_houses': val_houses,
        'large_map_range': 400,  # large map size: (-400, 400) * GridDet
        'small_map_range': 40,  # small map size; (-40, 40) * GridDet
    }
    return configs
