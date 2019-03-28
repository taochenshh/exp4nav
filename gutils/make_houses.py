# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import multiprocessing
import os
import shlex
import subprocess

parser = argparse.ArgumentParser(
    description='Create obj+mtl files for the houses in the dataset.')
parser.add_argument('-hf_name', help='house file name, this script will '
                    'convert hf_name.json to hf_name.obj', default='house')
parser.add_argument('-eqa_path', help='/path/to/eqa.json', required=True)
parser.add_argument(
    '-suncg_toolbox_path', help='/path/to/SUNCGtoolbox', required=True)
parser.add_argument(
    '-suncg_data_path', help='/path/to/suncg/data_root', required=True)
parser.add_argument(
    '-num_processes',
    help='number of threads to use',
    default=multiprocessing.cpu_count())
args = parser.parse_args()
args.suncg_data_path = os.path.abspath(args.suncg_data_path)
args.suncg_toolbox_path = os.path.abspath(args.suncg_toolbox_path)
eqa_data = json.load(open(args.eqa_path, 'r'))
houses = list(eqa_data['questions'].keys())
start_dir = os.getcwd()


def extract_threaded(house):
    os.chdir(os.path.join(args.suncg_data_path, 'house', house))
    subprocess.call(
        shlex.split('%s %s.json %s.obj' % (os.path.join(
            args.suncg_toolbox_path, 'gaps', 'bin', 'x86_64', 'scn2scn'), 
            args.hf_name, args.hf_name)))
    print('extracted', house)


pool = multiprocessing.Pool(args.num_processes)
pool.map(extract_threaded, houses)
