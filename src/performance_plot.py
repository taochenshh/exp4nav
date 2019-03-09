import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

FONT_SIZE = 20
AREA_PER_CELL = 0.0025


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--steps', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    algo_data_dirs = os.listdir(args.data_dir)
    rcParams.update({'figure.autolayout': True,
                     'legend.fontsize': 16})

    area_fig, area_ax = plt.subplots(figsize=(6, 6))
    len_threshold = args.steps
    for algo_data_dir in algo_data_dirs:
        algo_data_dir_path = os.path.join(args.data_dir, algo_data_dir)

        data_file = os.path.join(algo_data_dir_path, 'data_1000.json')
        time_steps = np.arange(len_threshold)

        if not os.path.exists(data_file):
            continue
        with open(data_file, 'r') as f:
            data = json.load(f)
        cum_area = [x[:len_threshold] for x in data['cum_area'] if len(x) >= len_threshold]

        cum_area = np.array(cum_area)
        cum_area_mean = np.mean(cum_area, axis=0) * AREA_PER_CELL
        area_ax.plot(time_steps, cum_area_mean, label=algo_data_dir,
                     linewidth=2)

    area_ax.set_xlabel('Time steps', fontsize=FONT_SIZE)
    area_ax.set_ylabel('Coverage (in $m^2$)', fontsize=FONT_SIZE)

    area_ax.tick_params(axis='x', labelsize=FONT_SIZE)
    area_ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    area_ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    area_ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    area_ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    area_ax.set_ylim(-12.5, 170)
    area_ax.legend()
    area_ax.grid(True)
    gridlines = area_ax.get_xgridlines() + area_ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')
    plt.show()


if __name__ == '__main__':
    main()
