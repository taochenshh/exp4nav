import argparse
import json
import os

import cv2
import numpy as np
from House3D import objrender, Environment, load_config

'''
eqa_humans.json
'loc': [x, z (robot height), y, yaw]
'''
FONT = cv2.FONT_HERSHEY_SIMPLEX


def print_keyboard_help(mode):
    if mode == 'auto':
        print(' press \'i\' to speed up play')
        print(' press \'d\' to slow down play')
        print(' press \'r\' to restart play from beginning')
        print(' press \'q\' or \'esc\' to exit')
    elif mode == 'key':
        print(' press \'n\' to play forward by one step')
        print(' press \'p\' to play backward by one step')
        print(' press \'r\' to restart play from beginning')
        print(' press \'q\' or \'esc\' to exit')


def get_mat(env, text_height, text):
    mat = env.debug_render()
    # show question and answer in the bottom
    pad_text = np.ones((text_height, mat.shape[1],
                        mat.shape[2]), dtype=np.uint8)
    mat_w_text = np.concatenate((np.copy(mat), pad_text), axis=0)
    cv2.putText(mat_w_text, text, (10, mat_w_text.shape[0] - 10),
                FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return mat_w_text


def save_demo_video(env, locs, text, args, text_height, save_folder):
    print('saving video ...')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('{0:s}/demo_{1:d}_video.avi'
                                ''.format(save_folder, args.demo_id), fourcc,
                                6.0, (2 * args.width,
                                      2 * args.height + text_height))
    for t in range(len(locs)):
        env.reset(x=locs[t][0], y=locs[t][2], yaw=locs[t][3])
        mat = get_mat(env, text_height, text)
        video_out.write(mat)
    video_out.release()


def save_demo_map(env, locs, demo_id, save_folder):
    print('saving map ...')
    map2d_demo = env.gen_2dmap_with_traj(xs=locs[:, 0], ys=locs[:, 2])
    map2d_demo = map2d_demo[:, :, ::-1]
    cv2.imwrite('{0:s}/demo_{1:d}_map2d.png'
                ''.format(save_folder, demo_id), map2d_demo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        default='../path_data/cleaned_human_demo_data.json',
                        type=str)
    parser.add_argument('--demo_id',
                        default=0,
                        type=int)
    parser.add_argument('--width',
                        type=int,
                        default=600)
    parser.add_argument('--height',
                        type=int,
                        default=450)
    parser.add_argument('--no_display',
                        action='store_true',
                        help='disable display on screen')
    parser.add_argument('--play_mode',
                        default='auto',
                        type=str,
                        choices=['key', 'auto'],
                        help='play the human demonstration in '
                             'keyboard-control mode or auto-play mode')
    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    demo = data[args.demo_id]
    print('Total time steps:', len(demo['loc']))
    locs = np.array(demo['loc'][1:])
    ques = demo['question']
    answer = demo['answer']
    text = 'Q: {0:s}   A: {1:s}'.format(ques, answer)
    text_height = 60
    auto_freq = 15  # 15 hz

    cfg = load_config('config.json')
    api = objrender.RenderAPI(w=args.width, h=args.height, device=0)
    env = Environment(api, demo['house_id'], cfg, GridDet=0.05)

    save_folder = 'demo_{0:d}'.format(args.demo_id)
    os.makedirs(save_folder, exist_ok=True)

    if not args.no_display:
        print_keyboard_help(args.play_mode)
        t = 0
        while True:
            # print(t)
            if t < len(locs):
                env.reset(x=locs[t][0], y=locs[t][2], yaw=locs[t][3])
            mat = get_mat(env, text_height, text)
            cv2.imshow("human_demo", mat)
            if args.play_mode == 'key':
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):  # esc
                    break
                elif key == ord('n'):
                    if t < len(locs) - 1:
                        t += 1
                elif key == ord('p'):
                    if t > 0:
                        t -= 1
                elif key == ord('r'):
                    t = 0
                else:
                    print("Unknown key: {}".format(key))
            else:
                key = cv2.waitKey(int(1000 / auto_freq))
                if key == 27 or key == ord('q'):  # esc
                    break
                elif key == -1:
                    if t < len(locs) - 1:
                        t += 1
                elif key == ord('r'):
                    t = 0
                    auto_freq = 15
                elif key == ord('i'):
                    if auto_freq < 30:
                        auto_freq += 4
                elif key == ord('d'):
                    if auto_freq > 3:
                        auto_freq -= 4
                else:
                    print("Unknown key: {}".format(key))


if __name__ == '__main__':
    main()
