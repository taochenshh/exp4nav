import argparse
import json
import pathlib
from itertools import count

import cv2
import numpy as np
from House3D import objrender, create_default_config
from House3D.objrender import RenderMode


def key_control(mat, cam):
    print('press \'esc\' or \'q\' to quit')
    cv2.imshow("window", mat)
    key = cv2.waitKey(0)
    if key == 27 or key == ord('q'):  # esc
        return True
    elif key == ord('w'):
        cam.pos += cam.front * 0.5
    elif key == ord('s'):
        cam.pos -= cam.front * 0.5
    elif key == ord('a'):
        cam.pos -= cam.right * 0.5
    elif key == ord('d'):
        cam.pos += cam.right * 0.5
    elif key == ord('h'):
        cam.yaw -= 5
        # need to call updateDirection to make the change to yaw/pitch
        # take effect
        cam.updateDirection()
    elif key == ord('l'):
        cam.yaw += 5
        cam.updateDirection()
    elif key == ord('j'):
        cam.pitch -= 5
        # need to call updateDirection to make the change to yaw/pitch
        # take effect
        cam.updateDirection()
    elif key == ord('k'):
        cam.pitch += 5
        cam.updateDirection()
    elif key == ord('z'):
        cam.pos += cam.up * 0.5
    elif key == ord('x'):
        cam.pos -= cam.up * 0.5
    else:
        print("Unknown key:", key)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        default='../path_data/cleaned_human_demo_data.json',
                        type=str)
    parser.add_argument('--suncg_house_dir',
                        default='../../suncg_data/house',
                        type=str)
    parser.add_argument('--mode',
                        default='rgb',
                        type=str,
                        choices=['rgb', 'semantic', 'instance', 'depth'])
    parser.add_argument('--demo_id',
                        default=0,
                        type=int)
    parser.add_argument('--width',
                        type=int,
                        default=800)
    parser.add_argument('--height',
                        type=int,
                        default=600)
    parser.add_argument('--device',
                        type=int,
                        default=0)
    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    demo = data[args.demo_id]

    cfg = create_default_config('.')
    api = objrender.RenderAPI(args.width, args.height, device=args.device)
    api.printContextInfo()

    obj_path = pathlib.Path(args.suncg_house_dir).joinpath(demo['house_id'], 'house.obj')
    print('You select house id: {0:s}'.format(demo['house_id']))
    api.loadScene(str(obj_path.resolve()), cfg['modelCategoryFile'], cfg['colorFile'])
    cam = api.getCamera()
    api_mode = getattr(RenderMode, args.mode.upper())
    api.setMode(api_mode)

    for t in count():
        mat = np.array(api.render())
        if api_mode == RenderMode.DEPTH:
            infmask = mat[:, :, 1]
            mat = mat[:, :, 0] * (infmask == 0)
        else:
            mat = mat[:, :, ::-1]

        if api_mode == RenderMode.INSTANCE:
            center_rgb = mat[args.height // 2, args.width // 2, ::-1]
            center_instance = api.getNameFromInstanceColor(center_rgb[0],
                                                           center_rgb[1],
                                                           center_rgb[2])
            print("Instance ID in the center: ", center_instance)
        exit_flag = key_control(mat, cam)
        if exit_flag:
            break


if __name__ == '__main__':
    main()
