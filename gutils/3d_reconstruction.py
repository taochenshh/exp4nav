import argparse
import json

import cv2
import numpy as np
from House3D import objrender, Environment, load_config
from House3D.objrender import RenderMode
from gen_point_cloud import gen_point_cloud

try:
    from open3d import *

    USE_OPEN3D = True
except:
    print('Open3d not found, point cloud cannot be visualized')
    USE_OPEN3D = False
from tqdm import tqdm

'''
eqa_humans.json
'loc': [x, z (robot height), y, yaw]
'''
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_mat(env, text_height, text):
    mat = env.debug_render()
    # show question and answer in the bottom
    pad_text = np.ones((text_height, mat.shape[1], mat.shape[2]), dtype=np.uint8)
    mat_w_text = np.concatenate((np.copy(mat), pad_text), axis=0)
    cv2.putText(mat_w_text, text, (10, mat_w_text.shape[0] - 10),
                FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return mat_w_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        default='../path_data/eqa_humans.json',
                        type=str)
    parser.add_argument('--demo_id', default=0, type=int)
    parser.add_argument('--width', type=int, default=600)
    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--filter_height', type=bool, default=True)
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

    cfg = load_config('config.json')
    api = objrender.RenderAPI(w=args.width, h=args.height, device=0)
    env = Environment(api, demo['house_id'], cfg)

    L_min = env.house.L_min_coor
    L_max = env.house.L_max_coor
    L_min = np.array([[env.house.L_lo[0], L_min[1], env.house.L_lo[1]]])
    L_max = np.array([[env.house.L_hi[0], L_max[1], env.house.L_hi[1]]])
    grid_size = env.house.grid_det
    n_row = env.house.n_row
    grid_num = np.array([n_row[0] + 1,
                         int((L_max[0][1] - L_min[0][1]) / (grid_size + 1e-8)) + 1,
                         n_row[1] + 1])
    print('Grid size:', grid_size)
    print('Number of grid in [x, y, z]:', grid_num)

    all_grids = np.zeros(tuple(grid_num), dtype=bool)
    grid_colors = np.zeros(tuple(grid_num) + (3,), dtype=np.uint8)
    loc_map = env.gen_locmap()
    obs_map = env.house.obsMap.T
    obs_pos = obs_map == 1
    for t in tqdm(range(len(locs))):
        env.reset(x=locs[t][0], y=locs[t][2], yaw=locs[t][3])
        depth = env.render(RenderMode.DEPTH)
        rgb = env.render(RenderMode.RGB)
        semantic = env.render(RenderMode.SEMANTIC)
        infmask = depth[:, :, 1]
        depth = depth[:, :, 0] * (infmask == 0)
        true_depth = depth.astype(np.float32) / 255.0 * 20.0
        extrinsics = env.cam.getExtrinsicsNumpy()
        points, points_colors = gen_point_cloud(true_depth, rgb, extrinsics)
        grid_locs = np.floor((points - L_min) / grid_size).astype(int)
        all_grids[grid_locs[:, 0], grid_locs[:, 1], grid_locs[:, 2]] = True
        grid_colors[grid_locs[:, 0], grid_locs[:, 1], grid_locs[:, 2]] = points_colors
        depth = np.stack([depth] * 3, axis=2)

        loc_map[grid_locs[:, 2], grid_locs[:, 0], :] = np.array([250, 120, 120])
        loc_map[obs_pos] = 0
        rad = env.house.robotRad / env.house.grid_det
        x, y = env.cam.pos.x, env.cam.pos.z
        x, y = env.house.to_grid(x, y)
        loc_map_cp = loc_map.copy()
        cv2.circle(loc_map_cp, (x, y), int(rad), (0, 0, 255), thickness=-1)
        loc_map_resized = cv2.resize(loc_map_cp, env.resolution)
        concat1 = np.concatenate((rgb, semantic), axis=1)
        concat2 = np.concatenate((depth, loc_map_resized), axis=1)
        ret = np.concatenate((concat1, concat2), axis=0)
        ret = ret[:, :, ::-1]
        pad_text = np.ones((text_height, ret.shape[1], ret.shape[2]), dtype=np.uint8)
        mat_w_text = np.concatenate((np.copy(ret), pad_text), axis=0)
        cv2.putText(mat_w_text, text, (10, mat_w_text.shape[0] - 10),
                    FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("human_demo", mat_w_text)
        cv2.waitKey(20)

    valid_grids = np.argwhere(all_grids)
    valid_grid_center = valid_grids * grid_size + L_min
    valid_grid_color = grid_colors[valid_grids[:, 0],
                                   valid_grids[:, 1],
                                   valid_grids[:, 2]]

    if args.filter_height:
        height_lim = [-0.1, 2.3]
        valid_idx = np.logical_and(valid_grid_center[:, 1] >= height_lim[0],
                                   valid_grid_center[:, 1] <= height_lim[1])

        valid_grid_center = valid_grid_center[valid_idx]
        valid_grid_color = valid_grid_color[valid_idx]
        valid_grids = valid_grids[valid_idx]

    loc_map = env.gen_locmap()

    obs_map = env.house.obsMap.T
    loc_map[valid_grids[:, 2], valid_grids[:, 0], :] = np.array([250, 120, 120])
    loc_map[obs_map == 1] = 0

    loc_map = cv2.resize(loc_map, env.resolution)
    cv2.imwrite('seen_map.png', loc_map)
    if USE_OPEN3D:
        pcd = PointCloud()
        pcd.points = Vector3dVector(valid_grid_center)
        pcd.colors = Vector3dVector(valid_grid_color / 255.0)
        coord = create_mesh_coordinate_frame(3, [40, 0, 35])
        draw_geometries([pcd, coord])


if __name__ == '__main__':
    main()
