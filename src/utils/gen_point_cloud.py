import json

import cv2
import numpy as np

try:
    from open3d import *

    USE_OPEN3D = True
except:
    print('Open3d not found, point cloud cannot be visualized')
    USE_OPEN3D = False
import os.path as osp

Image_changed = True
Height = None
Width = None
Iinv = None
Cam_to_img_mat = None
Img_pixs_ones = None


def img_to_world(depth, rgb, E, depth_threshold=None):
    global Image_changed, Height, Width, Iinv, Cam_to_img_mat, Img_pixs_ones
    # use global variable here to compute some values just once
    # so as to speed up calculation
    if Height is None or Height != depth.shape[0]:
        Height = depth.shape[0]
        Image_changed = True
    if Width is None or Width != depth.shape[1]:
        Width = depth.shape[1]
        Image_changed = True
    if Image_changed:
        img_pixs = np.mgrid[0: depth.shape[0], 0: depth.shape[1]].reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]  # swap (v, u) into (u, v)
        Img_pixs_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        if Iinv is None:
            I = get_intrinsics(height=Height, width=Width, vfov=60)
            Iinv = np.linalg.inv(I[:3, :3])
        Cam_to_img_mat = np.dot(Iinv, Img_pixs_ones)
        Image_changed = False

    depth_vals = depth.reshape(-1)
    rgb_vals = rgb.reshape(-1, 3)
    if depth_threshold is not None:
        assert type(depth_threshold) is tuple
        valid = depth_vals > depth_threshold[0]
        if len(depth_threshold) > 1:
            valid = np.logical_and(valid,
                                   depth_vals < depth_threshold[1])
        depth_vals = depth_vals[valid]
        rgb_vals = rgb_vals[valid]
        points_in_cam = np.multiply(Cam_to_img_mat[:, valid], depth_vals)
    else:
        points_in_cam = np.multiply(Cam_to_img_mat, depth_vals)
    points_in_cam = np.concatenate((points_in_cam,
                                    np.ones((1, points_in_cam.shape[1]))),
                                   axis=0)
    points_in_world = np.dot(E, points_in_cam)
    points_in_world = points_in_world.T
    return points_in_world[:, :3], rgb_vals


def get_intrinsics(height, width, vfov):
    # calculate the intrinsic matrix from vertical_fov
    # notice that hfov and vfov are different if height != width
    # we can also get the intrinsic matrix from opengl's
    # perspective matrix
    # http://kgeorge.github.io/2014/03/08/calculating-opengl-
    # perspective-matrix-from-opencv-intrinsic-matrix
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * width / float(height)
    fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = height / 2.0 / tan_half_vfov
    I = np.array([[fx, 0, width / 2.0],
                  [0, fy, height / 2.0],
                  [0, 0, 1]])
    return I


def gen_point_cloud(depth, rgb, E, depth_threshold=(0.,), inv_E=True):
    if inv_E:
        E = np.linalg.inv(np.array(E).reshape((4, 4)))
    # different from real-world camera coordinate system
    # opengl uses negative z axis as the camera front direction
    # x axes are same, hence y axis is reversed as well
    # https://learnopengl.com/Getting-started/Camera
    rot = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    E = np.dot(E, rot)
    points, points_rgb = img_to_world(depth, rgb, E,
                                      depth_threshold=depth_threshold)
    return points, points_rgb


def show_point_cloud(depth_img, rgb_img, cam_pos_file, dest):
    depth = cv2.imread(osp.join(dest, depth_img))[:, :, 0]
    depth = depth.astype(np.float32) / 255.0 * 20.0
    rgb = cv2.imread(osp.join(dest, rgb_img))
    rgb = rgb[:, :, ::-1]
    with open(osp.join(dest, cam_pos_file), 'r') as f:
        data = json.load(f)
    extrinsics = data['extrinsics']
    return gen_point_cloud(depth, rgb, extrinsics)


if __name__ == '__main__':
    FILTER_HEIGHT = True
    dest = 'imgs'
    pcds = []
    files = ['depth.png', 'rgb.png', 'cam.json']
    all_points = np.empty((0, 3))
    all_colors = np.empty((0, 3))
    for i in range(3):
        new_files = []
        for file in files:
            file_split = file.split('.')
            file = '.'.join([file_split[0] + '_%d' % i, file_split[1]])
            new_files.append(file)
        points, colors = show_point_cloud(*new_files, dest=dest)
        all_points = np.concatenate((all_points, points))
        all_colors = np.concatenate((all_colors, colors))

    if FILTER_HEIGHT:
        height_lim = [0.1, 2.3]
        valid_idx = np.logical_and(all_points[:, 1] > height_lim[0],
                                   all_points[:, 1] < height_lim[1])
        all_points = all_points[valid_idx]
        all_colors = all_colors[valid_idx]
    if USE_OPEN3D:
        pcd = PointCloud()
        pcd.points = Vector3dVector(all_points)
        pcd.colors = Vector3dVector(all_colors / 255.0)
        coord = create_mesh_coordinate_frame(3, [40, 0, 35])
        draw_geometries([pcd, coord])
