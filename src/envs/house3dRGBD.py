import os

import cv2
import numpy as np
from House3D import objrender, Environment, load_config
from scipy.stats import truncnorm

from configs.house3d_config import get_configs
from utils import seeding
from utils.gen_point_cloud import gen_point_cloud

RENDERING_GPU = 0
CUDA_DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES')
if CUDA_DEVICE:
    RENDERING_GPU = int(CUDA_DEVICE)
HEIGHT_THRESHOLD = (0.2, 1.8)


class House3DRGBD:
    def __init__(self,
                 train_mode=True,
                 area_reward_scale=1,
                 collision_penalty=0.1,
                 step_penalty=0.0005,
                 max_depth=3.0,
                 render_door=False,
                 start_indoor=False,
                 ignore_collision=False,
                 ob_dilation_kernel=5,
                 large_map_size=80):
        self.seed()
        self.configs = get_configs()
        self.configs['large_map_size'] = large_map_size
        self.env = None

        self.train_mode = train_mode
        self.render_door = render_door
        self.ignore_collision = ignore_collision
        self.start_indoor = start_indoor
        self.render_height = self.configs['render_height']
        self.render_width = self.configs['render_width']
        self.img_height = self.configs['output_height']
        self.img_width = self.configs['output_width']
        self.ob_dilation_kernel = ob_dilation_kernel
        self.config = load_config(self.configs['path'],
                                  prefix=self.configs['par_path'])
        self.move_sensitivity = self.configs['move_sensitivity']
        self.rot_sensitivity = self.configs['rot_sensitivity']
        self.train_houses = self.configs['train_houses']
        self.test_houses = self.configs['test_houses']

        if train_mode:
            self.houses_id = self.train_houses
            # print("Number of traning houses:", len(self.houses_id))
        else:
            self.houses_id = None  # self.test_houses
        self.depth_threshold = (0, max_depth)
        self.area_reward_scale = area_reward_scale
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.observation_space = [self.img_width, self.img_height, 3]
        self.action_space = [6]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, house_id=None, x=None, y=None, yaw=None):
        if house_id is None:
            house_id = self.np_random.choice(self.houses_id, 1)[0]
        self.hid = house_id
        if self.env is not None:
            del self.api
            del self.env
        self.api = objrender.RenderAPI(self.render_width,
                                       self.render_height,
                                       device=RENDERING_GPU)

        self.env = Environment(self.api, house_id, self.config,
                               GridDet=self.configs['GridDet'],
                               RenderDoor=self.render_door,
                               StartIndoor=self.start_indoor)

        if not self.train_mode:
            self.loc_map = self.env.gen_locmap()
            obs_map = self.env.house.obsMap.T
            self.obs_pos = obs_map == 1
            self.traj = []
            self.traj_actions = []
            self.grid_traj = []

        self.L_min = self.env.house.L_lo
        self.L_max = self.env.house.L_hi
        self.grid_size = self.env.house.grid_det
        grid_num = np.array([self.env.house.n_row[0] + 1,
                             self.env.house.n_row[1] + 1])
        self.grids_mat = np.zeros(tuple(grid_num), dtype=np.uint8)
        self.max_grid_size = np.max(grid_num)
        self.max_seen_area = float(np.prod(grid_num))
        self.env.reset(x=x, y=y, yaw=yaw)
        self.start_pos, self.grid_start_pos = self.get_camera_grid_pos()
        if not self.train_mode:
            print('start pose: ', self.start_pos)
            self.traj.append(self.start_pos.tolist())
            self.grid_traj.append(self.grid_start_pos.tolist())
        rgb, depth, extrinsics = self.get_obs()
        large_loc_map, small_loc_map = self.get_loc_map()
        self.seen_area = self.get_seen_area(rgb, depth,
                                            extrinsics,
                                            self.grids_mat)

        self.ep_len = 0
        self.ep_reward = 0
        self.collision_times = 0
        ob = (self.resize_img(rgb), large_loc_map, small_loc_map)
        return ob

    def step(self, action):
        if self.ignore_collision:
            collision_flag = self.motion_primitive_no_check(action)
        else:
            collision_flag = self.motion_primitive(action)
        rgb, depth, extrinsics = self.get_obs()
        large_loc_map, small_loc_map = self.get_loc_map()
        if not self.train_mode:
            current_pos, grid_current_pos = self.get_camera_grid_pos()
            self.traj_actions.append(int(action))
            self.traj.append(current_pos.tolist())
            self.grid_traj.append(grid_current_pos.tolist())

        reward, seen_area, raw_reward = self.cal_reward(rgb,
                                                        depth,
                                                        extrinsics,
                                                        collision_flag)
        self.ep_len += 1
        self.ep_reward += reward
        if collision_flag:
            self.collision_times += 1
        info = {'reward_so_far': self.ep_reward, 'steps_so_far': self.ep_len,
                'seen_area': seen_area, 'collisions': self.collision_times,
                'start_pose': self.start_pos, 'house_id': self.hid,
                'collision_flag': collision_flag}
        info = {**info, **raw_reward}
        done = False
        ob = (self.resize_img(rgb), large_loc_map, small_loc_map)
        return ob, reward, done, info

    def render(self):
        loc_map, small_map = self.get_loc_map()

        if not self.train_mode:
            rad = self.env.house.robotRad / self.env.house.grid_det
            x = int(loc_map.shape[0] / 2)
            y = int(loc_map.shape[1] / 2)
            cv2.circle(loc_map, (x, y), 1,
                       (255, 0, 255), thickness=-1)
            x = int(small_map.shape[0] / 2)
            y = int(small_map.shape[1] / 2)
            cv2.circle(small_map, (x, y), int(rad) * 2,
                       (255, 0, 255), thickness=-1)

        loc_map = cv2.resize(loc_map,
                             (self.render_width, self.render_height),
                             interpolation=cv2.INTER_CUBIC)
        if not self.train_mode:
            x = int(loc_map.shape[0] / 2)
            y = int(loc_map.shape[1] / 2)
            cv2.circle(loc_map, (x, y), 2, (255, 0, 255), thickness=-1)
        self.env.set_render_mode('rgb')
        rgb = self.env.render()

        img = np.concatenate((rgb, loc_map), axis=1)
        img = img[:, :, ::-1]
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imshow("nav", img)
        cv2.waitKey(40)

    def resize_img(self, img):
        return cv2.resize(img, (self.img_width, self.img_height),
                          interpolation=cv2.INTER_AREA)

    def motion_primitive(self, action):
        # 0: Forward
        # 1: Turn Left
        # 2: Turn Right
        # 3: Strafe Left
        # 4: Strafe Right
        # 5: Backward
        collision_flag = False
        if action == 0:
            if not self.train_mode:
                print('Action: Forward')
            if not self.env.move_forward(self.move_sensitivity):
                if not self.train_mode:
                    print('Cannot move forward, collision!!!')
                collision_flag = True
        elif action == 1:
            if not self.train_mode:
                print('Action: Turn Left')
            self.env.rotate(-self.rot_sensitivity)
        elif action == 2:
            if not self.train_mode:
                print('Action: Turn Right')
            self.env.rotate(self.rot_sensitivity)
        elif action == 3:
            if not self.train_mode:
                print('Action: Strafe Left')
            if not self.env.move_forward(dist_fwd=0,
                                         dist_hor=-self.move_sensitivity):
                if not self.train_mode:
                    print('Cannot strafe left, collision!!!')
                collision_flag = True
        elif action == 4:
            if not self.train_mode:
                print('Action: Strafe Right')
            if not self.env.move_forward(dist_fwd=0,
                                         dist_hor=self.move_sensitivity):
                if not self.train_mode:
                    print('Cannot strafe right, collision!!!')
                collision_flag = True
        elif action == 5:
            if not self.train_mode:
                print('Action: Backward')
            if not self.env.move_forward(-self.move_sensitivity):
                if not self.train_mode:
                    print('Cannot move backward, collision!!!')
                collision_flag = True
        else:
            raise ValueError('unknown action type: [{0:d}]'.format(action))
        return collision_flag

    def move_forward(self, dist_fwd, dist_hor=0):
        """
        Move with `fwd` distance to the front and `hor` distance to the right.
        Both distance are float numbers.
        Ignore collision !!!
        """
        pos = self.env.cam.pos
        pos = pos + self.env.cam.front * dist_fwd
        pos = pos + self.env.cam.right * dist_hor
        self.env.cam.pos.x = pos.x
        self.env.cam.pos.z = pos.z

    def motion_primitive_no_check(self, action):
        # motion primitive without collision checking
        # 0: Forward
        # 1: Turn Left
        # 2: Turn Right
        # 3: Strafe Left
        # 4: Strafe Right
        # 5: Backward
        collision_flag = False
        if action == 0:
            self.move_forward(self.move_sensitivity)
        elif action == 1:
            self.env.rotate(-self.rot_sensitivity)
        elif action == 2:
            self.env.rotate(self.rot_sensitivity)
        elif action == 3:
            self.move_forward(dist_fwd=0,
                              dist_hor=-self.move_sensitivity)
        elif action == 4:
            self.move_forward(dist_fwd=0,
                              dist_hor=self.move_sensitivity)
        elif action == 5:
            self.move_forward(-self.move_sensitivity)
        else:
            raise ValueError('unknown action type: [{0:d}]'.format(action))
        return collision_flag

    def get_obs(self):
        self.env.set_render_mode('rgb')
        rgb = self.env.render()
        self.env.set_render_mode('depth')
        depth = self.env.render()
        infmask = depth[:, :, 1]
        depth = depth[:, :, 0] * (infmask == 0)
        true_depth = depth.astype(np.float32) / 255.0 * 20.0
        extrinsics = self.env.cam.getExtrinsicsNumpy()
        return rgb, true_depth, extrinsics

    def get_seen_area(self, rgb, depth, extrinsics, out_mat, inv_E=True):
        points, points_colors = gen_point_cloud(depth, rgb, extrinsics,
                                                depth_threshold=self.depth_threshold,
                                                inv_E=inv_E)
        grid_locs = np.floor((points[:, [0, 2]] - self.L_min) / self.grid_size).astype(int)
        grids_mat = np.zeros((self.grids_mat.shape[0],
                              self.grids_mat.shape[1]),
                             dtype=np.uint8)

        high_filter_idx = points[:, 1] < HEIGHT_THRESHOLD[1]
        low_filter_idx = points[:, 1] > HEIGHT_THRESHOLD[0]
        obstacle_idx = np.logical_and(high_filter_idx, low_filter_idx)

        self.safe_assign(grids_mat, grid_locs[high_filter_idx, 0],
                         grid_locs[high_filter_idx, 1], 2)
        kernel = np.ones((3, 3), np.uint8)
        grids_mat = cv2.morphologyEx(grids_mat, cv2.MORPH_CLOSE, kernel)

        obs_mat = np.zeros((self.grids_mat.shape[0], self.grids_mat.shape[1]), dtype=np.uint8)
        self.safe_assign(obs_mat, grid_locs[obstacle_idx, 0],
                         grid_locs[obstacle_idx, 1], 1)
        kernel = np.ones((self.ob_dilation_kernel, self.ob_dilation_kernel), np.uint8)
        obs_mat = cv2.morphologyEx(obs_mat, cv2.MORPH_CLOSE, kernel)
        obs_idx = np.where(obs_mat == 1)
        self.safe_assign(grids_mat, obs_idx[0], obs_idx[1], 1)
        out_mat[np.where(grids_mat == 2)] = 2
        out_mat[np.where(grids_mat == 1)] = 1
        seen_area = np.sum(out_mat > 0)
        return seen_area

    def cal_reward(self, rgb, depth, extrinsics, collision_flag):
        filled_grid_num = self.get_seen_area(rgb, depth,
                                             extrinsics,
                                             self.grids_mat,
                                             inv_E=True)
        area_reward = (filled_grid_num - self.seen_area)
        reward = area_reward * self.area_reward_scale
        if collision_flag:
            reward -= self.collision_penalty
        reward -= self.step_penalty
        self.seen_area = filled_grid_num
        raw_reward = {'area': area_reward, 'collision_flag': collision_flag}
        return reward, filled_grid_num, raw_reward

    def get_loc_map(self):
        top_down_map = self.grids_mat.T.copy()

        half_size = max(top_down_map.shape[0],
                        top_down_map.shape[1],
                        self.configs['large_map_range']) * 3
        ego_map = np.ones((half_size * 2, half_size * 2, 3),
                          dtype=np.uint8) * 255
        loc_map = np.zeros((top_down_map.shape[0], top_down_map.shape[1], 3),
                           dtype=np.uint8)
        loc_map[top_down_map == 0] = np.array([255, 255, 255])
        loc_map[top_down_map == 1] = np.array([0, 0, 255])
        loc_map[top_down_map == 2] = np.array([0, 255, 0])
        current_pos, grid_current_pos = self.get_camera_grid_pos()
        x_start = half_size - grid_current_pos[1]
        y_start = half_size - grid_current_pos[0]
        x_end = x_start + top_down_map.shape[0]
        y_end = y_start + top_down_map.shape[1]
        assert x_start >= 0 and y_start >= 0 and \
               x_end <= ego_map.shape[0] and y_end <= ego_map.shape[1]
        ego_map[x_start: x_end, y_start: y_end] = loc_map
        center = (half_size, half_size)
        rot_angle = self.constrain_to_pm_pi(90 + current_pos[2])
        M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        ego_map = cv2.warpAffine(ego_map, M,
                                 (ego_map.shape[1], ego_map.shape[0]),
                                 flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        start = half_size - self.configs['small_map_range']
        end = half_size + self.configs['small_map_range']
        small_ego_map = ego_map[start: end, start:end]

        start = half_size - self.configs['large_map_range']
        end = half_size + self.configs['large_map_range']
        assert start >= 0
        assert end <= ego_map.shape[0]
        large_ego_map = ego_map[start: end, start:end]
        return cv2.resize(large_ego_map, (self.configs['large_map_size'],
                                          self.configs['large_map_size']),
                          interpolation=cv2.INTER_AREA), small_ego_map

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def constrain_to_pm_pi(self, theta):
        # make sure theta is within [-180, 180)
        return (theta + 180) % 360 - 180

    def get_camera_grid_pos(self):
        current_pos = np.array([self.env.cam.pos.x,
                                self.env.cam.pos.z,
                                self.constrain_to_pm_pi(self.env.cam.yaw)])
        grid_pos = np.array(self.env.house.to_grid(current_pos[0], current_pos[1]))
        return current_pos, grid_pos

    def truncated_norm(self, mu, sigma, lower_limit, upper_limit, size):
        if sigma == 0:
            return mu
        lower_limit = (lower_limit - mu) / sigma
        upper_limit = (upper_limit - mu) / sigma
        r = truncnorm(lower_limit, upper_limit, loc=mu, scale=sigma)
        return r.rvs(size)
