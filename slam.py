# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6         # Sets the probability threshold above which a cell is considered occupied.
                                             # This threshold is used to determine whether a cell is considered occupied based on the probability of occupancy.
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX

        x_clipped = np.clip(x, s.xmin, s.xmax)
        y_clipped = np.clip(y, s.ymin, s.ymax)

        x_relative = (x_clipped - s.xmin) / s.resolution
        y_relative = (y_clipped - s.ymin) / s.resolution

        x_grid = np.ceil(x_relative).astype(np.int16)
        y_grid = np.ceil(y_relative).astype(np.int16)

        grid_cell = np.vstack((x_grid, y_grid))
        return grid_cell



class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.init_sensor_model()
        s.Q = Q
        # dynamics noise for the state (x,y,yaw)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)       #s.map is my object

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir, 'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir, 'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))
            #the above takes timestamp 't' as the input and and returns the index of the joint
            #timestep that is closest to the timestep "t" by finding the absolute difference
    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w): #paraphrase
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        num_particles = p.shape[1]  # Number of particles
        cumulative_weights = np.cumsum(w) / np.sum(w)  # Create a normalized cumulative distribution
        random_numbers = (np.arange(num_particles) + np.random.uniform(size=num_particles)) / num_particles

        p_new = np.zeros_like(p)
        w_new = np.ones(num_particles) / num_particles  # Uniform weights for new particles

        # Use roulette wheel method for resampling
        idx_new = 0
        idx_old = 0
        while idx_new < num_particles:
            while cumulative_weights[idx_old] < random_numbers[idx_new]:
                idx_old += 1
            p_new[:, idx_new] = p[:, idx_old]
            idx_new += 1

        return p_new, w_new

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())


    def rays2world(s,rpy, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        for i in range(len(d)):
            if d[i] < s.lidar_dmin:
                d[i] = s.lidar_dmin
            elif d[i] > s.lidar_dmax:
                d[i] = s.lidar_dmax

        # 1. from lidar distances to points in the LiDAR frame
        pt_x = np.cos(angles) * d #element wise multiplication for x
        pt_y =  np.sin(angles) * d #element wise for y
        pt_z = np.zeros(len(d))  #z = 0
        pts = np.vstack((pt_x , pt_y ,pt_z))

        # 2. from LiDAR frame to the body frame
        # 3. from body frame to world frame
        v = np.array([0,0,s.lidar_height])
        T_lh = euler_to_se3(0,0,0,v)
        v = np.zeros(3)
        T_hb = euler_to_se3(0,head_angle, neck_angle , v)
        v = np.array([p[0] , p[1] , s.head_height])
        T_bw = euler_to_se3(rpy[0] , rpy[1], rpy[2], v)

        points = np.vstack((pts, np.ones((1, len(d)))))      # format it to multiply with transformation matrices

        pts_world = T_bw @ T_hb @ T_lh @ points

        xy = pts_world[:2 , :]
        return xy

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        ####d TODO: XXXXXXXXXXX
        #doubt: lidar pose w.r.t what frame and "smart minus and smart plus"
        u1 = s.lidar[t]['xyth']             #pose of the lidar at t
        u2 = s.lidar[t-1]['xyth']           #pose of the lidar at t-1
        control = smart_minus_2d(u1 , u2)
        return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        U = s.get_control(t)    #finding the control taken in
        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], U) #adding the control to all the particles
            noise = np.random.multivariate_normal([0, 0, 0], s.Q)       #adding the noise
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        log_p = obs_logp - slam_t.log_sum_exp(obs_logp)
        w_updated = np.zeros(np.shape(w))
        for i in range(len(w)):
            w_updated[i] = w[i] * np.exp(log_p[i])

        return w_updated/np.sum(w_updated)


    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX

        RPY = s.lidar[t]['rpy'] #finding the Roll pitch yaw angles for a particular timestep t
        neck , head = s.joint['head_angles'][0] , s.joint['head_angles'][1] #row 1 is neck, row 2 is head

        #for initial timestep t = 0 , we only take 1 particle (1 pose) and sets
        #whatver the laser observes initially as my initial environment

        if t == 0:
            #calling s.rays2world
            p = s.p[: , 0] #3 x 1 array
            d = s.lidar[t]['scan']
            i = s.find_joint_t_idx_from_lidar(t)
            head_angle = head[i]
            neck_angle = neck[i]
            angles = s.lidar_angles  #angles of the rays of the Hokuyo

            end_pt = s.rays2world(RPY , p, d, head_angle, neck_angle, angles)
            occupied = s.map.grid_cell_from_xy(end_pt[0, :], end_pt[1, :])  #cell indices occupied in grid as returned by lidar

            #initial_map
            s.map.cells[occupied[0], occupied[1]] = 1                  #s.cells is my binarized map

            #creating path by adding the first particle
            s.path = s.map.grid_cell_from_xy(p[0], p[1])
            s.path.reshape((2,1))


        else:                                                        #implementation when t not equal 0
            log_obs = np.zeros(s.p.shape[1])                         #s.p.shape[1] = no of particles, for keeping track of log(P) of each particle
            for i in range(np.shape(s.p)[1]):
                p = s.p[:, i]
                d = s.lidar[t]['scan']
                index = s.find_joint_t_idx_from_lidar(t)
                head_angle = head[index]
                neck_angle = neck[index]
                angles = s.lidar_angles
                end_pts = s.rays2world(RPY, p, d, head_angle, neck_angle, angles)

                occupied = s.map.grid_cell_from_xy(end_pts[0, :], end_pts[1, :]) #cell indices occupied in grid as returned by lidar
                print("1", occupied.shape)
                #check if the occupied cells as given by our lider match the occupied cells
                #in the binazized map crated from the past observations.

                for j in range(np.shape(occupied)[1]): # count the number of lidar detected cells that are already occupied for each particle
                    log_obs[i] = log_obs[i] +  s.map.cells[occupied[0, j], occupied[1, j]]

            #updating of weights
            s.w = s.update_weights(s.w, log_obs)



            #largest weight particle would be the next pose
            large_w_p = s.p[:, np.argmax(s.w)]
            print("largest",large_w_p)


            ############################
            #adding the particle to path
            grid_cell = s.map.grid_cell_from_xy(large_w_p[0], large_w_p[1])

            # Reshape the grid cell into a column vector with dimensions (2, 1)
            reshaped_grid_cell = np.reshape(grid_cell, (2, 1))

            # Concatenate the reshaped grid cell with the existing path
            extended_path = np.hstack((s.path, reshaped_grid_cell))

            # Update the path attribute of s with the extended path
            s.path = extended_path

            #find readings for the new pose
            d = s.lidar[t]['scan']
            index = s.find_joint_t_idx_from_lidar(t)
            head_angle = head[index]
            neck_angle = neck[index]
            angles = s.lidar_angles
            end_pts = s.rays2world(RPY, large_w_p, d, head_angle, neck_angle, angles)

            #occupied points observed
            occupied = s.map.grid_cell_from_xy(end_pts[0, :], end_pts[1, :])
            print("2", occupied.shape)   #2 x1081 always fix

            #unoccupied = s.get_free_coor(occupied, )
            unoccupied = np.zeros([2,1])
            current = s.map.grid_cell_from_xy(large_w_p[0], large_w_p[1])
            #find all the free cells between the robot estimated position and the cell where laser scan hit something

            for i in range(np.shape(occupied)[1]):
                #line for x co of current pose and all end pts:
                start_x = current[0]
                stop_x = occupied[0,i]
                steps_x = int(np.linalg.norm(occupied[:,i] - current))
                x_id = np.linspace(start_x,stop_x,int(steps_x), dtype=int, endpoint=False)

                #for y
                start_y = current[1]
                stop_y = occupied[1, i]
                steps_y = int(np.linalg.norm(occupied[:, i] - current))
                y_id = np.linspace(start_y,stop_y,int(steps_y),dtype=int, endpoint=False)

                if i == 0:
                    unoccupied[0,0] = x_id[0]
                    unoccupied[1,0] = y_id[0]

                # Extract the x and y coordinates of unoccupied cells excluding the first one
                x_coordinates = np.reshape(x_id[1:], (1, len(x_id) - 1))
                y_coordinates = np.reshape(y_id[1:], (1, len(y_id) - 1))

                # Vertically stack the x and y coordinates to create a matrix of coordinates
                coordinates_matrix = np.vstack((x_coordinates, y_coordinates))

                # Horizontally stack the unoccupied cells with the coordinates matrix to obtain the free cells
                free = np.hstack((unoccupied, coordinates_matrix))

            free = np.unique(free, return_index=False, axis=1)
            print("3",free.shape)
            print(occupied.shape)

            #lod_add map, adding valye if occupied and subtracting if unoccupied
            for i in range(np.shape(occupied)[1]):
                x_idx = int(occupied[0, i])  # Convert to integer
                y_idx = int(occupied[1, i])  # Convert to integer
                s.map.log_odds[x_idx, y_idx] += s.lidar_log_odds_occ

            for i in range(np.shape(free)[1]):
                x_idx = int(free[0, i])  # Convert to integer
                y_idx = int(free[1, i])  # Convert to integer
                s.map.log_odds[x_idx, y_idx] += s.lidar_log_odds_free

            #constraint between min and max
            s.map.log_odds[s.map.log_odds < -s.map.log_odds_max] = -s.map.log_odds_max
            s.map.log_odds[s.map.log_odds > s.map.log_odds_max] = s.map.log_odds_max

            # set map.cells occupancy grid to 1 if log_odds at that point is larger than threshold

            s.map.cells = np.zeros_like(s.map.cells)
            s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
            s.map.cells[s.map.log_odds <= s.lidar_log_odds_free] = 0

            s.resample_particles()


    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6         # Sets the probability threshold above which a cell is considered occupied.
                                             # This threshold is used to determine whether a cell is considered occupied based on the probability of occupancy.
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX

        x_clipped = np.clip(x, s.xmin, s.xmax)
        y_clipped = np.clip(y, s.ymin, s.ymax)

        x_relative = (x_clipped - s.xmin) / s.resolution
        y_relative = (y_clipped - s.ymin) / s.resolution

        x_grid = np.ceil(x_relative).astype(np.int16)
        y_grid = np.ceil(y_relative).astype(np.int16)

        grid_cell = np.vstack((x_grid, y_grid))
        return grid_cell



class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.init_sensor_model()
        s.Q = Q
        # dynamics noise for the state (x,y,yaw)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)       #s.map is my object

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir, 'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir, 'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))
            #the above takes timestamp 't' as the input and and returns the index of the joint
            #timestep that is closest to the timestep "t" by finding the absolute difference
    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w): #paraphrase
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        num_particles = p.shape[1]  # Number of particles
        cumulative_weights = np.cumsum(w) / np.sum(w)  # Create a normalized cumulative distribution
        random_numbers = (np.arange(num_particles) + np.random.uniform(size=num_particles)) / num_particles

        p_new = np.zeros_like(p)
        w_new = np.ones(num_particles) / num_particles  # Uniform weights for new particles

        # Use roulette wheel method for resampling
        idx_new = 0
        idx_old = 0
        while idx_new < num_particles:
            while cumulative_weights[idx_old] < random_numbers[idx_new]:
                idx_old += 1
            p_new[:, idx_new] = p[:, idx_old]
            idx_new += 1

        return p_new, w_new

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())


    def rays2world(s,rpy, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        for i in range(len(d)):
            if d[i] < s.lidar_dmin:
                d[i] = s.lidar_dmin
            elif d[i] > s.lidar_dmax:
                d[i] = s.lidar_dmax

        # 1. from lidar distances to points in the LiDAR frame
        pt_x = np.cos(angles) * d #element wise multiplication for x
        pt_y =  np.sin(angles) * d #element wise for y
        pt_z = np.zeros(len(d))  #z = 0
        pts = np.vstack((pt_x , pt_y ,pt_z))

        # 2. from LiDAR frame to the body frame
        # 3. from body frame to world frame
        v = np.array([0,0,s.lidar_height])
        T_lh = euler_to_se3(0,0,0,v)
        v = np.zeros(3)
        T_hb = euler_to_se3(0,head_angle, neck_angle , v)
        v = np.array([p[0] , p[1] , s.head_height])
        T_bw = euler_to_se3(rpy[0] , rpy[1], rpy[2], v)

        points = np.vstack((pts, np.ones((1, len(d)))))      # format it to multiply with transformation matrices

        pts_world = T_bw @ T_hb @ T_lh @ points

        xy = pts_world[:2 , :]
        return xy

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        ####d TODO: XXXXXXXXXXX
        #doubt: lidar pose w.r.t what frame and "smart minus and smart plus"
        u1 = s.lidar[t]['xyth']             #pose of the lidar at t
        u2 = s.lidar[t-1]['xyth']           #pose of the lidar at t-1
        control = smart_minus_2d(u1 , u2)
        return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        U = s.get_control(t)    #finding the control taken in
        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], U) #adding the control to all the particles
            noise = np.random.multivariate_normal([0, 0, 0], s.Q)       #adding the noise
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        log_p = obs_logp - slam_t.log_sum_exp(obs_logp)
        w_updated = np.zeros(np.shape(w))
        for i in range(len(w)):
            w_updated[i] = w[i] * np.exp(log_p[i])

        return w_updated/np.sum(w_updated)


    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX

        RPY = s.lidar[t]['rpy'] #finding the Roll pitch yaw angles for a particular timestep t
        neck , head = s.joint['head_angles'][0] , s.joint['head_angles'][1] #row 1 is neck, row 2 is head

        #for initial timestep t = 0 , we only take 1 particle (1 pose) and sets
        #whatver the laser observes initially as my initial environment

        if t == 0:
            #calling s.rays2world
            p = s.p[: , 0] #3 x 1 array
            d = s.lidar[t]['scan']
            i = s.find_joint_t_idx_from_lidar(t)
            head_angle = head[i]
            neck_angle = neck[i]
            angles = s.lidar_angles  #angles of the rays of the Hokuyo

            end_pt = s.rays2world(RPY , p, d, head_angle, neck_angle, angles)
            occupied = s.map.grid_cell_from_xy(end_pt[0, :], end_pt[1, :])  #cell indices occupied in grid as returned by lidar

            #initial_map
            s.map.cells[occupied[0], occupied[1]] = 1                  #s.cells is my binarized map

            #creating path by adding the first particle
            s.path = s.map.grid_cell_from_xy(p[0], p[1])
            s.path.reshape((2,1))


        else:                                                        #implementation when t not equal 0
            log_obs = np.zeros(s.p.shape[1])                         #s.p.shape[1] = no of particles, for keeping track of log(P) of each particle
            for i in range(np.shape(s.p)[1]):
                p = s.p[:, i]
                d = s.lidar[t]['scan']
                index = s.find_joint_t_idx_from_lidar(t)
                head_angle = head[index]
                neck_angle = neck[index]
                angles = s.lidar_angles
                end_pts = s.rays2world(RPY, p, d, head_angle, neck_angle, angles)

                occupied = s.map.grid_cell_from_xy(end_pts[0, :], end_pts[1, :]) #cell indices occupied in grid as returned by lidar
                print("1", occupied.shape)
                #check if the occupied cells as given by our lider match the occupied cells
                #in the binazized map crated from the past observations.

                for j in range(np.shape(occupied)[1]): # count the number of lidar detected cells that are already occupied for each particle
                    log_obs[i] = log_obs[i] +  s.map.cells[occupied[0, j], occupied[1, j]]

            #updating of weights
            s.w = s.update_weights(s.w, log_obs)



            #largest weight particle would be the next pose
            large_w_p = s.p[:, np.argmax(s.w)]
            print("largest",large_w_p)


            ############################
            #adding the particle to path
            grid_cell = s.map.grid_cell_from_xy(large_w_p[0], large_w_p[1])

            # Reshape the grid cell into a column vector with dimensions (2, 1)
            reshaped_grid_cell = np.reshape(grid_cell, (2, 1))

            # Concatenate the reshaped grid cell with the existing path
            extended_path = np.hstack((s.path, reshaped_grid_cell))

            # Update the path attribute of s with the extended path
            s.path = extended_path

            #find readings for the new pose
            d = s.lidar[t]['scan']
            index = s.find_joint_t_idx_from_lidar(t)
            head_angle = head[index]
            neck_angle = neck[index]
            angles = s.lidar_angles
            end_pts = s.rays2world(RPY, large_w_p, d, head_angle, neck_angle, angles)

            #occupied points observed
            occupied = s.map.grid_cell_from_xy(end_pts[0, :], end_pts[1, :])
            print("2", occupied.shape)   #2 x1081 always fix

            #unoccupied = s.get_free_coor(occupied, )
            unoccupied = np.zeros([2,1])
            current = s.map.grid_cell_from_xy(large_w_p[0], large_w_p[1])
            #find all the free cells between the robot estimated position and the cell where laser scan hit something

            for i in range(np.shape(occupied)[1]):
                #line for x co of current pose and all end pts:
                start_x = current[0]
                stop_x = occupied[0,i]
                steps_x = int(np.linalg.norm(occupied[:,i] - current))
                x_id = np.linspace(start_x,stop_x,int(steps_x), dtype=int, endpoint=False)

                #for y
                start_y = current[1]
                stop_y = occupied[1, i]
                steps_y = int(np.linalg.norm(occupied[:, i] - current))
                y_id = np.linspace(start_y,stop_y,int(steps_y),dtype=int, endpoint=False)

                if i == 0:
                    unoccupied[0,0] = x_id[0]
                    unoccupied[1,0] = y_id[0]

                # Extract the x and y coordinates of unoccupied cells excluding the first one
                x_coordinates = np.reshape(x_id[1:], (1, len(x_id) - 1))
                y_coordinates = np.reshape(y_id[1:], (1, len(y_id) - 1))

                # Vertically stack the x and y coordinates to create a matrix of coordinates
                coordinates_matrix = np.vstack((x_coordinates, y_coordinates))

                # Horizontally stack the unoccupied cells with the coordinates matrix to obtain the free cells
                free = np.hstack((unoccupied, coordinates_matrix))

            free = np.unique(free, return_index=False, axis=1)
            print("3",free.shape)
            print(occupied.shape)

            #lod_add map, adding valye if occupied and subtracting if unoccupied
            for i in range(np.shape(occupied)[1]):
                x_idx = int(occupied[0, i])  # Convert to integer
                y_idx = int(occupied[1, i])  # Convert to integer
                s.map.log_odds[x_idx, y_idx] += s.lidar_log_odds_occ

            for i in range(np.shape(free)[1]):
                x_idx = int(free[0, i])  # Convert to integer
                y_idx = int(free[1, i])  # Convert to integer
                s.map.log_odds[x_idx, y_idx] += s.lidar_log_odds_free

            #constraint between min and max
            s.map.log_odds[s.map.log_odds < -s.map.log_odds_max] = -s.map.log_odds_max
            s.map.log_odds[s.map.log_odds > s.map.log_odds_max] = s.map.log_odds_max

            # set map.cells occupancy grid to 1 if log_odds at that point is larger than threshold

            s.map.cells = np.zeros_like(s.map.cells)
            s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
            s.map.cells[s.map.log_odds <= s.lidar_log_odds_free] = 0

            s.resample_particles()


    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')