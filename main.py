# Pratik Chaudhari (pratikac@seas.upenn.edu)

import click, tqdm, random
import matplotlib.pyplot as plt

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))             #initialized object
    slam.read_data(src_dir, idx, split)         #calling function

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    odo_data = xyth[:,:2]
    print("odo_shape",odo_data.shape)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    pf_data = ps.T
    pf_data = pf_data[:, :2]
    print("pf_data",pf_data.shape)
    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)          #initialized object called slam
    slam.read_data(src_dir, idx, split)     #calling function the read data
    #src_dir = location of data directory
    # idx = idx in the joint timestamp array such that the timestamp at that idx is t

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def odometry_data(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    slam = slam_t(Q=1e-8 * np.eye(3))  # initialized object
    slam.read_data(src_dir, idx, split)  # calling function
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1], p['xyth'][2]])
    xyth = np.array(xyth)

    odo_data = xyth[:, :2]
    print("odo_shape", odo_data.shape)

    return odo_data
def run_slam(src_dir, log_dir, idx, split):         #idx represnts the map I want to run
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    # 2e-3, 2e-3, 4e-3
    # important to set a good Q noise matrix so the algorithm is spreading out good particles every step
    slam = slam_t(resolution=0.05, Q=np.diag([2e-3, 2e-3, 3e-2]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)

    # raise NotImplementedError
    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    # initialize the occupancy grid using one particle and calling the observation_step
    # function
    #### TODO: XXXXXXXXXXX
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[slam.find_joint_t_idx_from_lidar(t0)]['rpy'][2]
    slam.init_particles(n=1, p=xyth.reshape((3, 1)), w=np.array([1]))
    slam.observation_step(t=0)

    # reinitialize with desired amount of particles, assign initial particle positions
    n = 100
    w = np.ones(n) / float(n)
    p = np.zeros((3, n))
    # p = np.random.normal(size=(3, n))
    for i in range(n):
        p[0, i] = np.random.randint(-20, 21)
        p[1, i] = np.random.randint(-20, 21) / 10
        p[2, i] = np.random.randint(-np.pi/3, np.pi/3)
    slam.init_particles(n, p, w)

    # create a moving plot showing SLAM process
    plt.figure()
    plt.ion()
    plt.xlim([0, slam.map.szx])
    plt.ylim([0, slam.map.szy])

    graph, = plt.plot(np.array([0]), 'o', markersize=1, color='black')
    position, = plt.plot(np.array([0]), np.array([0]), 'x', markersize=1, color='green')

    odo_data = odometry_data(src_dir, log_dir, idx, split, t0=0, draw_fig=False)
    odo_x = odo_data[:, 0]
    odo_y = odo_data[:, 1]

    odomx , odomy = slam.map.grid_cell_from_xy(odo_x,odo_y)
    odo_plot, = plt.plot(odomx, odomy, '.r', markersize=1)  # Assuming green dots for visualization

    step_size = 20

    # slam, save data to be plotted later
    #### TODO: XXXXXXXXXXX
    for time in range(step_size, T, step_size):
        # SLAM
        slam.dynamics_step(time)
        slam.observation_step(time)
        print(time)

        x, y = np.where(slam.map.cells == 1)
        graph.set_data(slam.map.szx - x,y)
        x = slam.path[0, :]
        y = slam.path[1, :]
        position.set_data(800 - x,y)

        plt.draw()
        plt.pause(0.05)

    plt.savefig("logs/plot%d.png" % (0))

    plt.figure(2)
    plt.spy(slam.map.cells)
    plt.show()



# @click.command()
# @click.option('--src_dir', default='./', help='data directory', type=str)
# @click.option('--log_dir', default='logs', help='directory to save logs', type=str)
# @click.option('--idx', default='0', help='dataset number', type=int)
# @click.option('--split', default='train', help='train/test split', type=str)
# @click.option('--mode', default='slam',
#               help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main('./', 'logs', 3, 'train', 'slam')