SLAM for Mobile Robots using a Particle Filter:

o	Leveraged Particle Filtering alongside iterative weight resampling, incorporating Lidar and odometry data to forecast the robot's subsequent state.

o	Utilized a depth camera for environmental data capture, followed by log-based graph mapping.


Using the Dynamics step, we Obtain the trajectory created using the Odometry sensors and the Particle Filter. The important thing to appreciate here is that the data observed from the sensors are susceptible to noise and are highly inaccurate whereas the Trajectories that were created using the particle filter filter out the noise to give a good approximation of the optimal trajectory to be followed. The graphs that were created below are attached here:

![SLAM using Particle Filter](https://github.com/hardikshukla7/SLAM-using-Particle-Filter/blob/main/slam_1.png?raw=true)

In our code, we map our environment using Logistic probability. The Lidar camera finds the obstacles in the Robot’s
Environment and gives us the cells in the grid that are potentially or likely to be occupied. By comparing this data with
the previous estimate of our environment (at the previous pose), we update our logarithmic graph, and upon multiple
observations, if the logarithmic value corresponding to a particular cell increases a threshold value, we tell that grid point
in the world to be occupied. Similarly, for unoccupied cells, we shoot out a line segment from our current pose to all the
endpoints found by the lidar, and all the grid cells between these two positions are observed to be unoccupied. We iteratively
compare this data with the previous observed data and decrease the logarithmic value corresponding to that particular cell
which when less than a minimum threshold value, tells us that the cell is unoccupied in the world

![SLAM using Particle Filter - Example 2](https://github.com/hardikshukla7/SLAM-using-Particle-Filter/blob/main/slam_2.png?raw=true)

