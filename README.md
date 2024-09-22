SLAM for Mobile Robots using a Particle Filter				
o	Leveraged Particle Filtering alongside iterative weight resampling, incorporating Lidar and odometry data to forecast the robot's subsequent state.
o	Utilized a depth camera for environmental data capture, followed by log-based graph mapping.


Using the Dynamics step, we Obtain the trajectory created using the Odometry sensors and the Particle Filter. The important thing to appreciate here is that the data observed from the sensors are susceptible to noise and are highly inaccurate whereas the Trajectories that were created using the particle filter filter out the noise to give a good approximation of the optimal trajectory to be followed. The graphs that were created below are attached here:

![SLAM using Particle Filter](https://github.com/hardikshukla7/SLAM-using-Particle-Filter/blob/main/slam_1.png?raw=true)

![SLAM using Particle Filter - Example 2](https://github.com/hardikshukla7/SLAM-using-Particle-Filter/blob/main/slam_2.png?raw=true)

