import time
from djitellopy import Tello
import numpy as np

# Connect to Tello drone
tello = Tello()
tello.connect()

# Define destination coordinates
dest_x, dest_y, dest_z = 100, 100, 100

# Set initial velocity
vx, vy, vz = 0, 0, 0

# Set gain values for position and velocity
Kp_pos, Ki_pos, Kd_pos = 0.1, 0.01, 0.1
Kp_vel, Ki_vel, Kd_vel = 0.3, 0.01, 0.3

# Set initial error and integral values
error_sum_pos = np.array([0, 0, 0])
error_sum_vel = np.array([0, 0, 0])
prev_error_pos = np.array([0, 0, 0])
prev_error_vel = np.array([0, 0, 0])

# Define time step and duration of flight
dt = 0.1
duration = 10

# Start timer
start_time = time.time()

# Main loop for control
while (time.time() - start_time) < duration:
    
    # Get current position and orientation from motion capture system
    pos_x, pos_y, pos_z = # code to get position from motion capture system
    ori_x, ori_y, ori_z = # code to get orientation from motion capture system
    
    # Calculate error between current position and destination
    error_pos = np.array([dest_x - pos_x, dest_y - pos_y, dest_z - pos_z])
    
    # Calculate error between current velocity and desired velocity
    error_vel = np.array([vx - ori_x, vy - ori_y, vz - ori_z])
    
    # Calculate derivative of position error
    error_deriv_pos = (error_pos - prev_error_pos) / dt
    
    # Calculate derivative of velocity error
    error_deriv_vel = (error_vel - prev_error_vel) / dt
    
    # Update integral of position error
    error_sum_pos += error_pos * dt
    
    # Update integral of velocity error
    error_sum_vel += error_vel * dt
    
    # Calculate PID control outputs for position and velocity
    pos_output = Kp_pos * error_pos + Ki_pos * error_sum_pos + Kd_pos * error_deriv_pos
    vel_output = Kp_vel * error_vel + Ki_vel * error_sum_vel + Kd_vel * error_deriv_vel
    
    # Set drone velocities based on PID outputs
    tello.send_rc_control(int(vel_output[1]), int(-vel_output[0]), int(-vel_output[2]), 0)
    
    # Update previous errors for next iteration
    prev_error_pos = error_pos
    prev_error_vel = error_vel
    
    # Show heatmap for 1 second
    plt.imshow(np.random.rand(10, 10), cmap='hot', interpolation='nearest')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

# Land the drone
tello.land()