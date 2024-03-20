import numpy as np

# Constants for the local navigation
WALL_THRESHOLD = 1500
OBST_SPEED_GAIN = [3, 2, -4, -3, -4] #weights for proximity sensors
AVOIDANCE_GAIN = 1 / 90

# Controller for the speed
MAX_SPEED = 300
MIN_SPEED = 100
DECREASING_FACTOR = 10
DISTANCE_THRESHOLD = 0.1

# Controller for the direction
SPEED_GAIN_P = 100.0
SPEED_GAIN_D = 30.0

# Global variables for the PID controller
last_delta_angle = 0.0

def motors(l_speed=500, r_speed=500):
    """
    Sets the motor speeds of the Thymio 
    param l_speed: left motor speed
    param r_speed: right motor speed
    """
    return {
        "motor.left.target": [l_speed],
        "motor.right.target": [r_speed],
    }
    
def controller_pd(prox_horizontal, dist, deltaAngle, state):
    
    if state == 0: 
        #changes to state 1 if one of the sensors detects a nearby object 
        if any([x>WALL_THRESHOLD for x in prox_horizontal[:-2]]):
            state = 1
                
    elif state == 1:
        #goes to state 0 if all the sensors do not detect any nearby objects
        if all([x<WALL_THRESHOLD for x in prox_horizontal[:-2]]):
            state = 0
    
    # Controller for the direction
    proportional_speed = SPEED_GAIN_P * deltaAngle
    
    global last_delta_angle
    derivative_speed = SPEED_GAIN_D * (deltaAngle - last_delta_angle)
    last_delta_angle = deltaAngle
    
    speed_orientation = proportional_speed + derivative_speed

    # Controller for the distance: sigmoid function
    speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) / (1 + np.exp(-DECREASING_FACTOR * (DISTANCE_THRESHOLD - dist)))

    # Set the motor speeds according to the direction and the distance
    motor_left_target = int(speed + speed_orientation)
    motor_right_target = int(speed - speed_orientation)     
               
    # Calculates the robot's speed in local avoidance as a function of the values captured by the proximity sensors
    if state == 1:
        for i in range(5):
            motor_left_target += int(prox_horizontal[i] * OBST_SPEED_GAIN[i] * AVOIDANCE_GAIN)
            motor_right_target += int(prox_horizontal[i] * OBST_SPEED_GAIN[4-i] * AVOIDANCE_GAIN)
                
    return (motor_left_target, motor_right_target)