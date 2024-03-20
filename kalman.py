import numpy as np

# Constants for the Kalman filter

thymio_speed_to_ms = 0.2 / 1000 # Conversion factor from thymio speed to m/s, measured experimentally, our Thymio is slow :(
std_speed = 12.306044894776846 * thymio_speed_to_ms # Convert the std to the right unit
q_wheel = std_speed/2 # variance on speed state
r_wheel = std_speed/2 # variance on speed measurement 
# Definition of H^wheel ( the jacobian of h^wheel ):
Zeros_matrix = np.zeros((2, 3))
Eye_matrix = np.eye(2)
H_wheel = np.hstack((Zeros_matrix, Eye_matrix))
# Definition of H^camera ( the jacobian of h^camera ):
Zeros_matrix = np.zeros((3, 2))
Eye_matrix = np.eye(3)
H_camera = np.hstack((Eye_matrix, Zeros_matrix))

R_camera = np.diag((15e-5, 15e-5, 14e-4)) # Camera measurement noise
R_wheel = np.diag((r_wheel,r_wheel)) # Wheel measurement noise
Q = np.diag((5e-3,5e-3,5e-2,q_wheel,q_wheel)) # Process noise


def Predict(x_kk, P_kk, time_step):
    theta = x_kk[2]
    l = 90/2000 # half of the wheelbase [m]
    v_rl = x_kk[3] + x_kk[4]  #v_r + v_l
    x_dot = 1/2 * np.array([np.cos(theta) * v_rl,
                            np.sin(theta) * v_rl,
                            (-x_kk[3] + x_kk[4])/l,
                            0,
                            0])

    x_kk1 = x_kk + x_dot * time_step # Forward euler scheme 
    x_kk1[2] = Wrap2Pi(x_kk1[2])

    #compute F ( the Jacobian of f)
    F = 1/2 * np.array([[0, 0, -np.sin(theta) * v_rl, np.cos(theta), np.cos(theta)],
                        [0, 0, np.cos(theta) * v_rl, np.sin(theta), np.sin(theta)],
                        [0, 0, 0, -1/l, 1/l],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
    
    P_kk1 = P_kk + time_step * (F @ P_kk + P_kk @ np.transpose(F) + Q)
    return x_kk1, P_kk1

def CameraUpdate(measurement, x_kk1, P_kk1):
    h = x_kk1[:3]
    #Kalman gain
    K = P_kk1 @ np.transpose(H_camera) @ np.linalg.inv(H_camera @ P_kk1 @ np.transpose(H_camera) + R_camera)

    innovation = measurement - h
    innovation[2] = Wrap2Pi(innovation[2])

    x_kk = x_kk1 + K @ innovation
    x_kk[2] = Wrap2Pi(x_kk[2])

    P_kk = (np.eye(5) - K @ H_camera) @ P_kk1 
    return x_kk, P_kk

def WheelUpdate(measurement, x_kk1, P_kk1):
    h = x_kk1[3:]
    #Kalman gain
    K = P_kk1 @ np.transpose(H_wheel) @ np.linalg.inv(H_wheel @ P_kk1 @ np.transpose(H_wheel) + R_wheel)

    innovation = measurement - h

    x_kk = x_kk1 + K @ innovation
    x_kk[2] = Wrap2Pi(x_kk[2])

    P_kk = (np.eye(5) - K @ H_wheel) @ P_kk1 
    return x_kk, P_kk

def Wrap2Pi(angle):
    if angle >= np.pi:
        return angle % (2*np.pi) - 2*np.pi
    elif angle < -np.pi:
        return angle % (2*np.pi)
    else:
        return angle