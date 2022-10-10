"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:
        12/17/2018 - RWB
        1/14/2019 - RWB
"""

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
various tools to be used in mavPySim
"""

def Quaternion2Euler(quaternion):
    """
    converts a quaternion attitude to an euler angle attitude
    :param quaternion: the quaternion to be converted to euler angles in a np.matrix
    :return: the euler angle equivalent (phi, theta, psi) in a np.array
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
    theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)

    return phi, theta, psi

def Euler2Quaternion(phi, theta, psi):
    """
    Converts an euler angle attitude to a quaternian attitude
    :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
    :return: Quaternian attitude in np.array(e0, e1, e2, e3)
    """

    e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.array([[e0],[e1],[e2],[e3]])

def Euler2Rotation(phi, theta, psi):
    """
    Converts euler angles to rotation matrix (R_b^i)
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    #R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    R = R_yaw @ R_pitch @ R_roll

    # rotation is body to inertial frame
    # R = np.array([[c_theta*c_psi, s_phi*s_theta*c_psi-c_phi*s_psi, c_phi*s_theta*c_psi+s_phi*s_psi],
    #               [c_theta*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, c_phi*s_theta*s_psi-s_phi*c_psi],
    #               [-s_theta, s_phi*c_theta, c_phi*c_theta]])

    return R

def Quaternion2Rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
    R = R/linalg.det(R)

    return R

def Rotation2Quaternion(R):
    """
    converts a rotation matrix to a unit quaternion
    """
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    tmp=r11+r22+r33
    if tmp>0:
        e0 = 0.5*np.sqrt(1+tmp)
    else:
        e0 = 0.5*np.sqrt(((r12-r21)**2+(r13-r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=r11-r22-r33
    if tmp>0:
        e1 = 0.5*np.sqrt(1+tmp)
    else:
        e1 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=-r11+r22-r33
    if tmp>0:
        e2 = 0.5*np.sqrt(1+tmp)
    else:
        e2 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    tmp=-r11+-22+r33
    if tmp>0:
        e3 = 0.5*np.sqrt(1+tmp)
    else:
        e3 = 0.5*np.sqrt(((r12-r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    return np.array([[e0], [e1], [e2], [e3]])

def hat(omega):
    """
    vector to skew symmetric matrix associated with cross product
    """
    a = omega.item(0)
    b = omega.item(1)
    c = omega.item(2)

    omega_hat = np.array([[0, -c, b],
                          [c, 0, -a],
                          [-b, a, 0]])
    return omega_hat
# load message types
#from tools.rotations import Quaternion2Euler, Quaternion2Rotation

# Input Values
Ts = 0.01  # Time Step
mass = 10
l = 1
w = 1
h = 1

# Initial State Variables of MAV
north0 = 0
east0 = 0
down0 = -100  # If the k-axis points to the ground then shouldn't this value be negative for it to make sense with the coordinate frame.
u0 = 0
v0 = 0
w0 = 0
e0 = 1
e1 = 0
e2 = 0
e3 = 0
p0 = 0
q0 = 0
r0 = 0

# Moment of Inertia Matrix
J = np.array([
    [1 / 12 * mass * (h ** 2 + l ** 2), 0, 0],
    [0, 1 / 12 * mass * (w ** 2 + h ** 2), 0],
    [0, 0, 1 / 12 * mass * (w ** 2 + l ** 2)]
])

Jx = J[0][0]
Jyx = J[0][1]
Jzx = J[0][2]
Jxy = J[1][0]
Jy = J[1][1]
Jzy = J[1][2]
Jxz = J[2][0]
Jyz = J[2][1]
Jz = J[2][2]

# Rotational Inertia Matrix
gamma = Jx * Jz - Jxz ** 2
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + Jxz * 2) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + Jxz ** 2) / gamma
gamma8 = Jx / gamma


class MsgState:
    def __init__(self):
        self.north = 0.  # inertial north position in meters
        self.east = 0.  # inertial east position in meters
        self.altitude = 0.  # inertial altitude in meters
        self.phi = 0.  # roll angle in radians
        self.theta = 0.  # pitch angle in radians
        self.psi = 0.  # yaw angle in radians
        self.Va = 0.  # airspeed in meters/sec
        self.alpha = 0.  # angle of attack in radians
        self.beta = 0.  # sideslip angle in radians
        self.p = 0.  # roll rate in radians/sec
        self.q = 0.  # pitch rate in radians/sec
        self.r = 0.  # yaw rate in radians/sec
        self.Vg = 0.  # groundspeed in meters/sec
        self.gamma = 0.  # flight path angle in radians
        self.chi = 0.  # course angle in radians
        self.wn = 0.  # inertial windspeed in north direction in meters/sec
        self.we = 0.  # inertial windspeed in east direction in meters/sec
        self.bx = 0.  # gyro bias along roll axis in radians/sec
        self.by = 0.  # gyro bias along pitch axis in radians/sec
        self.bz = 0.  # gyro bias along yaw axis in radians/sec


class MavDynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array([[north0],  # North position in inertial frame
                                [east0],  # East position in inertial frame
                                [down0],  # Down position in inertial frame
                                [u0],  # Velocity along i-axis in body frame
                                [v0],  # Velocity along j-axis in body frame
                                [w0],  # Velocity along k-axis in body frame
                                [e0],  # Real/Scalar Component of Quaternion
                                [e1],  # i-component of Quaternion
                                [e2],  # j-component of Quaternion
                                [e3],  # k-component of Quaternion
                                [p0],  # Angular velocity around i-axis in body frame ROLL RATE
                                [q0],  # Angular velocity around j-axis in body frame PITCH RATE
                                [r0]])  # Angular velocity around k-axis in body frame YAW RATE
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics.
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6][0] = self._state.item(6) / normE
        self._state[7][0] = self._state.item(7) / normE
        self._state[8][0] = self._state.item(8) / normE
        self._state[9][0] = self._state.item(9) / normE

        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        north = state.item(0)
        east = state.item(1)
        down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)  # Force in i-axis direction in body frame
        fy = forces_moments.item(1)  # Force in j-axis direction in body frame
        fz = forces_moments.item(2)  # Force in k-axis direction in body frame
        l = forces_moments.item(3)  # Moment about i-axis in body frame
        m = forces_moments.item(4)  # Moment about j-axis in body frame
        n = forces_moments.item(5)  # Moment about k-axis in body frame

        # position kinematics
        north_dot = u * (e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2) + v * (2 * (e1 * e2 - e3 * e0)) + w * (
                    2 * (e1 * e3 + e2 * e0))
        east_dot = u * (2 * (e1 * e2 + e3 * e0)) + v * (e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2) + w * (
                    2 * (e2 * e3 - e1 * e0))
        down_dot = u * (2 * (e1 * e3 - e2 * e0)) + v * (2 * (e2 * e3 + e1 * e0)) + w * (
                    e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2)
        # position dynamics
        u_dot = r * v - q * w + fx / (mass)
        v_dot = p * w - r * u + fy / (mass)
        w_dot = q * u - p * v + fz / (mass)

        # rotational kinematics
        e0_dot = 0.5 * (0 * e0 - p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 - 0 * e1 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + 0 * e2 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2 + 0 * e3)

        # rotational dynamics
        p_dot = gamma1 * p * q - gamma2 * q * r + gamma3 * l + gamma4 * n
        q_dot = gamma5 * p * r - gamma6 * (p ** 2 - r ** 2) + (1 / Jy) * m
        r_dot = gamma7 * p * q - gamma1 * q * r + gamma4 * l + gamma8 * n

        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)


# Initializing MAV Dynamics
MAV = MavDynamics(Ts)
MAV._state = np.array([
    [north0],
    [east0],
    [down0],
    [u0],
    [v0],
    [w0],
    [e0],
    [e1],
    [e2],
    [e3],
    [p0],
    [q0],
    [r0]
])

fx = 4
fy = 4
fz = 9.81 * mass
l = 3
m = 6
n = 1

forces_moments = np.array([
    [fx],
    [fy],
    [fz],
    [l],
    [m],
    [n]
])

# MAV.update(forces_moments)

states_history = [down0]
states_history2 = [north0]
states_history3 = [east0]
t = 0
time_history = [t]
for i in range(0, 1000, 1):
    MAV._state = MAV._state + MAV._derivatives(MAV._state, forces_moments) * Ts
    if MAV._state[2][0] > 0:
        break
    states_history.append(MAV._state[2][0])
    #if MAV._state[0][0] < 0:
        #break
    states_history2.append(MAV._state[0][0])
    states_history3.append(MAV._state[1][0])
    t = t + Ts
    time_history.append(t)


fig = plt.figure()
ax = plt.axes(projection = '3d')
z = states_history
x = states_history3
y = states_history2

ax.plot3D(x, y, z, 'green')
ax.set_xlabel('east0')
ax.set_zlabel('down0')
ax.set_ylabel('north0')
plt.show()
