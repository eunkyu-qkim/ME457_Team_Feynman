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

# load message types
from tools.rotations import Quaternion2Euler, Quaternion2Rotation

#Input Values
Ts = 0.01 #Time Step
mass = 10
l = 1
w = 1
h = 1

#Initial State Variables of MAV
north0 = 0
east0 = 0
down0 = 100 #If the k-axis points to the ground then shouldn't this value be negative for it to make sense with the coordinate frame.
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

#Moment of Inertia Matrix
J = np.array([
    [1/12*mass*(h**2+l**2), 0,  0],
    [0, 1/12*mass*(w**2+h**2), 0],
    [0, 0, 1/12*mass*(w**2+l**2)]
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

#Rotational Inertia Matrix
gamma = Jx*Jz - Jxz**2
gamma1 = (Jxz*(Jx-Jy+Jz))/gamma
gamma2 = (Jz*(Jz-Jy)+Jxz*2)/gamma
gamma3 = Jz/gamma
gamma4 = Jxz/gamma
gamma5 = (Jz-Jx)/Jy
gamma6 = Jxz/Jy
gamma7 = ((Jx - Jy)*Jx + Jxz**2)/gamma
gamma8 = Jx/gamma


class MsgState:
    def __init__(self):
        self.north = 0.      # inertial north position in meters
        self.east = 0.      # inertial east position in meters
        self.altitude = 0.       # inertial altitude in meters
        self.phi = 0.     # roll angle in radians
        self.theta = 0.   # pitch angle in radians
        self.psi = 0.     # yaw angle in radians
        self.Va = 0.      # airspeed in meters/sec
        self.alpha = 0.   # angle of attack in radians
        self.beta = 0.    # sideslip angle in radians
        self.p = 0.       # roll rate in radians/sec
        self.q = 0.       # pitch rate in radians/sec
        self.r = 0.       # yaw rate in radians/sec
        self.Vg = 0.      # groundspeed in meters/sec
        self.gamma = 0.   # flight path angle in radians
        self.chi = 0.     # course angle in radians
        self.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.we = 0.      # inertial windspeed in east direction in meters/sec
        self.bx = 0.      # gyro bias along roll axis in radians/sec
        self.by = 0.      # gyro bias along pitch axis in radians/sec
        self.bz = 0.      # gyro bias along yaw axis in radians/sec

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
        fx = forces_moments.item(0) #Force in i-axis direction in body frame
        fy = forces_moments.item(1) #Force in j-axis direction in body frame
        fz = forces_moments.item(2) #Force in k-axis direction in body frame
        l = forces_moments.item(3) #Moment about i-axis in body frame
        m = forces_moments.item(4) #Moment about j-axis in body frame
        n = forces_moments.item(5) #Moment about k-axis in body frame

        # position kinematics
        north_dot= u*(e1**2+e0**2-e2**2-e3**2) + v*(2*(e1*e2-e3*e0)) + w*(2*(e1*e3+e2*e0))
        east_dot = u*(2*(e1*e2+e3*e0))+ v*(e2**2+e0**2-e1**2-e3**2) + w*(2*(e2*e3-e1*e0))
        down_dot = u*(2*(e1*e3-e2*e0)) + v*(2*(e2*e3+e1*e0)) + w*(e3**2+e0**2-e1**2-e2**2)
        # position dynamics
        u_dot = r*v-q*w + fx/(mass)
        v_dot = p*w-r*u + fy/(mass)
        w_dot = q*u-p*v + fz/(mass)
        
        # rotational kinematics
        e0_dot = 0.5 * (0*e0-p*e1-q*e2-r*e3)
        e1_dot = 0.5 * (p*e0-0*e1+r*e2-q*e3)
        e2_dot = 0.5 * (q*e0-r*e1+0*e2+p*e3)
        e3_dot = 0.5 * (r*e0+q*e1-p*e2+0*e3)

        # rotational dynamics
        p_dot = gamma1*p*q-gamma2*q*r + gamma3*l + gamma4*n
        q_dot = gamma5*p*r-gamma6*(p**2 - r**2) + (1/Jy)*m
        r_dot = gamma7*p*q - gamma1*q*r + gamma4*l + gamma8*n

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


#Initializing MAV Dynamics
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


fx = 0
fy = 0
fz = -9.81*mass
l = 0
m = 0
n = 0

forces_moments = np.array([
    [fx],
    [fy],
    [fz],
    [l],
    [m],
    [n]
    ])

#MAV.update(forces_moments)

states_history = [down0]
t = 0
time_history = [t]
for i in range(0, 1000, 1):
    MAV._state = MAV._state + MAV._derivatives(MAV._state, forces_moments)*Ts
    if MAV._state[2][0] < 0:
        break
    states_history.append(MAV._state[2][0])
    t = t+Ts
    time_history.append(t)

plt.plot(time_history, states_history)
plt.xlabel('Time [s]')
plt.ylabel('Altitude [m]')
plt.title('Altitude of Falling Cube')
plt.show()


