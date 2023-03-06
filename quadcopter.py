import numpy as np
import sympy as sym

g = 9.81    # Gravity acceleration [m/s^2]


class Quadcopter6DOF:

    def __init__(self, m, m_motor, l_arm, dt):
        # Physical parameters of the quadcopter
        self.m = m              # Quadcopter mass [kg]
        self.m_motor = m_motor  # Motor mass [kg]
        self.l_arm = l_arm      # Quadcopter length [m]
        self.dt = dt            # Integration step

        # Compute Moments of Inertia [kg * m^2] (assume the motors as point masses)
        self.Ixx = 2 * (self.m_motor * (self.l_arm ** 2))
        self.Iyy = 2 * (self.m_motor * (self.l_arm ** 2))
        self.Izz = 4 * (self.m_motor * (self.l_arm ** 2))

        # We consider the following output of the system
        # y.T = [x, y, z, psi, theta, phi]
        self.C = np.block([[np.eye(6), np.zeros((6, 6))],
                           [np.zeros((6, 12))]])
        self.D = np.zeros((6, 4))

    def linearize(self, x_bar, u_bar):
        """
        Inputs:
        x: equilibrium state of the quadcopter system as a numpy array [x, y, z, psi, theta, phi, vx, vy, vz, p, q, r]
        u: equilibrium control as numpy array [Total_Force, Roll_Torque, Pitch_Torque, Yaw_Torque]

        Output:
        Matrices A and B of the state space system linearized in the equilibrium point
        """
        # Symbolic variables
        x, y, z, psi, theta, phi, vx, vy, vz, p, q, r, U1, U2, U3, U4 = \
            sym.symbols('x y z psi theta phi vx vy vz p q r U1 U2 U3 U4')

        # Equilibrium point
        x_eq, y_eq, z_eq, psi_eq, theta_eq, phi_eq, vx_eq, vy_eq, vz_eq, p_eq, q_eq, r_eq, U1_eq, U2_eq, U3_eq, U4_eq =\
            x_bar[0], x_bar[1], x_bar[2], x_bar[3], x_bar[4], x_bar[5], \
            x_bar[6], x_bar[7], x_bar[8], x_bar[9], x_bar[10], x_bar[11], \
            u_bar[0], u_bar[1], u_bar[2], u_bar[3]

        # Equations of motion
        x_dot = vx
        y_dot = vy
        z_dot = vz
        psi_dot = q * sym.sin(phi) / sym.cos(theta) + r * sym.cos(phi) / sym.cos(theta)
        theta_dot = q * sym.cos(phi) - r * sym.sin(phi)
        phi_dot = p + q * sym.sin(phi) * sym.tan(theta) + r * sym.cos(phi) * sym.tan(theta)
        vx_dot = 0 - U1 / self.m * (sym.sin(phi) * sym.sin(psi) + sym.cos(phi) * sym.cos(psi) * sym.sin(theta))
        vy_dot = 0 - U1 / self.m * (sym.cos(psi) * sym.sin(phi) - sym.cos(phi) * sym.sin(psi) * sym.sin(theta))
        vz_dot = g - U1 / self.m * sym.cos(phi) * sym.cos(theta)
        p_dot = (self.Iyy - self.Izz) / self.Ixx * q * r + U2 / self.Ixx
        q_dot = (self.Izz - self.Ixx) / self.Iyy * p * r + U3 / self.Iyy
        r_dot = (self.Ixx - self.Iyy) / self.Izz * p * q + U4 / self.Izz

        # Linearization
        nonlin_mat = sym.Matrix([x_dot, y_dot, z_dot, psi_dot, theta_dot, phi_dot,
                                 vx_dot, vy_dot, vz_dot, p_dot, q_dot, r_dot])
        Ja = nonlin_mat.jacobian(sym.Matrix([x, y, z, psi, theta, phi, vx, vy, vz, p, q, r]))
        Jb = nonlin_mat.jacobian(sym.Matrix([U1, U2, U3, U4]))
        A_lin = Ja.subs(
            [(x, x_eq), (y, y_eq), (z, z_eq), (psi, psi_eq), (theta, theta_eq), (phi, phi_eq),
             (vx, vx_eq), (vy, vy_eq), (vz, vz_eq), (p, p_eq), (q, q_eq), (r, r_eq),
             (U1, U1_eq), (U2, U2_eq), (U3, U3_eq), (U4, U4_eq)])
        B_lin = Jb.subs(
            [(x, x_eq), (y, y_eq), (z, z_eq), (psi, psi_eq), (theta, theta_eq), (phi, phi_eq),
             (vx, vx_eq), (vy, vy_eq), (vz, vz_eq), (p, p_eq), (q, q_eq), (r, r_eq),
             (U1, U1_eq), (U2, U2_eq), (U3, U3_eq), (U4, U4_eq)])

        return np.array(A_lin, dtype='float64'), np.array(B_lin, dtype='float64')

    def linearize_discr(self, x_bar, u_bar):
        """
        Inputs:
        x: equilibrium state of the quadcopter system as a numpy array [x, y, z, psi, theta, phi, vx, vy, vz, p, q, r]
        u: equilibrium control as numpy array [Total_Force, Roll_Torque, Pitch_Torque, Yaw_Torque]

        Output:
        Matrices A and B of the state space system linearized in the equilibrium point
        and discretized according to the integration step
        """
        # Symbolic variables
        x, y, z, psi, theta, phi, vx, vy, vz, p, q, r, U1, U2, U3, U4 = \
            sym.symbols('x y z psi theta phi vx vy vz p q r U1 U2 U3 U4')

        # Equilibrium point
        x_eq, y_eq, z_eq, psi_eq, theta_eq, phi_eq, vx_eq, vy_eq, vz_eq, p_eq, q_eq, r_eq, U1_eq, U2_eq, U3_eq, U4_eq =\
            x_bar[0], x_bar[1], x_bar[2], x_bar[3], x_bar[4], x_bar[5], \
            x_bar[6], x_bar[7], x_bar[8], x_bar[9], x_bar[10], x_bar[11], \
            u_bar[0], u_bar[1], u_bar[2], u_bar[3]

        # Equations of motion
        x_dot = vx
        y_dot = vy
        z_dot = vz
        psi_dot = q * sym.sin(phi) / sym.cos(theta) + r * sym.cos(phi) / sym.cos(theta)
        theta_dot = q * sym.cos(phi) - r * sym.sin(phi)
        phi_dot = p + q * sym.sin(phi) * sym.tan(theta) + r * sym.cos(phi) * sym.tan(theta)
        vx_dot = 0 - U1 / self.m * (sym.sin(phi) * sym.sin(psi) + sym.cos(phi) * sym.cos(psi) * sym.sin(theta))
        vy_dot = 0 - U1 / self.m * (sym.cos(psi) * sym.sin(phi) - sym.cos(phi) * sym.sin(psi) * sym.sin(theta))
        vz_dot = g - U1 / self.m * sym.cos(phi) * sym.cos(theta)
        p_dot = (self.Iyy - self.Izz) / self.Ixx * q * r + U2 / self.Ixx
        q_dot = (self.Izz - self.Ixx) / self.Iyy * p * r + U3 / self.Iyy
        r_dot = (self.Ixx - self.Iyy) / self.Izz * p * q + U4 / self.Izz

        # Linearization
        nonlin_mat = sym.Matrix([x_dot, y_dot, z_dot, psi_dot, theta_dot, phi_dot,
                                 vx_dot, vy_dot, vz_dot, p_dot, q_dot, r_dot])
        Ja = nonlin_mat.jacobian(sym.Matrix([x, y, z, psi, theta, phi, vx, vy, vz, p, q, r]))
        Jb = nonlin_mat.jacobian(sym.Matrix([U1, U2, U3, U4]))
        A_lin = Ja.subs(
            [(x, x_eq), (y, y_eq), (z, z_eq), (psi, psi_eq), (theta, theta_eq), (phi, phi_eq),
             (vx, vx_eq), (vy, vy_eq), (vz, vz_eq), (p, p_eq), (q, q_eq), (r, r_eq),
             (U1, U1_eq), (U2, U2_eq), (U3, U3_eq), (U4, U4_eq)])
        B_lin = Jb.subs(
            [(x, x_eq), (y, y_eq), (z, z_eq), (psi, psi_eq), (theta, theta_eq), (phi, phi_eq),
             (vx, vx_eq), (vy, vy_eq), (vz, vz_eq), (p, p_eq), (q, q_eq), (r, r_eq),
             (U1, U1_eq), (U2, U2_eq), (U3, U3_eq), (U4, U4_eq)])

        # Compute discretized A and B for LQR
        A_discr = np.array(sym.eye(12) + A_lin * self.dt, dtype='float64')
        B_discr = np.array(B_lin * self.dt, dtype='float64')

        return A_discr, B_discr

    def model_integrator(self, x_prev, u):
        """
        Inputs:
        x: state of the quadcopter system as a numpy array [x, y, z, psi, theta, phi, vx, vy, vz, p, q, r]
        u: control as numpy array [Total_Force, Roll_Torque, Pitch_Torque, Yaw_Torque]

        Output:
        New state of the quadcopter as a numpy array
        """
        x, y, z, psi, theta, phi, vx, vy, vz, p, q, r, U1, U2, U3, U4 = \
            x_prev[0], x_prev[1], x_prev[2], x_prev[3], x_prev[4], x_prev[5], \
            x_prev[6], x_prev[7], x_prev[8], x_prev[9], x_prev[10], x_prev[11], u[0], u[1], u[2], u[3]

        # Equations of motion
        x_dot = vx
        y_dot = vy
        z_dot = vz
        psi_dot = q * sym.sin(phi) / sym.cos(theta) + r * sym.cos(phi) / sym.cos(theta)
        theta_dot = q * sym.cos(phi) - r * sym.sin(phi)
        phi_dot = p + q * sym.sin(phi) * sym.tan(theta) + r * sym.cos(phi) * sym.tan(theta)
        vx_dot = 0 - U1 / self.m * (sym.sin(phi) * sym.sin(psi) + sym.cos(phi) * sym.cos(psi) * sym.sin(theta))
        vy_dot = 0 - U1 / self.m * (sym.cos(psi) * sym.sin(phi) - sym.cos(phi) * sym.sin(psi) * sym.sin(theta))
        vz_dot = g - U1 / self.m * sym.cos(phi) * sym.cos(theta)
        p_dot = (self.Iyy - self.Izz) / self.Ixx * q * r + U2 / self.Ixx
        q_dot = (self.Izz - self.Ixx) / self.Iyy * p * r + U3 / self.Iyy
        r_dot = (self.Ixx - self.Iyy) / self.Izz * p * q + U4 / self.Izz

        # Compute next state
        x_next = x + self.dt * x_dot
        y_next = y + self.dt * y_dot
        z_next = z + self.dt * z_dot
        psi_next = psi + self.dt * psi_dot
        theta_next = theta + self.dt * theta_dot
        phi_next = phi + self.dt * phi_dot
        vx_next = vx + self.dt * vx_dot
        vy_next = vy + self.dt * vy_dot
        vz_next = vz + self.dt * vz_dot
        p_next = p + self.dt * p_dot
        q_next = q + self.dt * q_dot
        r_next = r + self.dt * r_dot

        x = np.array(
            [x_next, y_next, z_next, psi_next, theta_next, phi_next, vx_next, vy_next, vz_next, p_next, q_next, r_next],
            dtype='float64')

        return x


class Quadcopter3DOF:

    def __init__(self, m, I, r, dt):
        self.m = m      # Quadcopter mass [kg]
        self.I = I      # Quadcopter inertia [kg * m^2]
        self.r = r      # Quadcopter length [m]
        self.dt = dt    # Integration step

    def model_integrator(self, x, u):
        """
        Inputs:
        x: state of the quadcopter system as a numpy array [x, y, theta, vx, vy, omega]
        u: control as numpy array [ThrustRotor1, ThrustRotor1]

        Output:
        New state of the quadcopter as a numpy array
        """
        z = x[0]
        y = x[1]
        th = x[2]
        vx = x[3]
        vy = x[4]
        om = x[5]
        x_next = (z + self.dt * vx)
        y_next = (y + self.dt * vy)
        th_next = (th + self.dt * om)
        vx_next = vx + self.dt*(-np.sin(th)*(u[0]+u[1])/self.m)
        vy_next = vy + self.dt*(-g+(np.cos(th)*(u[0]+u[1])/self.m))
        om_next = om + self.dt*((u[0]-u[1])/self.I)*self.r
        x = np.array([x_next, y_next, th_next, vx_next, vy_next, om_next])
        return x
