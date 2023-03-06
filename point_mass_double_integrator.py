import numpy as np


class PointMassDoubleIntegrator:
    """
    3DOF State Space model approximated as a Point Mass Double Integrator System (Mechanical System of the First Order)

    State x:            [x, y, z, vx, vy, vz]
    Control Input u:    [ax, ay, az]
    Output y:           [x, y, z]
    """

    def __init__(self, m):
        self.m = m

        k = 18.6 * (10 ** (-6))  # Viscous friction coefficient air (at 27°C)
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, -k / self.m, 0, 0],
                           [0, 0, 0, 0, -k / self.m, 0],
                           [0, 0, 0, 0, 0, -k / self.m]])
        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1 / self.m, 0, 0],
                           [0, 1 / self.m, 0],
                           [0, 0, 1 / self.m]])
        self.C = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        self.D = np.zeros((3, 3))
