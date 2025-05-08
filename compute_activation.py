import numpy as np
from scipy.optimize import minimize

def solve_tendon_forces(RT, tau):
    m = RT.shape[1]
    def obj(F): return np.sum(F**2)
    cons = {'type': 'eq', 'fun': lambda F: RT @ F - tau}
    min_force = 0
    bounds = [(min_force, None)] * m
    F0_guess = np.ones(m)
    res = minimize(obj, F0_guess, constraints=cons, bounds=bounds)
    if res.success:
        return res.x
    else:
        raise RuntimeError('QP solver failed')

def invert_flv(F_desired, FL, FV, FP, F0):
    a = np.zeros_like(F_desired)
    valid_indices = (FL > 0) & (FV > 0)
    a[valid_indices] = (F_desired[valid_indices] / F0[valid_indices] - FP[valid_indices]) / (FL[valid_indices] * FV[valid_indices])
    return np.clip(a, 0, 1)


def fill_actuator_moments(num_muscles, num_joints,actuator_moments, muscle_joint_map=None):

    if muscle_joint_map is None:
        muscle_joint_map = [
            [0, 1],    # muscle 0 → joints 0,1
            [0],       # muscle 1 → joint 0
            [0, 1],    # muscle 2 → joints 0,1
            [2, 3],    # muscle 3 → joints 2,3
            [2],       # muscle 4 → joint 2
            [2, 3],    # muscle 5 → joints 2,3
            [4, 5],    # muscle 6 → joints 4,5
            [4],       # muscle 7 → joint 4
            [4, 5],    # muscle 8 → joints 4,5
            [6, 7],    # muscle 9 → joints 6,7
            [6],       # muscle 10 → joint 6
            [6, 7],    # muscle 11 → joints 6,7
        ]

    
    moment_matrix = np.zeros((num_muscles, num_joints))
    nonzero_values = actuator_moments[actuator_moments != 0]
    counter = 0

    for muscle_idx, joint_indices in enumerate(muscle_joint_map):
        for joint_idx in joint_indices:
            if counter < len(nonzero_values):
                moment_matrix[muscle_idx, joint_idx] = nonzero_values[counter]
                counter += 1
            else:
                break  # all values used

    return moment_matrix