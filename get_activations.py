
import numpy as np
import matplotlib.pyplot as plt
import mujoco 
import mujoco.viewer
import time
import desired_kinematic as dsk
import compute_activation as ca
import force_length_velocity_functions as flv
import os


def compute_and_save_activations(model_path, omega, dt, duration_in_seconds,activation_folder):
    """
    Computes muscle activations for a given model and saves them to a file.

    Parameters:
    model_path (str): Path to the MuJoCo model XML file.
    omega (float): Frequency for cyclical movements.
    dt (float): Time step for simulation.
    duration_in_seconds (int): Duration of the simulation in seconds.

    Returns:
    str: Path to the saved activation file.
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    RB, RF, LB, LF = dsk.create_cyclical_movements_fcn(omega, attempt_length=duration_in_seconds, dt=dt)

    # Get joint positions, velocities, and accelerations
    all_qpos = np.concatenate((RB[:, 0:2], RF[:, 0:2], LB[:, 0:2], LF[:, 0:2]), axis=1)
    all_qvel = np.concatenate((RB[:, 2:4], RF[:, 2:4], LB[:, 2:4], LF[:, 2:4]), axis=1)
    all_qacc = np.concatenate((RB[:, 4:6], RF[:, 4:6], LB[:, 4:6], LF[:, 4:6]), axis=1)

    # Initialize variables
    n_steps = all_qpos.shape[0]
    n_joints = all_qpos.shape[1]
    n_muscles = model.nu

    # joint_torques = np.zeros((n_steps, n_joints))
    F0 = np.ones(n_muscles) * 1000

    muscel_forces = np.zeros((n_steps, n_muscles))
    activations = np.zeros((n_steps, n_muscles))

    # Compute activations
    for t in range(n_steps):
        data.qpos[:] = all_qpos[t, :]
        data.qvel[:] = all_qvel[t, :]
        data.qacc[:] = all_qacc[t, :]

        mujoco.mj_inverse(model, data)

        joint_torque = data.qfrc_inverse
        actuator_moment_values = data.actuator_moment
        moment_matrix = ca.fill_actuator_moments(n_muscles, n_joints, actuator_moment_values)
        actuator_moments_transpose = moment_matrix.T

        L_t = data.actuator_length
        V_t = data.actuator_velocity

        FL_t = np.array([flv.compute_FL(L) for L in L_t])
        FP_t = np.array([flv.compute_FP(L) for L in L_t])
        FV_t = np.array([flv.compute_FV(V) for V in V_t])

        F_t = ca.solve_tendon_forces(actuator_moments_transpose, joint_torque)
        # a_t = ca.invert_flv(F_t, FL_t, FV_t, FP_t, F0)

        a_t = F_t / F0

        muscel_forces[t, :] = F_t
        activations[t, :] = np.clip(a_t, 0, 1)

    # Save activations to a file
    activation_file = f"activation_{omega}.txt"

    os.makedirs(activation_folder, exist_ok=True)
    activation_file = os.path.join(activation_folder, f"activation_{omega}.txt")
    np.savetxt(activation_file, activations)

    return activation_file, all_qpos, all_qvel, all_qacc