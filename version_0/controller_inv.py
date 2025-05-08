import numpy as np
import matplotlib.pyplot as plt
import mujoco 
import mujoco.viewer
import time
import desired_kinematic as dsk
import compute_activation as ca
import force_length_velocity_functions as flv

path_to_model = "../quadruped_inverse_dynamics/tendon_quadruped_ws_inair.xml"
model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)

omega = 0.5  #  lead QP solver failed for 1.5
# attempt_length = 5
dt = 0.005
duration_in_seconds = 10

RB,RF,LB,LF = dsk.create_cyclical_movements_fcn(omega , attempt_length = duration_in_seconds, dt=dt)

# get joint pos, qvel qacc individually
all_qpos = np.concatenate((RB[:,0:2], RF[:,0:2], LB[:,0:2], LF[:,0:2]), axis=1)
all_qvel = np.concatenate((RB[:,2:4], RF[:,2:4], LB[:,2:4], LF[:,2:4]), axis=1)
all_qacc = np.concatenate((RB[:,4:6], RF[:,4:6], LB[:,4:6], LF[:,4:6]), axis=1)


# solve compute joint torques and applied forces 

n_steps = all_qpos.shape[0]
n_joints = all_qpos.shape[1]
n_muscles = model.nu

joint_torques = np.zeros((n_steps, n_joints))
F0 = np.ones(n_muscles) * 1000

muscel_forces = np.zeros((n_steps, n_muscles))
activations = np.zeros((n_steps, n_muscles))


for t in range(n_steps):
    data.qpos[:] = all_qpos[t,:]
    data.qvel[:] = all_qvel[t,:]
    data.qacc[:] = all_qacc[t,:]

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
    a_t = F_t/F0

    muscel_forces[t, :] = F_t
    activations[t, :] = np.clip(a_t,0,1)

# save muscle activations in a file called activation_omega.txt where omega changes if omega changes
np.savetxt(f"activation_{omega}.txt", activations)

#plot activation of each of the 12 muscles in on a four subplots where three muscles are plotted in each subplot
# fig, axs = plt.subplots(4, 3, figsize=(15, 10))
# for i in range(4):
#     for j in range(3):
#         muscle_index = i * 3 + j
#         axs[i, j].plot(activations[:, muscle_index])
#         axs[i, j].set_title(f'Muscle {muscle_index + 1}')
#         axs[i, j].set_xlabel('Time step')
#         axs[i, j].set_ylabel('Activation')
#         axs[i, j].grid()
# plt.tight_layout()
# plt.show()

# fig.close()

# run mujoco model with the computed activations



