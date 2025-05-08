
import numpy as np
import matplotlib.pyplot as plt
import mujoco 
import mujoco.viewer
import time


path_to_model = "../quadruped_inverse_dynamics/tendon_quadruped_ws_inair.xml"
model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)
muscle_activations = np.loadtxt("../quadruped_inverse_dynamics/activation_0.5.txt")


idx = 0
n_steps = muscle_activations.shape[0]
all_q_pos = np.zeros((n_steps, 8))
all_q_vel = np.zeros((n_steps, 8))
all_q_acc = np.zeros((n_steps, 8))

wait_time = 0.5

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:

    for idx in range(n_steps):
        step_start = time.time()
        data.ctrl[:] = muscle_activations[idx,:] * 100
        mujoco.mj_step(model, data)
        all_q_pos[idx,:] = data.qpos
        all_q_vel[idx,:] = data.qvel
        all_q_acc[idx,:] = data.qacc
        viewer.sync()
    

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


# create a function where the input are qpos or qvel or qacc, and the output is a plot with 4 subplots on each plot q0 q1 are plotted then q2 q3 and so on

# def plot_qpos_qvel_qacc(qpos):
#     fig, axs = plt.subplots(4, 1, figsize=(10, 10))
 

#     for i in range(4):
#         axs[i].plot(qpos[:, i*2], label='q' + str(i*2))
#         axs[i].plot(qpos[:, i*2 + 1], label='q' + str(i*2 + 1))
#         axs[i].set_title('Joint Position')
#         axs[i].legend()
#         axs[i].grid()

#     plt.tight_layout()
#     plt.show()

# plot_qpos_qvel_qacc(all_q_pos)