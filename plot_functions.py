import numpy as np
import matplotlib.pyplot as plt



def plot_muscle_activations(activations):
    """
    Plots the activation of each muscle in a 4x3 grid of subplots.

    Parameters:
    activations (numpy.ndarray): A 2D array where each column represents the activation of a muscle over time.
    """
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    for i in range(4):
        for j in range(3):
            muscle_index = i * 3 + j
            axs[i, j].plot(activations[:, muscle_index])
            axs[i, j].set_title(f'Muscle {muscle_index + 1}')
            axs[i, j].set_xlabel('Time step')
            axs[i, j].set_ylabel('Activation')
            axs[i, j].grid()
    plt.tight_layout()
    plt.show()

def plot_joint_state(qstate):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
 

    for i in range(4):
        axs[i].plot(qstate[:, i*2], label='q' + str(i*2))
        axs[i].plot(qstate[:, i*2 + 1], label='q' + str(i*2 + 1))
        axs[i].set_title('Joint Position')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    plt.show()