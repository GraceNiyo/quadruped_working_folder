import numpy as np 
import mujoco 

from scipy.optimize import minimize




path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'
model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)

desired_qpos = np.array([0.0, 0.0, 0.0, 0.0])  # [-0.014,-0.069, 0.05,-0.2]

data.qpos = desired_qpos
data.qvel = np.zeros(model.nv)
data.qacc = np.zeros(model.nv)

mujoco.mj_forward(model, data)

mujoco.mj_inverse(model, data)
required_torque = data.qfrc_inverse
print("Required Torque:", required_torque)


moment_arm_matrix = data.actuator_moment.copy()

def objective_function(ctrl):
    return np.sum(ctrl**2)  

def constraint_function(ctrl):
    produced_torques = moment_arm_matrix.T @ (model.actuator_gainprm[:,0] * ctrl)
    return produced_torques - required_torque

cons = {'type': 'eq', 'fun': constraint_function}
bounds = [(0,1)] * model.nu  

initial_guess = np.zeros(model.nu)


result = minimize(objective_function, initial_guess, bounds=bounds, constraints=cons)
if result.success:
    optimal_ctrl = result.x
    print("Optimal Control Inputs:", optimal_ctrl)
else:
    print("Optimization failed:", result.message)






