
# com.py computes the center of mass of a MuJoCo model
# import mujoco
import numpy as np


def com(model, data):
    total_mass = 0.0
    com = np.zeros(3)
    
    for i in range(model.nbody):
        mass = model.body_mass[i]
        pos = data.xipos[i]
        
        if mass > 0:
            com += mass * pos
            total_mass += mass
    
    if total_mass > 0:
        com /= total_mass
    else:
        com = np.zeros(3)
    
    return com