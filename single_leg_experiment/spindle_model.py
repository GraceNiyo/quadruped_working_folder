import numpy as np

# class spindle_model tbd -- should be a class 
def gamma_driven_spindle_model_(length, velocity, muscle_length_range,optimal_length,gamma_static, gamma_dynamics):


    min_length, max_length = muscle_length_range

    # Calculate the normalized length
    normalized_length = (length - min_length) / (max_length - min_length)
   