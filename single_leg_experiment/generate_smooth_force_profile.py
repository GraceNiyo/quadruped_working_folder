
import numpy as np
import matplotlib.pyplot as plt

def generate_smooth_force_profile(
    duration,
    timestep,
    rise_time,
    decay_time,
    peak_force,

):
    """
    Generates a smooth step force profile for the entire simulation duration.
    """
    if rise_time < 0 or decay_time < 0:
        raise ValueError("Rise and decay times must be non-negative.")
    if duration < rise_time + decay_time:
        raise ValueError("Total duration must be greater than the sum of rise and decay times.")
    
    if timestep <= 0:
        raise ValueError("Timestep must be a positive value.")
    
    force_values = []

    steady_state_duration = duration - rise_time - decay_time
    num_steps = int(duration / timestep)

    for i in range(num_steps):
        current_time = i * timestep

        if current_time < rise_time:
            # Rise phase: Smoothly transition from 0 to 1.
            t = current_time / rise_time
            force_value = (t * t * (3 - 2 * t)) * peak_force
        elif current_time < rise_time + steady_state_duration:
            # Steady state: Value remains at 1.
            force_value = peak_force
        else:
            # Decay phase: Smoothly transition from 1 to 0.
            t = (current_time - (duration - decay_time)) / decay_time
            force_value = (1 - (t * t * (3 - 2 * t))) * peak_force

        force_values.append(force_value)

    return force_values


if __name__ == "__main__":
    duration = 1.0  # Total simulation time in seconds
    timestep = 0.005  # Time step for the simulation
    rise_time = 0.1  # Time to reach peak force
    decay_time = 0.1  # Time to decay to zero force
    peak_force = -1.0  # Peak force in Newtons (negative for downward force)
    

    force_profile = generate_smooth_force_profile(
        duration,
        timestep,
        rise_time,
        decay_time,
        peak_force
    )

    plt.plot(np.arange(0, duration, timestep), force_profile)
    plt.title("Smooth Step Force Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.show()


   