
import numpy as np
import matplotlib.pyplot as plt

def generate_smooth_force_profile(
    total_duration,
    timestep,
    pulse_start_time,
    pulse_end_time,
    peak_force,
    transition_duration
):
    """
    Generates a smooth step force profile for the entire simulation duration.
    """
    num_steps = int(total_duration / timestep) + 1
    time_points = np.linspace(0, total_duration, num_steps)
    force_profile = np.zeros(num_steps)

    for i, t in enumerate(time_points):
        force = 0.0
        # Ramp up phase
        if t >= pulse_start_time and t < pulse_start_time + transition_duration:
            t_scaled = (t - pulse_start_time) / transition_duration
            force_factor = 0.5 * (1 + np.tanh(5 * (t_scaled - 0.5)))
            force = peak_force * force_factor
        # Full force phase
        elif t >= pulse_start_time + transition_duration and t < pulse_end_time:
            force = peak_force
        # Ramp down phase
        elif t >= pulse_end_time and t < pulse_end_time + transition_duration:
            t_scaled = (t - pulse_end_time) / transition_duration
            force_factor = 0.5 * (1 - np.tanh(5 * (t_scaled - 0.5)))
            force = peak_force * force_factor
        # No force outside these ranges
        else:
            force = 0.0
        
        force_profile[i] = force
    
    return force_profile

# Example usage
if __name__ == "__main__":
    total_duration = 5.0  # Total simulation time in seconds
    timestep = 0.005  # Time step for the simulation
    pulse_start_time = 2.0  # Start time of the force pulse
    pulse_end_time = 3  # End time of the force pulse
    peak_force = -1.0  # Peak force in Newtons (negative for downward force)
    transition_duration = 0.5  # Duration of the ramp up/down in seconds

    force_profile = generate_smooth_force_profile(
        total_duration,
        timestep,
        pulse_start_time,
        pulse_end_time,
        peak_force,
        transition_duration
    )

    plt.plot(np.arange(0, total_duration + timestep, timestep), force_profile)
    plt.title("Smooth Step Force Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.show()


   