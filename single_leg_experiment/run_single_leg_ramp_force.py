import mujoco
import mujoco.viewer
import numpy as np
import time
import os


try:
    import compute_global_com
    import compute_ground_reaction_force as compute_grf
except ImportError:
    print("Warning: 'compute_global_com' or 'compute_ground_reaction_force' modules not found.")
    print("Please ensure they are in Python path or provide their implementations.")



# --- Configuration Parameters ---
path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'
# base_data_dir = '../all_data/single_leg_experiment/force_pulse_data'
# os.makedirs(base_data_dir, exist_ok=True)

# Muscle activation levels (M0, M1, M2)
M0 = 0
M1 = 0
M2 = 0.5
muscle_activations = np.array([M0, M1, M2])


# Apply force from 0 N down to -50 N (in z-direction)
force_vector = np.arange(-0.1, -5.1, -0.1)
#  phase_durations for total simulation time and phase transitions
delay_time = 5  # seconds
pulse_duration = 1  # seconds
post_pulse_record_duration = 5 # seconds

# Reflex gain values
reflex_gains = np.arange(0, 110, 10)


closed_loop = True  # enable feedback control
is_beta = False   # (True:beta drive, False: alpha, gamma drive)
collateral = False  # (True: alpha_gamma_with_collateral, False: no collateral)


# --- Model Loading and Initialization ---
try:
    model = mujoco.MjModel.from_xml_path(path_to_model)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading MuJoCo model from {path_to_model}: {e}")
    exit()

# Get IDs from the model
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
if body_id == -1:
    print("Error: 'torso' body not found in model.")
    exit()
force_location_on_model = data.xpos[body_id]

torso_id = model.body("torso").id
if torso_id == -1:
    print("Error: 'torso' body not found in model.")
    exit()

touch_sensor_id = model.sensor("rbfoot_touch_sensor").id
if touch_sensor_id == -1:
    print("Error: 'rbfoot_touch_sensor' not found in model.")
    exit()

foot_geom_id = model.geom("rbfoot").id
if foot_geom_id == -1:
    print("Error: 'rbfoot' geom not found in model.")
    exit()

floor_geom_id = model.geom("floor").id
if floor_geom_id == -1:
    print("Error: 'floor' geom not found in model.")
    exit()


# List to store results from all simulation runs
all_simulation_results = []

# durations for each new phase
phase_durations = [
    2.0, # Phase 1: Only muscle activation, no external force (0 to 2s)
    3.0, # Phase 2: Muscle activation + Ia feedback, no external force (2 to 5s)
    1.0, # Phase 3: Muscle activation + Ia feedback + external force (the pulse) (5 to 6s)
    3.0, # Phase 4: Muscle activation + Ia feedback, no external force (6 to 9s)
    2.0  # Phase 5: Only muscle activation, no external force (9 to 11s)
]

# Calculate cumulative time thresholds for phase transitions
cumulative_times = np.cumsum(phase_durations)
total_sim_duration = cumulative_times[-1]


# --- Main Simulation Loop ---

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True) as viewer:

    # Set the camera to fixed view if available
    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0

    # Loop through each force magnitude
    for f in force_vector:
        # Define the force vector for the current run
        force = np.array([0, 0, f])
        torque = np.array([0, 0, 0]) # No torque applied


        for gain in reflex_gains:
            print(f'\n--- Running simulation for Force: {f} N, Reflex Gain: {gain} ---')

            # Reset the model and data to initial state for each new simulation run
            mujoco.mj_resetData(model, data)
            if viewer.is_running():
                viewer.sync()


            simulation_start_data_time = data.time

            # Initialize data storage for the current simulation run
            run_data = {
                'force': f,
                'gain': gain,
                'joint_position': [],
                'joint_velocity': [],
                'trunk_position': [],
                'global_com': [],
                'muscle_length': [],
                'muscle_velocity': [],
                'muscle_activation': [],
                'muscle_force': [],
                'touch_sensor': [],
                'ground_contact_force': [],
                'spindle_feedback': [],
                'time': []
            }

            # Initialize Ia (afferent feedback signal)
            Ia = np.zeros(model.nu)
            total_steps = int(total_sim_duration / model.opt.timestep) + 1 # Add 1 to ensure last step is included

            # Run the simulation for the calculated total number of steps
            for step_count in range(total_steps):
                step_start_real_time = time.time() # will be removed later, used for debugging now

                # Check if the viewer is still running. If closed, terminate the current run.
                if viewer.is_running():
                    viewer.sync() # Update the viewer display
                else:
                    print("Viewer closed. Terminating current simulation run and proceeding to next or finishing.")
                    break # Break out of the current simulation run loop


                current_sim_time_relative = data.time - simulation_start_data_time

                # --- Simulation Phase Logic ---

                if current_sim_time_relative < cumulative_times[0]: # Phase 1: 0 to 2 seconds
                    data.ctrl[:] = muscle_activations
                    data.qfrc_applied[:] = 0.0 # Ensure no external force is applied
                elif current_sim_time_relative < cumulative_times[1]: # Phase 2: 2 to 5 seconds
                    # Muscle activation + Ia feedback, no external force
                    if gain == 0:
                        drive_to_muscle = muscle_activations
                        Ia[:] = 0.0
                    else:
                        if is_beta:
                            drive_to_muscle = muscle_activations * gain + Ia
                        else:
                            drive_to_muscle = muscle_activations + Ia

                        if closed_loop:
                            if collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m] * gain) * data.actuator_velocity[m]

                            elif not collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (gain) * data.actuator_velocity[m]
                            elif is_beta and not collateral:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m]) * data.actuator_velocity[m]
                        else:
                            Ia[:] = 0.0
                    data.ctrl[:] = drive_to_muscle
                    data.qfrc_applied[:] = 0.0 # No external force
                elif current_sim_time_relative < cumulative_times[2]: # Phase 3: 5 to 6 seconds
                    # Muscle activation + Ia feedback + external force (the pulse)
                    if gain == 0:
                        drive_to_muscle = muscle_activations
                        Ia[:] = 0.0
                    else:
                        if is_beta:
                            drive_to_muscle = muscle_activations * gain + Ia
                        else:
                            drive_to_muscle = muscle_activations + Ia

                        if closed_loop:
                            if collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m] * gain) * data.actuator_velocity[m]

                            elif not collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (gain) * data.actuator_velocity[m]
                            elif not collateral and is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m]) * data.actuator_velocity[m]
                        else:
                            Ia[:] = 0.0
                    data.ctrl[:] = drive_to_muscle
                    # Apply the external force pulse
                    mujoco.mj_applyFT(model, data, force, torque, force_location_on_model, body_id, data.qfrc_applied)
                elif current_sim_time_relative < cumulative_times[3]: # Phase 4: 6 to 9 seconds
                    # Muscle activation + Ia feedback, no external force
                    if gain == 0:
                        drive_to_muscle = muscle_activations
                        Ia[:] = 0.0
                    else:
                        if is_beta:
                            drive_to_muscle = muscle_activations * gain + Ia
                        else:
                            drive_to_muscle = muscle_activations + Ia

                        if closed_loop:
                            if collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m] * gain) * data.actuator_velocity[m]

                            elif not collateral and not is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (gain) * data.actuator_velocity[m]
                            elif not collateral and is_beta:
                                for m in range(model.nu):
                                    Ia[m] = (drive_to_muscle[m]) * data.actuator_velocity[m]
                        else:
                            Ia[:] = 0.0
                    data.ctrl[:] = drive_to_muscle
                    data.qfrc_applied[:] = 0.0 # No external force
                else: # Phase 5: 9 to 11 seconds (and beyond, until total_steps is reached)
                    # Only muscle activation, no external force
                    data.ctrl[:] = muscle_activations
                    data.qfrc_applied[:] = 0.0

                # --- Step the simulation ---
                mujoco.mj_step(model, data)

                # --- Collect Data for the Current Step ---
                contact_force = compute_grf.get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id)

                run_data['joint_position'].append(data.qpos.copy())
                run_data['joint_velocity'].append(data.qvel.copy())
                run_data['trunk_position'].append(data.xpos[torso_id].copy())
                run_data['global_com'].append(compute_global_com.com(model, data))
                run_data['muscle_length'].append(data.actuator_length.copy())
                run_data['muscle_velocity'].append(data.actuator_velocity.copy())
                run_data['muscle_activation'].append(data.ctrl.copy())
                run_data['muscle_force'].append(data.actuator_force.copy())
                run_data['touch_sensor'].append(data.sensordata[touch_sensor_id].copy())
                run_data['ground_contact_force'].append(contact_force.copy())
                run_data['spindle_feedback'].append(Ia.copy())
                run_data['time'].append(data.time)


                # syncronize to model real_time simulation
                time_elapsed_real = time.time() - step_start_real_time
                # Calculate time remaining to match model.opt.timestep
                time_to_sleep = model.opt.timestep - time_elapsed_real
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else: # warning if simulation is running slower than real-time
                    print(f"Warning: Simulation step {step_count} exceeded the defined timestep! ({time_elapsed_real:.4f}s vs {model.opt.timestep:.4f}s)")


            # Determine the feedback type string for the filename
            # if is_beta:
            #     feedback_type_str = "beta"
            # elif collateral: # is_beta is False and collateral is True
            #     feedback_type_str = "alpha_gamma_collateral"
            # else: # is_beta is False and collateral is False
            #     feedback_type_str = "alpha_gamma"

            # base_filename = f"{feedback_type_str}_force_{abs(f)}_gain_{gain}"

            # # Save each data array to a separate .txt file
            # for data_key, data_list in run_data.items():
            #     # Skip 'force' and 'gain' as they are part of the filename
            #     if data_key in ['force', 'gain']:
            #         continue

            #     # Convert list of arrays to a single NumPy array for saving
            #     # Handle scalar data (like touch_sensor if it's a single value per step)
            #     if isinstance(data_list[0], (float, int, np.ndarray)) and np.ndim(data_list[0]) == 0:
            #         data_to_save = np.array(data_list).reshape(-1, 1) # Reshape to a column vector
            #     else:
            #         data_to_save = np.array(data_list)

            #     file_path = os.path.join(base_data_dir, f"{base_filename}_{data_key}.txt")
            #     np.savetxt(file_path, data_to_save, fmt='%.8f', delimiter='\t') # Using tab delimiter for MATLAB
            #     # print(f"Saved {data_key} to {file_path}")

            # Optionally, keep the data in all_simulation_results if you need it in memory for further processing
            # all_simulation_results.append(run_data)

# print("\nAll simulations completed.")
# print(f"Total {len(all_simulation_results)} simulation runs performed.")
# print(f"All data saved to: {base_data_dir}")




