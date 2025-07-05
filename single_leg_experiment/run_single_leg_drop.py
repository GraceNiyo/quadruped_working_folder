
import mujoco 
import mujoco.viewer
import numpy as np
import time 
import os   
import compute_global_com
import compute_ground_reaction_force as compute_grf


####### Load the MuJoCo model 

path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'


####### Define muscle activations

M0 = 0 
M1 = 0  
M2 = 0.5
muscle_activations = np.array([M0, M1, M2])  # muscle activations for the three muscles


drop_heights = np.round(np.arange(-0.04, 1.1, 0.1), 2) 


closed_loop = False
# fdbk_gain = 0.1  # feedback gain for gamma_drive
gamma_drive = [1,1,1]  #feedback gain for gamma_drive

is_beta = False # (True:alpha-gamma, False: gamma only)
collateral = True # (True: collateral, False: no collateral)

wait_after_touchdown = 10  # seconds to wait after touchdown before stopping the simulation

model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)

torso_id = model.body("torso").id
touch_sensor_id = model.sensor("rbfoot_touch_sensor").id
foot_geom_id = model.geom("rbfoot").id
floor_geom_id = model.geom("floor").id
touch_sensor_id = model.sensor("rbfoot_touch_sensor").id



with mujoco.viewer.launch_passive(model, data, show_left_ui=False,show_right_ui=True) as viewer:

    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0
        # mujoco.mj_resetData(model, data)
        # viewer.sync()


    for drop_pos in drop_heights:

        print('Reset the model and data for each drop height')
        mujoco.mj_resetData(model, data)
        viewer.sync()   

        joint_position = []
        joint_velocity = []

        trunk_position = []
        global_com = []

        muscle_length = []
        muscle_velocity = []

        muscle_activation = []
        muscle_force = []

        touch_sensor = []
        ground_contact_force = []
        Ia = np.zeros(model.nu) 

        print(f"Drop height: {drop_pos}")
        touchdown_cond = None
        # <key qpos='0.0284271 -0.0503782 0.140377 -0.159533'/>
        data.qpos[:] = [0.0284271, drop_pos, 0.140377,-0.159533]
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        # initial_grf = compute_grf.get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id)
        # print(f"Initial Touch Sensor: {data.sensordata[touch_sensor_id]:.4f}")
        # print(f"Initial Ground Reaction Force: {np.round(initial_grf, 4)}")

    
        while viewer.is_running():
            start_time = time.time()
            alpha_drive = muscle_activations + Ia
            data.ctrl[:] = alpha_drive
            # start = time.perf_counter()
            mujoco.mj_step(model, data)
            # end = time.perf_counter()
            # cpu_time_per_step = end - start
            # print(f"CPU time per step: {cpu_time_per_step:.6f} seconds")
            contact_force = compute_grf.get_ground_reaction_force(model, data, foot_geom_id, floor_geom_id)
            # print(f"Ground Reaction Force: {np.round(contact_force, 4)}")

            viewer.sync()


            joint_position.append(data.qpos.copy())
            joint_velocity.append(data.qvel.copy())

            trunk_position.append(data.xpos[torso_id].copy())
            global_com.append(compute_global_com.com(model,data))

            muscle_length.append(data.actuator_length.copy())
            muscle_velocity.append(data.actuator_velocity.copy())
            muscle_activation.append(data.ctrl.copy())
            muscle_force.append(data.actuator_force.copy())

            touch_sensor.append(data.sensordata[touch_sensor_id].copy())
            ground_contact_force.append(contact_force.copy())



            if closed_loop and collateral and not is_beta:
                for m in range(model.nu):
                    Ia[m] = (alpha_drive[m] * gamma_drive[m]) * data.actuator_velocity[m] 

            elif closed_loop and not collateral and not is_beta:
                for m in range(model.nu):
                    Ia[m] = (gamma_drive[m]) * data.actuator_velocity[m]

            elif closed_loop and not collateral and is_beta:
                for m in range(model.nu):
                    Ia[m] = (alpha_drive[m]) * data.actuator_velocity[m]
            elif not closed_loop:
                Ia[:] = 0.0

            
        
            if touchdown_cond is None and data.sensordata[touch_sensor_id] > 0.0:
                print(f"Touchdown")
                touchdown_cond = True
                time_since_touchdown = data.time

            if touchdown_cond is not None and data.time >= time_since_touchdown + wait_after_touchdown:
                print(f"Stopping simulation after:",data.time)
                break


            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


         
        # data_dir = '../all_data/single_leg_experiment/alpha_gamma_with_collateral/'
        # if not os.path.exists(data_dir):
        #     os.makedirs(data_dir)  

        # np.savetxt(os.path.join(data_dir, f"joint_position_{drop_pos:.2f}.txt"), np.array(joint_position))
        # np.savetxt(os.path.join(data_dir, f"joint_velocity_{drop_pos:.2f}.txt"), np.array(joint_velocity))
        # np.savetxt(os.path.join(data_dir, f"trunk_position_{drop_pos:.2f}.txt"), np.array(trunk_position))
        # np.savetxt(os.path.join(data_dir, f"muscle_length_{drop_pos:.2f}.txt"), np.array(muscle_length))
        # np.savetxt(os.path.join(data_dir, f"muscle_velocity_{drop_pos:.2f}.txt"), np.array(muscle_velocity))
        # np.savetxt(os.path.join(data_dir, f"muscle_activation_{drop_pos:.2f}.txt"), np.array(muscle_activation))

        # np.savetxt(os.path.join(data_dir, f"touch_sensor_{drop_pos:.2f}.txt"), np.array(touch_sensor))
        # np.savetxt(os.path.join(data_dir, f"ground_contact_force_{drop_pos:.2f}.txt"), np.array(ground_contact_force))
        # np.savetxt(os.path.join(data_dir, f"global_com_{drop_pos:.2f}.txt"), np.array(global_com))

        # # np.savetxt(os.path.join(data_dir, f"joint_position_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(joint_position))
        # # np.savetxt(os.path.join(data_dir, f"joint_velocity_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(joint_velocity))
        # # np.savetxt(os.path.join(data_dir, f"trunk_position_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(trunk_position))
        # # np.savetxt(os.path.join(data_dir, f"com_trunk_body_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(com_trunk_body))
        # # np.savetxt(os.path.join(data_dir, f"muscle_length_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(muscle_length))
        # # np.savetxt(os.path.join(data_dir, f"muscle_velocity_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(muscle_velocity))
        # # np.savetxt(os.path.join(data_dir, f"muscle_activation_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(muscle_activation))
        # # np.savetxt(os.path.join(data_dir, f"touch_sensor_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(touch_sensor))
        # # np.savetxt(os.path.join(data_dir, f"global_com_{drop_pos:.2f}_{fdbk_gain}.txt"), np.array(global_com))


        # print(f"Data saved to {data_dir}")






        


  

        














