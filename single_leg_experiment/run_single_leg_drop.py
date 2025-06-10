
import mujoco 
import mujoco.viewer
import numpy as np
import time 
import os   
import compute_global_com


####### Load the MuJoCo model 

path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'


####### Define muscle activations

M0 = 0.17    
M1 = 0.35    
M2 = 0.23
muscle_activations = np.array([M0, M1, M2])


QPOS = [-0.014,-0.069, 0.05,-0.2]

drop_heights = np.round(np.arange(-0.05, 2.05, 0.05), 2) #.nparange(-0.05, 1.1, 0.1) 

closed_loop = False
fdbk_gain = 1.0  # feedback gain for gamma_drive

is_beta = False # (True:alpha-gamma, False: gamma only)
collateral = True

wait_after_touchdown = 5.0  # seconds to wait after touchdown before stopping the simulation

model = mujoco.MjModel.from_xml_path(path_to_model)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False,show_right_ui=True) as viewer:

    if model.ncam > 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = 0
        mujoco.mj_resetData(model, data)

    for drop_pos in drop_heights:

        joint_position = []
        joint_velocity = []

        trunk_position = []
        com_trunk_body = []
        global_com = []

        muscle_length = []
        muscle_velocity = []

        muscle_activation = []
        touch_sensor = []

        II = np.zeros(model.nu)  # Initialize feedback control input



        print(f"Drop height: {drop_pos}")

        mujoco.mj_resetData(model, data)
        torso_id = model.body("torso").id

        sensor_id = model.sensor("rb_sensor").id
        touchdown_cond = None
    
        data.qpos[:] = [-0.014,drop_pos,0.05,-0.2]
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        viewer.sync()
      

        while viewer.is_running():
            start_time = time.time()
            alpha_drive = muscle_activations + II
            data.ctrl[:] = alpha_drive
            mujoco.mj_step(model, data)
            viewer.sync()

            joint_position.append(data.qpos.copy())
            joint_velocity.append(data.qvel.copy())
            com_trunk_body.append(data.xipos[torso_id].copy())
            trunk_position.append(data.xpos[torso_id].copy())
            muscle_length.append(data.actuator_length.copy())
            muscle_velocity.append(data.actuator_velocity.copy())
            muscle_activation.append(data.ctrl.copy())
            touch_sensor.append(data.sensordata.copy())
            global_com.append(compute_global_com.com(model,data))

            if closed_loop and collateral and not is_beta:
                for m in range(model.nu):
                    II[m] = (alpha_drive[m] * fdbk_gain) * data.actuator_velocity[m] 

            elif closed_loop and not collateral and not is_beta:
                for m in range(model.nu):
                    II[m] = (fdbk_gain) * data.actuator_velocity[m]

            elif closed_loop and not collateral and is_beta:
                for m in range(model.nu):
                    II[m] = (alpha_drive[m]) * data.actuator_velocity[m]
            elif not closed_loop:
                II[:] = 0.0

            
        
            if touchdown_cond is None and data.sensordata[sensor_id] > 0.0:
                print(f"Touchdown")
                touchdown_cond = True
                time_since_touchdown = data.time

            if touchdown_cond is not None and data.time >= time_since_touchdown + wait_after_touchdown:
                print(f"Stopping simulation after:",data.time)
                break


            time_until_next_step = model.opt.timestep - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


            
        data_dir = '../all_data/single_leg_experiment/no_feedback/test'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  

        # np.savetxt(os.path.join(data_dir, f"joint_position_{drop_pos:.2f}.txt"), np.array(joint_position))
        # np.savetxt(os.path.join(data_dir, f"joint_velocity_{drop_pos:.2f}.txt"), np.array(joint_velocity))
        # np.savetxt(os.path.join(data_dir, f"trunk_position_{drop_pos:.2f}.txt"), np.array(trunk_position))
        # np.savetxt(os.path.join(data_dir, f"com_trunk_body_{drop_pos:.2f}.txt"), np.array(com_trunk_body))
        # np.savetxt(os.path.join(data_dir, f"muscle_length_{drop_pos:.2f}.txt"), np.array(muscle_length))
        # np.savetxt(os.path.join(data_dir, f"muscle_velocity_{drop_pos:.2f}.txt"), np.array(muscle_velocity))
        # np.savetxt(os.path.join(data_dir, f"muscle_activation_{drop_pos:.2f}.txt"), np.array(muscle_activation))
        # np.savetxt(os.path.join(data_dir, f"touch_sensor_{drop_pos:.2f}.txt"), np.array(touch_sensor))
        np.savetxt(os.path.join(data_dir, f"global_com_{drop_pos:.2f}.txt"), np.array(global_com))

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

        


  

        














