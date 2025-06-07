
import mujoco 
import mujoco.viewer
import numpy as np
import time 
import os   


####### Load the MuJoCo model 

path_to_model = '../Working_Folder/single_leg_experiment/single_leg.xml'


####### Define muscle activations

M0 = 0.17    
M1 = 0.35    
M2 = 0.23
muscle_activations = np.array([M0, M1, M2])


QPOS = [-0.014,-0.069, 0.05,-0.2]

drop_heights = np.round(np.arange(-0.05, 1.0, 0.05), 2) #.nparange(-0.05, 1.1, 0.1) 

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

        muscle_length = []
        muscle_velocity = []

        muscle_activation = []
        touch_sensor = []



        print(f"Drop height: {drop_pos}")

        mujoco.mj_resetData(model, data)
        rootz_joint_id = model.joint("rootz").id

        sensor_id = model.sensor("rb_sensor").id
        touchdown_cond = None
    
        data.qpos[:] = [-0.014,drop_pos,0.05,-0.2]
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        viewer.sync()
      

        while viewer.is_running():
            start_time = time.time()

            data.ctrl[:] = muscle_activations
            mujoco.mj_step(model, data)
            viewer.sync()
            
        
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

            joint_position.append(data.qpos.copy())
            joint_velocity.append(data.qvel.copy())
            com_trunk_body.append(data.xipos[rootz_joint_id].copy())
            trunk_position.append(data.xpos[rootz_joint_id].copy())
            muscle_length.append(data.actuator_length.copy())
            muscle_velocity.append(data.actuator_velocity.copy())
            muscle_activation.append(data.ctrl.copy())
            touch_sensor.append(data.sensordata.copy())
            
        
        data_dir = '../all_data/single_leg_experiment/no_feedback'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)  

        np.savetxt(os.path.join(data_dir, f"joint_position_{drop_pos:.2f}.txt"), np.array(joint_position))
        np.savetxt(os.path.join(data_dir, f"joint_velocity_{drop_pos:.2f}.txt"), np.array(joint_velocity))
        np.savetxt(os.path.join(data_dir, f"trunk_position_{drop_pos:.2f}.txt"), np.array(trunk_position))
        np.savetxt(os.path.join(data_dir, f"com_trunk_body_{drop_pos:.2f}.txt"), np.array(com_trunk_body))
        np.savetxt(os.path.join(data_dir, f"muscle_length_{drop_pos:.2f}.txt"), np.array(muscle_length))
        np.savetxt(os.path.join(data_dir, f"muscle_velocity_{drop_pos:.2f}.txt"), np.array(muscle_velocity))
        np.savetxt(os.path.join(data_dir, f"muscle_activation_{drop_pos:.2f}.txt"), np.array(muscle_activation))
        np.savetxt(os.path.join(data_dir, f"touch_sensor_{drop_pos:.2f}.txt"), np.array(touch_sensor))


        print(f"Data saved to {data_dir}")

        


  

        














