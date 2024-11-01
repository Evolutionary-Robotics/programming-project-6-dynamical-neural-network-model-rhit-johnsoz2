import numpy as np
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as ps
import time
import matplotlib.pyplot as plt
import ctrnn

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, rgbBackground=[0, 0, 50])
p.setAdditionalSearchPath(pybullet_data.getDataPath())

camera_distance = 12  # Adjust this value to zoom in/out
camera_pitch = -30  # Adjust the pitch (vertical angle)
camera_yaw = 40  # Adjust the yaw (horizontal angle)
camera_target = [0, 0, 1]  # The point the camera looks at (x, y, z)

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)


def elephant_simulation_CRNN(ctrnn_file):
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("elephant.urdf")

    # Debugging!
    # # Print the available link names
    # link_names = p.getBodyInfo(robot_id)
    # print()
    # print('Body Names:')
    # print(link_names)
    #
    # # Additionally, check the total number of links
    # num_links = p.getNumJoints(robot_id)
    # print(f"Total joints: {num_links}")
    #
    # print('Joints:')
    # for index in range(num_links):
    #     joint = p.getJointInfo(robot_id, index)
    #     print(joint)
    # print()

    duration = 2000
    num_legs = 4
    step_height = 0.5

    ps.Prepare_To_Simulate(robot_id)
    p.setJointMotorControlArray(robot_id,
                                range(num_legs),
                                p.POSITION_CONTROL,
                                targetPositions=[0] * num_legs,
                                forces=[500] * num_legs)

    nn = ctrnn.CTRNN(num_legs)
    nn.load(ctrnn_file)
    nn.initialize_state(np.zeros(num_legs))

    for i in range(duration):
        foot_sensors = [
            ps.Get_Touch_Sensor_Value_For_Link(f'1'),
            ps.Get_Touch_Sensor_Value_For_Link(f'2'),
            ps.Get_Touch_Sensor_Value_For_Link(f'3'),
            ps.Get_Touch_Sensor_Value_For_Link(f'4')
        ]

        # Prepare inputs for the CTRNN
        nn.inputs = np.array(foot_sensors)  # Sensor values as inputs to the network

        # Step the CTRNN and get outputs
        dt = 0.01  # Define your time step
        nn.step(dt)  # Update the CTRNN
        nn_outputs = nn.outputs  # Get the outputs from the CTRNN

        # Update robot movement based on NN output
        movement_sim_CRNN(num_legs, step_height, robot_id, nn_outputs)

        p.stepSimulation()
        # print(foot_sensors)

        time.sleep(1 / 500)

    p.disconnect()


def movement_sim_CRNN(num_legs, step_height, robot_id, nn_outputs):
    target_positions = []

    # Use NN outputs to control leg positions
    for j in range(num_legs):
        # Scale the output if needed, or directly use it
        target_positions.append(nn_outputs[j] * step_height)

    # Set joint positions for walking motion
    p.setJointMotorControlArray(robot_id,
                                range(num_legs),
                                p.POSITION_CONTROL,
                                targetPositions=target_positions,
                                forces=[500] * num_legs)


def elephant_race_simulation(ctrnn_files):
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    num_elephants = len(ctrnn_files)
    robot_ids = [p.loadURDF("elephant.urdf", basePosition=[0, i * 6, 0]) for i in range(num_elephants)]
    nns = []

    # Debugging!
    # for robot_id in robot_ids:
    #     print()
    #     print(f'Robot ID: f{robot_id}')
    #     # Print the available link names
    #     link_names = p.getBodyInfo(robot_id)
    #     print('Body Names:')
    #     print(link_names)
    #
    #     # Additionally, check the total number of links
    #     num_links = p.getNumJoints(robot_id)
    #     print(f"Total joints: {num_links}")
    #
    #     print('Joints:')
    #     for index in range(num_links):
    #         joint = p.getJointInfo(robot_id, index)
    #         print(joint)
    #     print()

    duration = 200000
    num_legs = 4
    step_height = 0.5

    for robot_id in robot_ids:
        ps.Prepare_To_Simulate(robot_id)
        p.setJointMotorControlArray(robot_id,
                                    range(num_legs),
                                    p.POSITION_CONTROL,
                                    targetPositions=[0] * num_legs,
                                    forces=[500] * num_legs)

    for ctrnn_file in ctrnn_files:
        nn = ctrnn.CTRNN(num_legs)
        nn.load(ctrnn_file)
        nn.initialize_state(np.zeros(num_legs))
        nns.append(nn)

    # To track movement
    initial_positions = [p.getBasePositionAndOrientation(robot_id)[0][0] for robot_id in robot_ids]
    distances_moved = [0] * num_elephants

    for i in range(duration):
        for idx, robot_id in enumerate(robot_ids):

            foot_sensors = [
                ps.Get_Touch_Sensor_Value_For_Link(f'1'),
                ps.Get_Touch_Sensor_Value_For_Link(f'2'),
                ps.Get_Touch_Sensor_Value_For_Link(f'3'),
                ps.Get_Touch_Sensor_Value_For_Link(f'4')
                # ps.Get_Touch_Sensor_Value_For_Link(f'Foot_{j + 1}') for j in range(num_legs)
            ]

            nns[idx].inputs = np.array(foot_sensors)

            dt = 0.01
            nns[idx].step(dt)
            nn_outputs = nns[idx].outputs
            movement_sim_CRNN(num_legs, step_height, robot_id, nn_outputs)

            p.stepSimulation()

            current_position = p.getBasePositionAndOrientation(robot_id)[0][0]
            distances_moved[idx] += current_position - initial_positions[idx]

        time.sleep(1 / 500)

    # Disconnect from simulation
    p.disconnect()
    plot_results(num_elephants, distances_moved)


def plot_results(num_elephants, distances_moved):
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_elephants), distances_moved, color='skyblue')
    plt.xlabel('Elephant Movement Type')
    plt.ylabel('Distance Moved in X-Direction')
    plt.title('Distance Moved by Each Elephant with Different Movement Patterns')
    plt.xticks(range(num_elephants), [f'Elephant {i + 1}' for i in range(num_elephants)])
    plt.grid(axis='y')
    plt.show()


# elephant_simulation_CRNN('ctrnn_3.npz')

nn_arr = ['ctrnn_1.npz', 'ctrnn_2.npz', 'ctrnn_3.npz']
elephant_race_simulation(nn_arr)


# References

def elephant_simulation_preCRNN():
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("elephant.urdf")

    duration = 3000
    num_legs = 4

    ps.Prepare_To_Simulate(robot_id)
    p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=[0] * num_legs,
                                forces=[500] * num_legs)

    step_height = 0.5

    for i in range(duration):
        movement_sim_preCRNN(num_legs, step_height, robot_id, i)

        p.stepSimulation()
        time.sleep(1 / 500)

    p.disconnect()


def movement_sim_preCRNN(num_legs, step_height, robot_id, i):
    stagger_time = 50
    target_positions = []

    # Generate walking motion for legs
    for j in range(num_legs):
        if j % 2 == 0:  # Left legs
            # Front left leg
            if j == 0:  # Front left leg
                target_positions.append(np.sin(i / 50) * step_height)
            else:  # Back left leg
                target_positions.append(np.sin((i - stagger_time) / 50) * step_height)
        else:  # Right legs
            # Front right leg
            if j == 1:  # Front right leg
                target_positions.append(-np.sin(i / 50) * step_height)
            else:  # Back right leg
                target_positions.append(-np.sin((i - stagger_time) / 50) * step_height)

    # Set joint positions for walking motion
    p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=target_positions,
                                forces=[500] * num_legs)
