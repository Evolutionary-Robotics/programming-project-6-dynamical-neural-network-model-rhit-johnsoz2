#  def elephant_simulation_1():
#     p.setGravity(0, 0, -9.8)
#     p.loadURDF("plane.urdf")
#     robot_id = p.loadURDF("elephant.urdf")
#
#     duration = 3000
#     num_legs = 4
#
#     ps.Prepare_To_Simulate(robot_id)
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=[0] * num_legs,
#                                 forces=[500] * num_legs)
#
#     steps = 10
#     step_height = 0.5
#
#     for i in range(duration):
#         movement_sim_1(num_legs, steps, step_height, robot_id, i)
#
#         p.stepSimulation()
#         time.sleep(1 / 500)
#
#     p.disconnect()
#
#
# def movement_sim_1(num_legs, steps, step_height, robot_id, i):
#     target_positions = []
#
#     # Generate walking motion for legs
#     for j in range(num_legs):
#         if j % 2 == 0:  # Alternate leg motion
#             target_positions.append(np.sin((i + j * steps) / 50) * step_height)
#         else:
#             target_positions.append(-np.sin((i + j * steps) / 50) * step_height)
#
#         # Set joint positions for walking motion
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=target_positions,
#                                 forces=[500] * num_legs)
#
#
# def elephant_simulation_2():
#     p.setGravity(0, 0, -9.8)
#     p.loadURDF("plane.urdf")
#     robot_id = p.loadURDF("elephant.urdf")
#
#     duration = 3000
#     num_legs = 4
#
#     ps.Prepare_To_Simulate(robot_id)
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=[0] * num_legs,
#                                 forces=[500] * num_legs)
#
#     step_height = 0.5
#
#     for i in range(duration):
#         movement_sim_2(num_legs, step_height, robot_id, i)
#
#         p.stepSimulation()
#         time.sleep(1 / 500)
#
#     p.disconnect()
#
#
# def movement_sim_2(num_legs, step_height, robot_id, i):
#     target_positions = [0] * num_legs  # Initialize target positions
#
#     # Calculate the sinusoidal position for smooth movement
#     forward_position = np.sin(i / 50) * step_height  # Forward movement
#     backward_position = -np.sin(i / 50) * step_height  # Backward movement
#
#     # Apply the coordinated movement pattern
#     target_positions[0] = forward_position  # Front left leg forward
#     target_positions[1] = backward_position  # Front right leg backward
#     target_positions[2] = backward_position  # Back left leg backward
#     target_positions[3] = forward_position  # Back right leg forward
#
#     # Set joint positions for walking motion
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=target_positions,
#                                 forces=[500] * num_legs)

# def elephant_simulation_3():
#     p.setGravity(0, 0, -9.8)
#     p.loadURDF("plane.urdf")
#     robot_id = p.loadURDF("elephant.urdf")
#
#     duration = 3000
#     num_legs = 4
#
#     ps.Prepare_To_Simulate(robot_id)
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=[0] * num_legs,
#                                 forces=[500] * num_legs)
#
#     step_height = 0.5
#
#     for i in range(duration):
#         movement_sim_3(num_legs, step_height, robot_id, i)
#
#         p.stepSimulation()
#         time.sleep(1 / 500)
#
#     p.disconnect()
#
#
# def movement_sim_3(num_legs, step_height, robot_id, i):
#     stagger_time = 50
#     target_positions = []
#
#     # Generate walking motion for legs
#     for j in range(num_legs):
#         if j % 2 == 0:  # Left legs
#             # Front left leg
#             if j == 0:  # Front left leg
#                 target_positions.append(np.sin(i / 50) * step_height)
#             else:  # Back left leg
#                 target_positions.append(np.sin((i - stagger_time) / 50) * step_height)
#         else:  # Right legs
#             # Front right leg
#             if j == 1:  # Front right leg
#                 target_positions.append(-np.sin(i / 50) * step_height)
#             else:  # Back right leg
#                 target_positions.append(-np.sin((i - stagger_time) / 50) * step_height)
#
#     # Set joint positions for walking motion
#     p.setJointMotorControlArray(robot_id, range(num_legs), p.POSITION_CONTROL, targetPositions=target_positions,
#                                 forces=[500] * num_legs)
#

# def elephant_race_simulation():
#     p.setGravity(0, 0, -9.8)
#     p.loadURDF("plane.urdf")
#
#     # Create multiple elephants with different movement patterns
#     num_elephants = 3
#     robot_ids = [p.loadURDF("elephant.urdf", basePosition=[0, i * 6, 0]) for i in range(num_elephants)]
#
#     duration = 3000
#     num_legs = 4
#     steps = 10
#     step_height = 0.5
#
#     # To track movement
#     initial_positions = [p.getBasePositionAndOrientation(robot_id)[0][0] for robot_id in robot_ids]
#     distances_moved = [0] * num_elephants
#
#     for i in range(duration):
#         for idx, robot_id in enumerate(robot_ids):
#
#             if idx == 0:
#                 movement_sim_1(num_legs, steps, step_height, robot_id, i)
#             elif idx == 1:
#                 movement_sim_2(num_legs, step_height, robot_id, i)
#             elif idx == 2:
#                 movement_sim_3(num_legs, step_height, robot_id, i)
#
#             # Step the simulation
#             p.stepSimulation()
#
#             # Update distance moved
#             current_position = p.getBasePositionAndOrientation(robot_id)[0][0]
#             distances_moved[idx] += current_position - initial_positions[idx]
#
#         time.sleep(1 / 500)
#
#     # Disconnect from simulation
#     p.disconnect()
#     plot_results(num_elephants, distances_moved)
#
#
# def plot_results(num_elephants, distances_moved):
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(num_elephants), distances_moved, color='skyblue')
#     plt.xlabel('Elephant Movement Type')
#     plt.ylabel('Distance Moved in X-Direction')
#     plt.title('Distance Moved by Each Elephant with Different Movement Patterns')
#     plt.xticks(range(num_elephants), [f'Elephant {i + 1}' for i in range(num_elephants)])
#     plt.grid(axis='y')
#     plt.show()
#