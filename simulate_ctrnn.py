import ctrnn
import matplotlib.pyplot as plt
import numpy as np

# Parameters
size = 4
duration = 100
step_size = 0.01

# Data
time = np.arange(0.0, duration, step_size)
outputs = np.zeros((len(time), size))
states = np.zeros((len(time), size))

# Initialization
nn = ctrnn.CTRNN(size)

# Neural parameters written by hand
# 1 -> 1, weight of 5.422
# 1 -> 2, weight of -0.24... etc
#
# 1 time constraint = -4.108
# w = [[5.422, -0.24, 0.535, 0.123],
#      [-0.018, 4.59, -2.25, -0.123],
#      [2.75, 1.21, 3.885, 3.212],
#      [1.23, -2.123, 2.324, 0.3423]]
# b = [-4.108, -2.787, -1.114, -3.232]
# t = [1, 2.5, 1, 1.5]
# nn.set_parameters(w, b, t)
# nn.initialize_state(np.array([4.0, 2.0, 1.0, 0.5]))

# Neural parameters at random
nn.randomize_parameters()

# Initialization at zeros or random
nn.initialize_state(np.zeros(size))
# nn.initialize_state(np.random.random(size=size) * 20 - 10)

# Run simulation
step = 0
for t in time:
    nn.step(step_size)
    # states[step] = nn.states
    outputs[step] = nn.outputs
    step += 1

# How much is the neural activity changing over time
threshold = 1.0
activity = np.count_nonzero(np.sum(np.abs(np.diff(outputs, axis=0) > threshold) / size))
print("Overall activity: ", activity)

# Plot activity
plt.plot(time, outputs)
plt.xlabel("Time")
plt.ylabel("Outputs")
plt.title("Neural output activity")
plt.show()

# Plot activity
# plt.plot(time, states)
# plt.xlabel("Time")
# plt.ylabel("States")
# plt.title("Neural state activity")
# plt.show()

# Save CTRNN parameters for later
nn.save("ctrnn_3")
