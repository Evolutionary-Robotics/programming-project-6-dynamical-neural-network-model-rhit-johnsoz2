import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CTRNN:

    def __init__(self, size):
        self.size = size  # number of neurons in the circuit
        self.states = np.zeros(size)  # state of the neurons
        self.time_constants = np.ones(size)  # time-constant for each neuron
        self.inv_time_constants = 1.0 / self.time_constants
        self.biases = np.zeros(size)  # bias for each neuron
        self.weights = np.zeros((size, size))  # connection weight for each pair of neurons
        self.outputs = np.zeros(size)  # neuron outputs
        self.inputs = np.zeros(size)  # external input to each neuron
        self.time_step = 0.0

    def set_time_constants(self, value):
        self.time_constants = value
        self.inv_time_constants = 1.0 / self.time_constants

    def set_parameters(self, weights, biases, time_constants):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.set_time_constants(np.array(time_constants))

    def randomize_parameters(self):
        self.weights = np.random.uniform(-10, 10, size=(self.size, self.size))
        self.biases = np.random.uniform(-10, 10, size=self.size)
        self.set_time_constants(np.random.uniform(0.1, 5.0, size=self.size))

    def initialize_state(self, s):
        self.inputs = np.zeros(self.size)
        self.states = s
        self.outputs = sigmoid(self.states + self.biases)

    def step(self, dt):
        net_input = self.inputs + np.dot(self.weights.T, self.outputs)
        self.states += dt * (self.inv_time_constants * (-self.states + net_input))
        self.outputs = sigmoid(self.states + self.biases)

        self.outputs = np.clip(self.outputs, 0, 1)

        phase_offset = np.linspace(0, np.pi, self.size)  # Offset for each leg
        self.outputs += 0.5 * np.sin(2 * np.pi * (self.time_step * 0.1) + phase_offset)  # Create wave pattern

        self.outputs = np.clip(self.outputs, 0, 1)
        self.time_step += dt

    def save(self, filename):
        np.savez(filename, size=self.size, weights=self.weights, biases=self.biases, time_constants=self.time_constants)

    def load(self, filename):
        params = np.load(filename)
        self.size = params['size']
        self.weights = params['weights']
        self.biases = params['biases']
        self.set_time_constants(params['time_constants'])
