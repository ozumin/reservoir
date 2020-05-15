import numpy as np
from scipy import linalg
import math


class Reservoir:
    def __init__(self, data, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate = 0.1, activator = np.tanh):
        self.data = data
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.leak_rate = leak_rate
        self.activator = activator

        # initialize weight
        self.Win = (np.random.randint(0, 2, self.num_input_nodes * self.num_reservoir_nodes).reshape([self.num_input_nodes, self.num_reservoir_nodes]) * 2 - 1) * 0.1 # 0.1か-0.1の行列
        self.Wres = self._initialize_Wres(num_reservoir_nodes)
        self.Wout = np.zeros([num_output_nodes, num_reservoir_nodes])
        self.X = np.array([np.zeros(num_reservoir_nodes)])

    def get_output(self, X):
        return self.Wout @ X[-1]

    def update_reservoir_nodes(self, u):
        return self.activator((1 - self.leak_rate) * self.X[-1] + self.leak_rate * (self.Win * u + self.Wres @ self.X[-1]))

    def get_all_time_reservoir_nodes(self, u):
        new_x = self.update_reservoir_nodes(u)
        self.X = np.append(self.X, new_x, axis=0)

    def update_Wout(self, Y, lam = 0.1):
        self.Wout = (np.linalg.inv(self.X.T @ self.X + lam * np.eye(self.num_reservoir_nodes)) @ self.X.T) @ Y

    def train(self, Y):
        for d in self.data:
            self.get_all_time_reservoir_nodes(d)
        self.X = self.X[1:]
        self.update_Wout(Y)

    def predict(self, data):
        output = []
        for d in data:
            x = self.update_reservoir_nodes(d)
            output.append(self.get_output(x))
        return output

    def _initialize_Wres(self, num_reservoir_nodes):
        tmp = np.random.normal(0, 1, num_reservoir_nodes * num_reservoir_nodes).reshape([num_reservoir_nodes, num_reservoir_nodes])
        tmp /= max(abs(linalg.eigvals(tmp))) # 正規分布
        return tmp
