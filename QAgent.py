import torch
import torch.nn.functional
import numpy as np
import math

class QAgent:
    def __init__(self, state_shape, actions, state_min_values=None, state_max_values=None, Q=None):
        # Set internal variable
        self.state_shape = state_shape.copy()
        self.actions = actions

        if state_min_values.all and state_max_values.all:
            self.state_min=state_min_values
            self.state_max=state_max_values
        else:
            self.state_min=torch.zeros(self.state_shape)
            self.state_max=self.state_shape-1
        state_shape.append(actions)

        # Load Q-Table from file if specified
        if Q:
            self.Q=torch.load(Q)
        else:
            self.Q = torch.ones(tuple(state_shape))

        # BQTable is the binary reduction of the Q-Table, showing the greedy policy obtained with the current Q-Table
        # BQTable has the same shape as Q-Table, BQTable is binary and for all states x, the sum of Q[x,:] equals 1
        self.BQTable = None
        # QTableWithCertainty has the sam shape as Q-Table, The QTable entry is 0 if the actions will not be chosen in
        # a given state under a greedy policy and otherwise it takes the Q value (certainty) of that action
        self.QTableWithCertainty = None

        # Random number generators
        self.epsRNG = np.random.default_rng()
        self.aRNG = np.random.default_rng()

    # Get discrete value of a given continuous data point
    @staticmethod
    def getDiscreteValue(x, min_val, max_val, step_no):
        step = (max_val - min_val) / float(step_no)
        index = math.floor((x - min_val) / step)
        if x==max_val:
            return index-1
        else:
            return index

    # Get the discretised version of the input state vector
    def discrete(self, state):
        newState = torch.zeros(len(self.state_shape), dtype=int)
        for i in range(len(self.state_shape)):
            newState[i] = QAgent.getDiscreteValue(state[i], self.state_min[i], self.state_max[i], self.state_shape[i])
        return newState

# Take a random action with probability eps. Otherwise, choose the action with the maximum Q-Value for the current state.
    def getAction(self, state, eps=0):  # state is a non negative integer tuple dimension equal to state space dimension
        rnd = self.epsRNG.random(1)
        if rnd < eps:
            # Random action
            action = self.aRNG.integers(self.actions)
        else:
            state = self.discrete(state)
            action = np.argmax(self.Q.numpy()[tuple(state)])
        return action

        """
        # Optimal policy for MountainCar-v0
        state=self.discrete(state)
        if state[1]<self.state_shape[1]/2:
            ac=0
        else:
            ac=2
        return ac"""

    # Do a training step
    def updateQ(self, prevState, prevAction, reward, newState, discount, lr=0.01):
        prevState=self.discrete(prevState)
        newAction=self.getAction(newState,0)
        newState=self.discrete(newState)
        self.Q[tuple(prevState) + tuple([prevAction])] += lr * (
                reward + discount * (self.Q[tuple(newState) + tuple([newAction])]) - self.Q[
            tuple(prevState) + tuple([prevAction])])
        return

    # Do a training step for a final action
    def updateQFinal(self, prevState, prevAction):
        prevState = self.discrete(prevState)
        self.Q[tuple(prevState)+tuple([prevAction])]=0
        return

    # Get BQTable, defined as:
    # Binary table such that for all state-action pairs (x,a) we get the probability of taking action a at state x, following the greedy policy (i.e. either 0 or 1)
    def getBinaryQTable(self, index=None):
        self.BQTable = self.Q.clone()
        flatQTable = self.BQTable.view(-1, self.actions)
        for i, elem in enumerate(flatQTable):
            j = torch.argmax(elem)
            newQ = torch.zeros(self.actions)
            newQ[j] = 1
            flatQTable[i] = newQ
        if index:
            return self.BQTable[tuple(index)]
        else:
            return self.BQTable

    # QTableWithCertainty has the sam shape as Q-Table, The QTable entry is 0 if the actions will not be chosen in
    # a given state under a greedy policy and otherwise it takes the Q value (certainty) of that action
    def getQTableWithCertainty(self, index=None):
        self.QTableWithCertainty = self.Q.clone()
        flatQTable = self.QTableWithCertainty.view(-1, self.actions)
        for i, elem in enumerate(flatQTable):
            certainties = torch.nn.functional.softmax(elem)
            j = torch.argmax(certainties)
            newQ = torch.zeros(self.actions)
            newQ[j] = certainties[j]
            print(i, newQ[j])
            flatQTable[i] = newQ
        if index:
            return self.QTableWithCertainty[tuple(index)]
        else:
            return self.QTableWithCertainty

    def getQTableWithCertaintyInterpolated(self, resolution):
        QTableWithCertainty = self.getQTableWithCertainty(self)
        QTableWithCertaintyInterpolated = torch.zeros(resolution * self.actions)
        for i in range(QTableWithCertaintyInterpolated.shape[0]):
            for j in range(QTableWithCertaintyInterpolated.shape[1]):
                i_ref, j_ref = int(i / resolution), int(j / resolution)
                for k in range(len(self.actions)):
                    s = (i - resolution * i_ref) / resolution
                    t = (j - resolution * j_ref) / resolution
                    QTableWithCertaintyInterpolated[i, j, k] = (1-t) * (1-s) * QTableWithCertainty[i_ref, j_ref, k] \
                                                               + (1-t) * s * QTableWithCertainty[i_ref+1, j_ref, k] \
                                                               + t * (1-s) * QTableWithCertainty[i_ref, j_ref+1, k] \
                                                               + t * s * QTableWithCertainty[i_ref+1, j_ref+1, k]
        return QTableWithCertaintyInterpolated
