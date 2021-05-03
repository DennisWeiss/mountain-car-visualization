import torch
import numpy as np
import math

class QAgent:
    def __init__(self, state_shape, actions, state_min_values=None, state_max_values=None, Q=None):
        self.state_shape = state_shape.copy()
        self.actions = actions

        if state_min_values.all and state_max_values.all:
            self.state_min=state_min_values
            self.state_max=state_max_values
        else:
            self.state_min=torch.zeros(self.state_shape)
            self.state_max=self.state_shape-1
        state_shape.append(actions)
        if Q:
            self.Q=torch.load(Q)
        else:
            self.Q = torch.ones(tuple(state_shape))

        self.BQTable = None
        self.epsRNG = np.random.default_rng()
        self.aRNG = np.random.default_rng()

    @staticmethod
    def getDiscreteValue(x, min_val, max_val, step_no):
        step = (max_val - min_val) / float(step_no)
        index = math.floor((x - min_val) / step)
        if x==max_val:
            return index-1
        else:
            return index

    def discrete(self, state):
        newState = torch.zeros(len(self.state_shape), dtype=int)
        for i in range(len(self.state_shape)):
            newState[i] = QAgent.getDiscreteValue(state[i], self.state_min[i], self.state_max[i], self.state_shape[i])
        return newState

    def getAction(self, state, eps=0):  # state is a non negative integer tuple dimension equal to state space dimension
        rnd = self.epsRNG.random(1)
        if rnd < eps:
            # Random action
            action = self.aRNG.integers(self.actions)
        else:
            state = self.discrete(state)
            action = np.argmax(self.Q.numpy()[tuple(state)])
        return action

        """state=self.discrete(state)
        if state[1]<self.state_shape[1]/2:
            ac=0
        else:
            ac=2
        return ac"""

    def updateQ(self, prevState, prevAction, reward, newState, discount, lr=0.01):
        prevState=self.discrete(prevState)
        newAction=self.getAction(newState,0)
        newState=self.discrete(newState)
        self.Q[tuple(prevState) + tuple([prevAction])] += lr * (
                reward + discount * (self.Q[tuple(newState) + tuple([newAction])]) - self.Q[
            tuple(prevState) + tuple([prevAction])])
        return

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