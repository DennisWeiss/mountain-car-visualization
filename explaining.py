import torch
import numpy as np
import gym
from time import sleep
import math
from QAgent import QAgent
import matplotlib.pyplot as plt


# Get the indices for zeroes and ones of T seperately to plot with different colors
def getIndexes(T):
    s=T.shape
    i11=[]
    i10=[]
    i21=[]
    i20=[]
    for i in range(s[0]):
        for j in range(s[1]):
            if T[i,j] > 0:
                i11.append(i)
                i21.append(j)
            else:
                i10.append(i)
                i20.append(j)
    return i10,i20,i11,i21


def getCertainties(T):
    s = T.shape
    certainties = []
    for i in range(s[0]):
        for j in range(s[1]):
            if T[i, j] > 0:
                certainties.append(400 * T[i, j] ** 3)
                # certainties.append(i + j)
                print(1 - T[i, j])
    print(certainties)
    return certainties


def visualizeBQTable(A):
    plt.figure(figsize=(6, 6), dpi=300)
    BQT = A.getBinaryQTable()
    A0 = BQT[:, :, 0]
    A1 = BQT[:, :, 1]
    A2 = BQT[:, :, 2]
    print(A0)
    # Blue means go left, green means go right, blank means do nothing
    i10, i20, i11, i21 = getIndexes(A0)
    plt.scatter(i11, i21, c='b', s=200)
    i10, i20, i11, i21 = getIndexes(A2)
    plt.scatter(i11, i21, c='g', s=200)
    plt.show()
    plt.savefig(foldername + "/fig" + str(k) + "k.png")

    plt.close()


def visualizeQTableWithCertainty(A):
    plt.figure(figsize=(6, 6), dpi=300)
    QTWithCertainty = A.getQTableWithCertainty()
    A0 = QTWithCertainty[:, :, 0]
    A1 = QTWithCertainty[:, :, 1]
    A2 = QTWithCertainty[:, :, 2]
    i10, i20, i11, i21 = getIndexes(A0)
    plt.scatter(i11, i21, c='b', s=getCertainties(A0))
    i10, i20, i11, i21 = getIndexes(A2)
    plt.scatter(i11, i21, c='g', s=getCertainties(A2))
    plt.show()
    plt.savefig(foldername + "/figWithCertainty" + str(k) + "k.png")

    plt.close()


def visualizeQTableWithCertaintyInterpolated(A, resolution):
    QWithCertaintyInterpolated = A.getQTableWithCertaintyInterpolated(resolution)
    A0 = QWithCertaintyInterpolated[:, :, 0]
    A1 = QWithCertaintyInterpolated[:, :, 1]
    A2 = QWithCertaintyInterpolated[:, :, 2]
    plt.figure(figsize=(6, 6), dpi=300)
    plt.contourf(A0, cmap='Blues')
    plt.show()
    plt.close()
    plt.figure(figsize=(6, 6), dpi=300)
    plt.contourf(A1, cmap='Reds')
    plt.show()
    plt.close()
    plt.figure(figsize=(6, 6), dpi=300)
    plt.contourf(A2, cmap='Greens')
    plt.show()
    plt.close()



# Naming the model files as "q-<# of training episodes in thousands>k", we define the model numbers we want to load
ks=[0,1,2,3,4,5,6,7,8,9,10]
env = gym.make('MountainCar-v0')
# Number of episodes to simulate and frequency to show simulation before analysis
EPISODES=1
RENDER_EACH=1

foldername="eps02"
for k in [10]:
    A=QAgent([15,15],3,env.observation_space.low,env.observation_space.high,Q=foldername+"/q-"+str(k)+"k")
    print(A.Q)
    for ep in range(EPISODES):
        X=env.reset() # X is the state var

        done=False
        while not done:
            if ep % RENDER_EACH == 0:
                pass
                sleep(0.05)
                env.render()
            ac=A.getAction(X)
            (newX, r, done, i) = env.step(ac)
            X=newX
        if ep%RENDER_EACH==0:
            print("Episode "+str(RENDER_EACH*ep))
            if(r==0.0):
                print("Success! ",end='')
            else:
                print("Failure... ", end='')

    env.close()

    visualizeBQTable(A)
    visualizeQTableWithCertainty(A)
    visualizeQTableWithCertaintyInterpolated(A, 10)

"""
A=QAgent([18,14],[0,1,2],env.observation_space.low,env.observation_space.high)
for i in range(1000):
    X=env.reset()
    done=False
    while not done:
        env.render()
        sleep(0.01)
        pass
        ac=A.getAction(X)
        (X, r, done, _) = env.step(ac)
        #print("Action: "+str(ac)+"Reward: "+str(r))
        if done:
            print("Final X: "+str(X[0]))
env.close()"""
