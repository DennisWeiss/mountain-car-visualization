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
            if T[i,j]==1:
                i11.append(i)
                i21.append(j)
            else:
                i10.append(i)
                i20.append(j)
    return i10,i20,i11,i21

# Naming the model files as "q-<# of training episodes in thousands>k", we define the model numbers we want to load
ks=[0,1,2,3,4,5,6,7,8,9,10]
env = gym.make('MountainCar-v0')
# Number of episodes to simulate and frequency to show simulation before analysis
EPISODES=1
RENDER_EACH=1

foldername="eps02"
for k in ks:
    A=QAgent([15,15],3,env.observation_space.low,env.observation_space.high,Q=foldername+"/q-"+str(k)+"k")
    print(A.Q)
    for ep in range(EPISODES):
        X=env.reset() # X is the state var

        done=False
        while not done:
            if ep % RENDER_EACH == 0:
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

    #Visualise Q table for each action
    plt.figure(figsize=(6, 6), dpi=300)
    BQT=A.getBinaryQTable()
    A0=BQT[:,:,0]
    A1=BQT[:,:,1]
    A2=BQT[:,:,2]
    print(A0)
    # Blue means go left, green means go right, blank means do nothing
    i10,i20,i11,i21=getIndexes(A0)
    plt.scatter(i11,i21, c='b', s=100)
    i10,i20,i11,i21=getIndexes(A2)
    plt.scatter(i11,i21, c='g',s=200)
    #plt.show()
    plt.savefig(foldername+"/fig"+str(k)+"k.png")

    plt.close()
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
