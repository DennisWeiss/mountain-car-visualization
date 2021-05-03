import torch
import numpy as np
import gym
from time import sleep
import math
from QAgent import QAgent
import matplotlib.pyplot as plt



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

ks=[100]
env = gym.make('MountainCar-v0')
EPISODES=20
RENDER_EACH=1
for k in ks:
    A=QAgent([15,15],3,env.observation_space.low,env.observation_space.high,Q="aq"+str(k)+"k")
    print(A.Q)
    for ep in range(EPISODES):
        X=env.reset() # X is the state var

        done=False
        rew=0
        t=0
        while not done:
            t+=1
            if ep % RENDER_EACH == 0:
                sleep(0.05)
                env.render()
            ac=A.getAction(X)
            (newX, r, done, _) = env.step(ac)
            rew+=r
            X=newX
        if ep%RENDER_EACH==0:
            print("Episode "+str(RENDER_EACH*ep))
            print("R = "+str(float(r)/float(t)))

    env.close()

    #Visualise Q table for each action
    plt.figure(figsize=(6, 18), dpi=300)
    BQT=A.getBinaryQTable()
    A0=BQT[:,:,0]
    A1=BQT[:,:,1]
    A2=BQT[:,:,2]
    print(A0)

    plt.subplot(311)
    i10,i20,i11,i21=getIndexes(A0)
    plt.scatter(i10,i20, c='r', s=100)
    plt.scatter(i11,i21, c='g',s=200)

    plt.subplot(312)
    i10,i20,i11,i21=getIndexes(A1)
    plt.scatter(i10,i20, c='r', s=100)
    plt.scatter(i11,i21, c='g',s=200)

    plt.subplot(313)
    i10,i20,i11,i21=getIndexes(A2)
    plt.scatter(i10,i20, c='r', s=100)
    plt.scatter(i11,i21, c='g',s=200)

    #plt.show()
    plt.savefig("res"+str(k)+"k.png")

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