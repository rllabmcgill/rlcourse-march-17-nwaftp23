import numpy as np
import matplotlib.pyplot as plt
import copy as cp

# reward dictionary
#rewards={'-':-1,'St':-1,'C':-100,'G':0,'P':-1}
#rewards={'-':-1,'St':-1,'C':-100,'G':1000,'P':-1,'W':-100}


# takes an array Q 1xActions and epsilson
# returns an epsilon greedy policy array
# Given Q and state S a policy vector
def policy(Q,e):
    pol = list(range(4))
    W = np.argwhere(Q == np.amax(Q))
    ind = [i for i in pol if i not in list(W[:,0])]
    if ind == []:
         pol = [1/len(pol)]*len(pol)
    else:
        for inde in ind:
            pol[inde]=e/len(ind)
        for i in W:
            pol[i[0]]=(1-e)/len(W)
    return np.array(pol)

# selects an action based on policy
def select_action(S,e,A,Q,G,L,W):
    pol=policy(Q[S,:],e)
    Sg = convert_st(S,L,W)
    if G[Sg] == 'C':
        return 'C'
    else:
        c =np.random.choice(len(A),p=list(pol))
    return A[c]

# takes the action on the gridworld
def take_action(n,S,G,L,W):
    Sg = convert_st(S,L,W)
    if G[Sg] == 'C':
        new_state = [3,0]
    elif n == 'R':
        if Sg[1]==11:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([0,1])
    elif n == 'L':
        if Sg[1]==0:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([0,-1])
    elif n == 'U':
        if Sg[0]==0:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([-1,0])
    else:
        if Sg[0]==3:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([1,0])
    if G[tuple(new_state)]=='W':
        new_state = S
    else:
        new_state = (W-new_state[0])*L+new_state[1]
    return new_state

# convert number to grid cell
def convert_st(S,L,W):
    col = S % L
    g = (int(np.absolute((S-col)/L-W)),col)
    return g

# convert action i.e. U to a number
def conver_act(A):
    if A == 'C':
        a=0
    elif A == 'R':
        a=0
    elif A == 'L':
        a=1
    elif A == 'U':
        a=2
    else:
        a=3
    return a

# show path creates path on grid
def show_path(S,CG,RG, L, W):
    Sg=convert_st(S , L, W)
    if Sg[0] == 3 and Sg[1]>0:
        return np.copy(RG)
    else:
        G=np.copy(CG)
        G[Sg]='P'
        return G

# runs sarsa returns Q matrix
# need to check
def sarsa(Q , eps, alpa, S, CG , act, L, W,rewards):
    A = select_action(S,eps,act,Q,CG,L,W)
    Ac = conver_act(A)
    sum_R=0
    #RG = np.copy(CG)
    #CCG = np.copy(CG)
    while CG[convert_st(S,L,W)] != 'G':
        S_prime = take_action(A,S,CG,L,W)
        #CCG = show_path(S_prime,CCG,RG,L,W)
        #print(CCG)
        if S_prime == S:
            R = rewards['W']
        else:
            R = rewards[CG[convert_st(S_prime,L,W)]]
        sum_R += R
        A_prime = select_action(S_prime,eps,act,Q,CG,L,W)
        Ac_prime = conver_act(A_prime)
        Q[S,Ac] += alpa*(R+Q[S_prime,Ac_prime]-Q[S,Ac])
        S = cp.copy(S_prime)
        A = cp.copy(A_prime)
        Ac = conver_act(A)
    return Q , sum_R

# runs expected sarsa returns Q matrix
# need to check
def exp_sarsa(Q , eps, alpa, S, CG , act, L, W,rewards):
    sum_R=0
    #RG = np.copy(CG)
    #CCG = np.copy(CG)
    while CG[convert_st(S,L,W)] != 'G':
        A = select_action(S,eps,act,Q,CG,L,W)
        AA=conver_act(A)
        S_prime = take_action(A,S,CG,L,W)
        #CCG = show_path(S_prime,CCG,RG,L,W)
        #print(CCG)
        if S_prime == S:
            R = rewards['W']
        else:
            R = rewards[CG[convert_st(S_prime,L,W)]]
        sum_R += R
        Q[S,AA] += alpa*(R+np.dot(policy(Q[S_prime,:],eps), Q[S_prime,:])-Q[S,AA])
        S = cp.copy(S_prime)
    return Q , sum_R


# compare the two methods with cliff world to verify
# then design experiments to show variance
