from Exp_v_Sarsa import *
import time


# N is the number of episodes
# e is epsilon
# S_0 is the start
# M is the number of runs
N = 100
eps = .1
alpha = .6
S_0 = 0
M = 50

# Cliff Grid World
Grid=np.array([
    ['-','-','-','W','W','W'],
    ['-','-','-','-','-','G'],
    ['-','-','-','W','W','W'],
    ['St','-','-','W','W','W']
])

# S is the number of states
# A is the set of actions
S=np.size(Grid)
LL=np.shape(Grid)[1]
W1=np.shape(Grid)[0]-1
actions=['R','L','U','D']

# Rewards
#rewards={'-':-1,'St':-1,'C':-100,'G':0,'P':-1}
rewards={'-':-1,'St':-1,'C':-100,'G':1000,'P':-1,'W':-1}

w=[-1*10**(x*.5) for x in range(0,7)]


# Q matrix is SxA
Q=np.zeros((S,len(actions)))

np.random.seed(3)
L_exp_avgs=[]
L_sar_avgs=[]
for ww in w:
    print('Current penalty' , ww)
    rewards['W']=ww
    L_exp_avgs_r=[]
    L_sar_avgs_r=[]
    for m in range(M):
        run_tot_exp=0
        run_tot_sar=0
        print('Current run is' , m+1)
        start = time.time()
        Qn=np.copy(Q)
        Qn1=np.copy(Q)
        for u in range(N):
            L=exp_sarsa(Qn , eps, alpha, S_0, Grid, actions, LL, W1,rewards)
            L1=sarsa(Qn1 , eps, alpha, S_0, Grid, actions, LL, W1,rewards)
            Qn=L[0]
            run_tot_exp += L[1]
            Qn1=L1[0]
            run_tot_sar += L1[1]
        L_exp_avgs_r.append(run_tot_exp/N)
        L_sar_avgs_r.append(run_tot_sar/N)
        end = time.time()
        print('Approx run time in secs' , len(w)*M*(end - start))
        print(run_tot_exp/N)
        print(run_tot_sar/N)
    L_exp_avgs.append(np.mean(L_exp_avgs_r))
    L_sar_avgs.append(np.mean(L_sar_avgs_r))

print(L_exp_avgs)
print(L_sar_avgs)

print(np.array(L_exp_avgs)-np.array(L_sar_avgs))

w_plus=list(map(abs,w))
line1=plt.semilogx(w_plus,L_exp_avgs,'b-', label='Exp_Sarsa')
line2 = plt.semilogx(w_plus,L_sar_avgs,'r-', label='Sarsa')
plt.xlabel('rewards')
plt.ylabel('Average_Return')
plt.legend()
plt.show()
