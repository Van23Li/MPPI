import numpy as np
from numpy import matlib as mb
# import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# clear('all')
# close_('all','force')
# addpath('../planar_robot_7d/')
# rng('default')

def get_norm_samples(MU_ARR, SIGMA, N_TRAJ):
    result = np.zeros([2,30,50])
    for i in range(2):
        for j in range(30):
            for k in range(50):
                result[i,j,k] = np.random.normal((mb.repmat(MU_ARR,1,N_TRAJ).reshape([2,30,50]))[i,j,k], SIGMA)
    return result

def get_rollout(pos_init, u_sampl, dT):
    # u_sampl_T = np.array([(u_sampl[:,:,i]).T for i in range(50)]).reshape(30,2,50)
    u_sampl_T = np.array([(u_sampl[:,:,i]).T for i in range(50)]).transpose(1,2,0)
    u_sampl_time = np.array([np.dot(dT * INT_MAT, u_sampl_T[:,:,i]) for i in range(50)]).transpose(1,2,0)
    pos_delta = np.array([(u_sampl_time[:,:,i]).T for i in range(50)]).transpose(1,2,0)
    result = np.array([pos_init + pos_delta[:,:,i] for i in range(50)]).transpose(1,2,0)
    return result




## Constants and Parameters
N_ITER = 1000
H = 30
D = 2
SIGMA = np.array([1,0.1])
N_POL = np.size(SIGMA)
MU_ARR = np.zeros((D,H,N_POL))

dT = 0.1
N_TRAJ = 50
pos_init = np.zeros((D,1))
pos_goal = pos_init + 6
gamma_vec = np.append(np.array([0.98 ** i for i in np.linspace(1,H - 1,H - 1)]), 1.02**H)

beta = 0.9
mu_alpha = 0.9

## lambdas
# get_norm_samples = lambda MU_ARR, SIGMA, N_TRAJ: np.random.multivariate_normal((mb.repmat(MU_ARR,1,N_TRAJ)).reshape([2,30,50]),SIGMA)
INT_MAT = np.tril(np.ones((H,H)))
# get_rollout = lambda pos_init, u_sampl, dT: pos_init + pagetranspose(pagemtimes(dT * INT_MAT,pagetranspose(u_sampl)))
calc_reaching_cost = lambda rollout, goal: np.array([np.linalg.norm((rollout[:,:,i] - goal),ord=2,axis=0) for i in range(50)])
obs_dist = lambda rollout, obs_pos, obs_r: np.transpose(np.squeeze(vecnorm(rollout - obs_pos,2,1))) - obs_r
w_fun = lambda cost: np.exp(- 1 / beta * np.sum(np.multiply(gamma_vec,cost), 1))
#mu_upd_fun = @(mu, w, u) (1-mu_alpha)*mu +

## plot preparation
# plt.ion()
fig, ax = plt.subplots(figsize = (6, 6))
ax.set_xlim([-1, 10])
ax.set_ylim([-1, 10])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('MPPI example')
ax.scatter(pos_init[0],pos_init[1], color = [0, 0, 1], marker='*')
ax.scatter(pos_goal[0],pos_goal[1], color = [1, 0, 0], marker='*')
# r_h_arr = np.zeros((N_POL,N_TRAJ))
r_h_arr = []
r_h_arr_list = []
for i in range(N_POL):
    for j in range(N_TRAJ):
        color = np.array([1 - 1 / (i + 1), 1 - 1 / (i + 1), 1 / (i + 1), 0.5])
        # r_h = plt.scatter(np.zeros((1,H)), np.zeros((1,H)), color = color)
        r_h, = ax.plot([],[], color = color)
        r_h_arr_list.append(r_h)
    r_h_arr.append(r_h_arr_list)
best_traj_h, = ax.plot([],[], color = np.array([0,1,0.5,1]), linewidth = 2)
cur_pos_h, = ax.plot([],[],'r*')

## MPPI
cur_pos = pos_init
cur_vel = pos_init * 0
def mppi(num, cur_pos, cur_vel):
    # for i in range(N_ITER):
    best_cost_iter = 10000000000.0
    for j in range(N_POL):
        u = get_norm_samples(MU_ARR[:,:,j],SIGMA[j],N_TRAJ)
        u[:,:,-1] = u[:,:,-1] * 0

        #inject slowdown (for acceleration control)
        ss = 5
        u[:, 0:ss, -1] = u[:, 0:ss, -1] - cur_vel / dT / ss

        v_rollout = get_rollout(cur_vel,u,dT)
        rollout = get_rollout(cur_pos,v_rollout,dT)
        cost_p = calc_reaching_cost(rollout,pos_goal)
        cost_v = 0 * calc_reaching_cost(v_rollout,cur_vel * 0)
        #d1 = obs_dist(rollout, [3; 3], 2);
#d1(d1>0) = 0;
#d1(d1<0) = 500;
        cost = cost_p + cost_v
        w = w_fun(cost)
        w = w / sum(w)
        best_cost = np.max(w)
        best_idx = np.argmax(w)
        w_tens = np.expand_dims(np.transpose((mb.repmat(w, 1, H)).reshape([50, H])),0)
        MU_ARR[:,:,j] = (1 - mu_alpha) * MU_ARR[:,:,j] + mu_alpha * np.sum(np.multiply(w_tens,u), 2)
        for i_traj in range(N_TRAJ):
            h_tmp = r_h_arr[j][i_traj]
            h_tmp.set_data(rollout[0,:,i_traj], rollout[1,:,i_traj])
        if best_cost < best_cost_iter:
            cur_pos_temp = rollout[:,0,best_idx]
            cur_pos = np.expand_dims(cur_pos_temp, axis=1)
            cur_vel_temp = v_rollout[:,0,best_idx]
            cur_vel = np.expand_dims(cur_vel_temp, axis=1)
            cur_pos_h.set_data(cur_pos[0], cur_pos[1])
            best_cost_iter = best_cost
            best_traj_h.set_data(rollout[0,:,best_idx], rollout[1,:,best_idx])
    return cur_pos, cur_vel, h_tmp, cur_pos_h, best_traj_h

# Creating the Animation object
ani = animation.FuncAnimation(
    fig=fig, func=mppi, frames=range(1000), fargs=(cur_pos, cur_vel), interval=100)

plt.show()