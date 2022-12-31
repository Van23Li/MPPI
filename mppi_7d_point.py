import numpy as np
from numpy import matlib as mb
import time
# import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# clear('all')
# close_('all','force')
# addpath('../planar_robot_7d/')
# rng('default')
class MPPI:
    def __init__(self):
        # self.ax = ax
        ## Constants and Parameters
        self.D = 2
        self.H = 30
        self.SIGMA = np.array([1,0.1])
        self.N_POL = np.size(self.SIGMA)
        self.MU_ARR = np.zeros((self.D,self.H,self.N_POL))

        self.N_TRAJ = 50
        pos_init = np.zeros((self.D,1))
        self.pos_goal = pos_init + 6
        gamma_vec = np.append(np.array([0.98 ** i for i in np.linspace(1,self.H - 1,self.H - 1)]), 1.02**self.H)

        beta = 0.9

        ## lambdas
        # get_norm_samples = lambda MU_ARR, SIGMA, N_TRAJ: np.random.multivariate_normal((mb.repmat(MU_ARR,1,N_TRAJ)).reshape([2,30,50]),SIGMA)
        self.INT_MAT = np.tril(np.ones((self.H,self.H)))
        # get_rollout = lambda pos_init, u_sampl, dT: pos_init + pagetranspose(pagemtimes(dT * INT_MAT,pagetranspose(u_sampl)))
        self.calc_reaching_cost = lambda rollout, goal: np.array([np.linalg.norm((rollout[:,:,i] - goal),ord=2,axis=0) for i in range(50)])
        self.obs_dist = lambda rollout, obs_pos, obs_r: np.transpose(np.squeeze(vecnorm(rollout - obs_pos,2,1))) - obs_r
        self.w_fun = lambda cost: np.exp(- 1 / beta * np.sum(np.multiply(gamma_vec,cost), 1))
        #mu_upd_fun = @(mu, w, u) (1-mu_alpha)*mu +

        ## plot preparation
        # plt.ion()
        # self.ax.set_xlim([-1, 10])
        # self.ax.set_ylim([-1, 10])
        # self.ax.set_xlabel('x1')
        # self.ax.set_ylabel('x2')
        # self.ax.set_title('MPPI example')
        # self.ax.scatter(pos_init[0],pos_init[1], color = [0, 0, 1], marker='*')
        # self.ax.scatter(self.pos_goal[0],self.pos_goal[1], color = [1, 0, 0], marker='*')
        # r_h_arr = np.zeros((N_POL,N_TRAJ))
        # self.r_h_arr = []
        # r_h_arr_list = []
        # for i in range(self.N_POL):
        #     for j in range(self.N_TRAJ):
        #         color = np.array([1 - 1 / (i + 1), 1 - 1 / (i + 1), 1 / (i + 1), 0.5])
        #         # r_h = plt.scatter(np.zeros((1,H)), np.zeros((1,H)), color = color)
        #         r_h, = ax.plot([],[], color = color)
        #         r_h_arr_list.append(r_h)
        #     self.r_h_arr.append(r_h_arr_list)
        # self.best_traj_h, = ax.plot([],[], color = np.array([0,1,0.5,1]), linewidth = 2)
        # self.cur_pos_h, = ax.plot([],[],'r*')

        self.cur_pos = pos_init
        self.cur_vel = pos_init * 0

    def call(self):
        dT = 0.1
        mu_alpha = 0.9


        ## MPPI
        # def mppi(num, cur_pos, cur_vel):
            # for i in range(N_ITER):
        bias = 0.1
        st = time.time()
        while np.linalg.norm(self.cur_pos-self.pos_goal, ord = 2) > bias:
            best_cost_iter = 10000000000.0
            # print(self.cur_pos)
            for j in range(self.N_POL):
                u = self.get_norm_samples(self.MU_ARR[:,:,j],self.SIGMA[j],self.N_TRAJ)
                u[:,:,-1] = u[:,:,-1] * 0

                #inject slowdown (for acceleration control)
                ss = 5
                u[:, 0:ss, -1] = u[:, 0:ss, -1] - self.cur_vel / dT / ss

                v_rollout = self.get_rollout(self.cur_vel,u,dT)
                rollout = self.get_rollout(self.cur_pos,v_rollout,dT)
                cost_p = self.calc_reaching_cost(rollout,self.pos_goal)
                cost_v = 0 * self.calc_reaching_cost(v_rollout,self.cur_vel * 0)
                #d1 = obs_dist(rollout, [3; 3], 2);
        #d1(d1>0) = 0;
        #d1(d1<0) = 500;
                cost = cost_p + cost_v
                w = self.w_fun(cost)
                w = w / sum(w)
                best_cost = np.max(w)
                best_idx = np.argmax(w)
                w_tens = np.expand_dims(np.transpose((mb.repmat(w, 1, self.H)).reshape([self.N_TRAJ, self.H])),0)
                self.MU_ARR[:,:,j] = (1 - mu_alpha) * self.MU_ARR[:,:,j] + mu_alpha * np.sum(np.multiply(w_tens,u), 2)
                # for i_traj in range(self.N_TRAJ):
                #     self.h_tmp = self.r_h_arr[j][i_traj]
                #     self.h_tmp.set_data(rollout[0,:,i_traj], rollout[1,:,i_traj])
                if best_cost < best_cost_iter:
                    cur_pos_temp = rollout[:,0,best_idx]
                    self.cur_pos = np.expand_dims(cur_pos_temp, axis=1)
                    cur_vel_temp = v_rollout[:,0,best_idx]
                    self.cur_vel = np.expand_dims(cur_vel_temp, axis=1)
                    # self.cur_pos_h.set_data(self.cur_pos[0], self.cur_pos[1])
                    best_cost_iter = best_cost
                    # self.best_traj_h.set_data(rollout[0,:,best_idx], rollout[1,:,best_idx])

        print(time.time() - st)
        return self.cur_pos, self.cur_vel

    def get_norm_samples(self, MU_ARR, SIGMA, N_TRAJ):
        result = np.zeros([2, 30, 50])
        for i in range(2):
            for j in range(30):
                for k in range(50):
                    # result[i, j, k] = np.random.normal((mb.repmat(MU_ARR, 1, N_TRAJ).reshape([self.D, self.H, self.N_TRAJ]))[i, j, k],
                    #                                    SIGMA)
                    result[i, j, k] = np.random.normal(MU_ARR[i, j],SIGMA)
        return result

    def get_rollout(self, pos_init, u_sampl, dT):
        # u_sampl_T = np.array([(u_sampl[:,:,i]).T for i in range(50)]).reshape(30,2,50)
        u_sampl_T = np.array([(u_sampl[:, :, i]).T for i in range(50)]).transpose(1, 2, 0)
        u_sampl_time = np.array([np.dot(dT * self.INT_MAT, u_sampl_T[:, :, i]) for i in range(50)]).transpose(1, 2, 0)
        pos_delta = np.array([(u_sampl_time[:, :, i]).T for i in range(50)]).transpose(1, 2, 0)
        result = np.array([pos_init + pos_delta[:, :, i] for i in range(50)]).transpose(1, 2, 0)
        return result


# Creating the Animation object
# np.random.seed(0)
# fig, ax = plt.subplots(figsize = (6, 6))
ud = MPPI()
cur_pos, cur_vel = ud.call()
a = 2
# anim = animation.FuncAnimation(fig, ud, frames=1000, interval=100, blit=True)
# plt.show()