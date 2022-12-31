import numpy as np
from numpy import matlib as mb
# import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as scio
import torch
# clear('all')
# close_('all','force')
# addpath('../planar_robot_7d/')
# rng('default')
class MPPI:
    def __init__(self, ax):
        device = torch.device('mps')
        tensor_args = {'device': device, 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.ax = ax
        ## Constants and Parameters
        self.D = 2
        self.H = 30
        self.SIGMA = np.array([1,0.1])
        self.N_POL = np.size(self.SIGMA)
        self.MU_ARR = torch.zeros(self.D, self.H, self.N_POL, **tensor_args)

        self.N_TRAJ = 50
        pos_init = torch.zeros(self.D,1, **tensor_args)
        self.pos_goal = pos_init + 6
        self.gamma_vec = torch.cat((torch.tensor([0.98 ** i for i in np.linspace(1,self.H - 1,self.H - 1)]), torch.tensor([1.02**self.H])),0).to(**self.tensor_args)

        self.beta = 0.9

        ## plot preparation
        # plt.ion()
        self.ax.set_xlim([-1, 10])
        self.ax.set_ylim([-1, 10])
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_title('MPPI example')
        self.ax.scatter(pos_init[0],pos_init[1], color = [0, 0, 1], marker='*')
        self.ax.scatter(self.pos_goal[0],self.pos_goal[1], color = [1, 0, 0], marker='*')
        # r_h_arr = np.zeros((N_POL,N_TRAJ))
        self.r_h_arr = []
        r_h_arr_list = []
        for i in range(self.N_POL):
            for j in range(self.N_TRAJ):
                color = np.array([1 - 1 / (i + 1), 1 - 1 / (i + 1), 1 / (i + 1), 0.5])
                # r_h = plt.scatter(np.zeros((1,H)), np.zeros((1,H)), color = color)
                r_h, = ax.plot([],[], color = color)
                r_h_arr_list.append(r_h)
            self.r_h_arr.append(r_h_arr_list)
        self.best_traj_h, = ax.plot([],[], color = np.array([0,1,0.5,1]), linewidth = 2)
        self.cur_pos_h, = ax.plot([],[],'r*')

        self.cur_pos = pos_init
        self.cur_vel = pos_init * 0

    def __call__(self, num):
        bias = 0.1
        dT = 0.1
        mu_alpha = 0.9

        if  np.linalg.norm(self.cur_pos-self.pos_goal, ord = 2) < bias:
            return False
        else:
            ## MPPI
            # def mppi(num, cur_pos, cur_vel):
                # for i in range(N_ITER):
            best_cost_iter = 10000000000.0
            for j in range(self.N_POL):
                u = self.get_norm_samples(self.MU_ARR[:,:,j],self.SIGMA[j],self.N_TRAJ)
                # data = scio.loadmat('./u.mat')
                # u_temp = np.array(data['u'])
                # u = torch.tensor(u_temp).transpose(0,2).transpose(1,2)
                # u = u.type(torch.float32)
                u[-1,:,:] = u[-1,:,:] * 0


                #inject slowdown (for acceleration control)
                ss = 5
                u[-1, :, 0:ss] = u[-1, :, 0:ss] - self.cur_vel / dT / ss

                v_rollout = self.get_rollout(self.cur_vel,u,dT)
                rollout = self.get_rollout(self.cur_pos,v_rollout,dT)
                cost_p = self.calc_reaching_cost(rollout,self.pos_goal)
                cost_v = 0 * self.calc_reaching_cost(v_rollout,self.cur_vel * 0)
                #d1 = obs_dist(rollout, [3; 3], 2);
        #d1(d1>0) = 0;
        #d1(d1<0) = 500;
                cost = cost_p + cost_v
                w = self.w_fun(cost)
                w = w / torch.sum(w)
                best_cost = torch.max(w)
                best_idx = torch.argmax(w)
                self.MU_ARR[:, :, j] = (1 - mu_alpha) * self.MU_ARR[:, :, j] + mu_alpha * torch.sum(torch.mul(w.unsqueeze(1).unsqueeze(2),u), 0)

                for i_traj in range(self.N_TRAJ):
                    self.h_tmp = self.r_h_arr[j][i_traj]
                    self.h_tmp.set_data(rollout[i_traj,0,:], rollout[i_traj,1,:])
                if best_cost < best_cost_iter:
                    cur_pos_temp = rollout[best_idx,:,0]
                    self.cur_pos = cur_pos_temp.unsqueeze(1)
                    cur_vel_temp = v_rollout[best_idx,:,0]
                    self.cur_vel = cur_vel_temp.unsqueeze(1)
                    self.cur_pos_h.set_data(self.cur_pos[0], self.cur_pos[1])
                    best_cost_iter = best_cost
                    self.best_traj_h.set_data(rollout[best_idx, 0,:], rollout[best_idx, 1,:])
            return self.h_tmp, self.cur_pos_h, self.best_traj_h

    def get_norm_samples(self, MU_ARR, SIGMA, N_TRAJ):
        result_temp = torch.distributions.Normal(MU_ARR.repeat(self.N_TRAJ,1,1), SIGMA)
        result = result_temp.sample()
        return result

    def get_rollout(self, pos_init, u_sampl, dT):
        self.INT_MAT = torch.tril(torch.ones(self.H, self.H))
        # self.INT_MAT = torch.tril(torch.ones(self.H, self.H)).unsqueeze(0)
        u_sampl_T = u_sampl.transpose(2,1)
        u_sampl_time = torch.matmul(dT * self.INT_MAT, u_sampl_T)
        pos_delta = u_sampl_time.transpose(2, 1)
        result = pos_init + pos_delta
        return result

    def calc_reaching_cost(self, rollout, goal):
        result = torch.linalg.norm(rollout - goal, dim = 1)
        return result

    def w_fun(self, cost):
        result = torch.exp(- 1 / self.beta * torch.sum(self.gamma_vec * cost, 1))
        return result



# Creating the Animation object
np.random.seed(0)
fig, ax = plt.subplots(figsize = (6, 6))
ud = MPPI(ax)
anim = animation.FuncAnimation(fig, ud, frames=1000, interval=100, blit=True)
plt.show()