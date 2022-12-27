import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
clear('all')
close_('all','force')
addpath('../planar_robot_2d/')
rng('default')
## obstacle
scipy.io.loadmat('../planar_robot_2d/data/net50_pos_thr.mat')
y_f,dy_f = tanhNN(net)
obs_pos = np.transpose(np.array([7,0]))
obs_r = 2
## robot
r = np.array([4,4,0])
d = np.array([0,0,0])
alpha = np.array([0,0,0])
base = np.eye(4)
k1 = 0.9
k2 = 0.9
q_min = np.array([- k1 * np.pi,- k2 * np.pi,- 10,- 10])
q_max = np.array([k1 * np.pi,k2 * np.pi,10,10])
box = np.array([[q_min(np.arange(1,2+1))],[q_max(np.arange(1,2+1))]])
u_box = 2 * np.transpose(np.array([[- 1,1],[- 1,1]]))
pos_init = np.transpose(np.array([0.5 * np.pi,0.5 * np.pi]))
pos_goal = np.transpose(np.array([- 0.5 * np.pi,- 0.5 * np.pi]))
#figure
f_anim = plt.figure('Name','Animation','Position',np.array([100,100,1400,400]))
ax_anim = subplot(1,2,1)
plt.axis('equal')
plt.title('Planar robot')
plt.xlabel('x, m')
plt.ylabel('y, m')
axes(ax_anim)
ax_anim.XLim = np.array([[- 12],[12]])
ax_anim.YLim = np.array([[- 12],[12]])
hold('on')
robot_h = create_r(ax_anim,pos_init,r,d,alpha,base)
xc,yc = circle_pts(obs_pos(1),obs_pos(2),obs_r - 0.01)
crc_h = plt.plot(ax_anim,xc,yc,'r-','LineWidth',1.5)
## Constants and Parameters
N_ITER = 1000
H = 50
D = 2
SIGMA = np.array([1,0.5,0.1])
N_POL = SIGMA.shape[2-1]
MU_ARR = np.zeros((D,H,N_POL))
dT = 0.1
N_TRAJ = 30
gamma_vec = flip(np.array([0.98 ** np.linspace(1,H - 1,H - 1),0.98 ** H]))
beta = 0.9
mu_alpha = 0.99
## lambdas
get_norm_samples = lambda MU_ARR = None,SIGMA = None,N_TRAJ = None: normrnd(np.matlib.repmat(MU_ARR,np.array([1,1,N_TRAJ])),SIGMA)
INT_MAT = tril(np.ones((H,H)))
get_rollout = lambda pos_init = None,u_sampl = None,dT = None: pos_init + pagetranspose(pagemtimes(dT * INT_MAT,pagetranspose(u_sampl)))
calc_reaching_cost = lambda rollout = None,goal = None: np.transpose(np.squeeze(vecnorm(rollout - goal,2,1)))
obs_dist = lambda rollout = None,obs_pos = None,obs_r = None: np.transpose(np.squeeze(vecnorm(rollout - obs_pos,2,1))) - obs_r
w_fun = lambda cost = None: np.exp(- 1 / beta * np.sum(np.multiply(gamma_vec,cost), 2-1))
## plot preparation
ax_proj = subplot(1,2,2)
plt.axis('equal')
plt.title('2d MPPI vis')
plt.xlabel('q1')
plt.ylabel('q2')
hold('on')
#joint-space
x1_span = np.linspace(box(1,1),box(2,1),50)
x2_span = np.linspace(box(1,2),box(2,2),50)
X1_mg,X2_mg = np.meshgrid(x1_span,x2_span)
x = np.transpose(np.array([X1_mg,X2_mg]))
#nominal ds
# obstacle
inp = np.array([[x],[np.matlib.repmat(obs_pos,np.array([1,len(x)]))]])
val = y_f(np.array([[inp],[np.sin(inp)],[np.cos(inp)]]))
Z_mg = np.reshape(val, tuple(X1_mg.shape), order="F")
#distance heatmap and contour
__,obs_h = contourf(ax_proj,X1_mg,X2_mg,Z_mg,100,'LineStyle','none')
__,obs2_h = plt.contour(ax_proj,X1_mg,X2_mg,Z_mg,np.array([obs_r,obs_r + 0.01]),'LineStyle','-','LineColor','k','LineWidth',2)
ax_proj.XLim = box(:,1)
ax_proj.YLim = box(:,2)
plt.plot(pos_init(1),pos_init(2),'b*')
plt.plot(pos_goal(1),pos_goal(2),'r*')
r_h_arr = np.zeros((N_POL,N_TRAJ))
for i in np.arange(1,N_POL+1,1).reshape(-1):
    for j in np.arange(1,N_TRAJ+1,1).reshape(-1):
        r_h = plt.plot(ax_proj,np.zeros((1,H)),np.zeros((1,H)))
        r_h.Color = np.array([1 - 1 / i,1 - 1 / i,1 / i,0.5])
        r_h_arr[i,j] = r_h

best_traj_h = plt.plot(ax_proj,np.zeros((1,H)),np.zeros((1,H)))
best_traj_h.Color = np.array([0,1,0.5,1])
best_traj_h.LineWidth = 2
cur_pos_h = plt.plot(0,0,'r*')
## MPPI

cur_pos = pos_init
cur_vel = pos_init * 0
for i in np.arange(1,N_ITER+1,1).reshape(-1):
    best_cost_iter = 10000000000.0
    for j in np.arange(1,N_POL+1,1).reshape(-1):
        tic
        # sample controls (v or u)
        u = get_norm_samples(MU_ARR(:,:,j),SIGMA(j),N_TRAJ)
        # inject zero sample
        u[:,:,end()] = u(:,:,end()) * 0
        #inject slowdown (for acceleration control)
        ss = 5
        u[:,np.arange[1,ss+1],end()] = u(:,np.arange(1,ss+1),end()) - cur_vel / dT / ss
        #calculate rollouts
        v_rollout = get_rollout(cur_vel,u,dT)
        #v_rollout = u;
        rollout = get_rollout(cur_pos,v_rollout,dT)
        #calculate trajectory costs
        cost_p = calc_reaching_cost(rollout,pos_goal)
        cost_v = calc_reaching_cost(v_rollout,cur_vel * 0)
        cost_lim = calc_lim_cost(rollout,box(1,:),box(2,:))
        u_lim_cost = calc_lim_cost(u,u_box(1,:),u_box(2,:))
        v_lim_cost = calc_lim_cost(v_rollout,u_box(1,:),u_box(2,:))
        #smooth_cost = calc_smooth_cost(rollout);
        dists = calc_nn_dists(y_f,rollout,obs_pos,obs_r)
        coll_cost = 1000 * dist_cost(dists,0.1,0.2)
        cost = cost_p + cost_lim + u_lim_cost + v_lim_cost + coll_cost
        #calculate trajectory weights
        w = w_fun(cost)
        w = w / sum(w)
        best_cost,best_idx = np.amax(w)
        #update sampling means
        w_tens = np.reshape(np.transpose(np.matlib.repmat(w,np.array([1,H]))), tuple(np.array([1,H,N_TRAJ])), order="F")
        if j > 0:
            MU_ARR[:,:,j] = (1 - mu_alpha) * MU_ARR(:,:,j) + mu_alpha * np.sum(np.multiply(w_tens,u), 3-1)
        #shift sampling means
        MU_ARR[:,np.arange[1,end() - 1+1],:] = MU_ARR(:,np.arange(2,end()+1),:)
        #plots
        for i_traj in np.arange(1,N_TRAJ+1,1).reshape(-1):
            h_tmp = handle(r_h_arr(j,i_traj))
            h_tmp.XData = rollout(1,:,i_traj)
            h_tmp.YData = rollout(2,:,i_traj)
        if best_cost < best_cost_iter:
            cur_pos = rollout(:,1,best_idx)
            cur_vel = v_rollout(:,1,best_idx)
            cur_pos_h.XData = cur_pos(1)
            cur_pos_h.YData = cur_pos(2)
            best_cost_iter = best_cost
            best_traj_h.XData = rollout(1,:,best_idx)
            best_traj_h.YData = rollout(2,:,best_idx)
        toc
        move_r(robot_h,cur_pos,r,d,alpha,base)
        pause(0.05)
    best_cost

## functions

def calc_lim_cost(traj = None,v_min = None,v_max = None): 
cost = np.zeros((traj.shape[3-1],traj.shape[2-1]))
min_tens = np.matlib.repmat(np.transpose(v_min),1,traj.shape[2-1],traj.shape[3-1])
max_tens = np.matlib.repmat(np.transpose(v_max),1,traj.shape[2-1],traj.shape[3-1])
cost = np.logical_or(np.transpose(np.squeeze(np.any(traj < min_tens,1))),np.transpose(np.squeeze(np.any(traj > max_tens,1))))
#     for i = 1:1:length(v_min)
#         cost = cost | squeeze(traj(i,:,:)<v_min(i) | traj(i,:,:)>v_max(i))';
#     end
cost = double(cost)
return cost


def calc_nn_dists(y_f = None,rollout = None,obs_pos = None,obs_r = None): 
r_tmp = reshape(rollout,rollout.shape[1-1],[])
inp = np.array([[r_tmp],[np.matlib.repmat(obs_pos,1,r_tmp.shape[2-1])]])
dst = y_f(np.array([[inp],[np.sin(inp)],[np.cos(inp)]])) - obs_r
dst = np.transpose(reshape(dst,[],rollout.shape[3-1]))
return dst


def dist_cost(dst = None,thr0 = None,thr1 = None): 
cost = dst
idx_0 = dst > thr1

idx_1 = dst < thr0

idx_smooth = not (np.logical_or(idx_0,idx_1)) 
cost[idx_0] = 0
cost[idx_1] = 1
cost[idx_smooth] = 1 - cost(idx_smooth) / thr1
return cost


def calc_smooth_cost(traj = None): 
d_traj = np.diff(traj,1,2)
cost = np.squeeze(vecnorm(d_traj,2,1))
cost[end() + 1,:] = 0 * cost(end(),:)
cost = 10 * np.transpose(cost)
return cost

#robot and plotting

def calc_fk(j_state = None,r = None,d = None,alpha = None,base = None): 
P = dh_fk(j_state,r,d,alpha,base)
pts = np.zeros((3,3))
for i in np.arange(1,3+1,1).reshape(-1):
    v = np.array([0,0,0])
    R = P[i](np.arange(1,3+1),np.arange(1,3+1))
    T = P[i](np.arange(1,3+1),4)
    p = v * np.transpose(R) + np.transpose(T)
    pts[i,:] = p

return pts


def create_r(ax_h = None,j_state = None,r = None,d = None,alpha = None,base = None): 
pts = calc_fk(j_state,r,d,alpha,base)
handle = plt.plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,'Marker','o','MarkerFaceColor','k','MarkerSize',4)
return handle


def move_r(r_handle = None,j_state = None,r = None,d = None,alpha = None,base = None): 
pts = calc_fk(j_state,r,d,alpha,base)
r_handle.XData = pts(:,1)
r_handle.YData = pts(:,2)
return


def circle_pts(x = None,y = None,r = None): 
hold('on')
th = np.linspace(0,2 * np.pi,50)
xc = r * np.cos(th) + x
yc = r * np.sin(th) + y
return xc,yc
