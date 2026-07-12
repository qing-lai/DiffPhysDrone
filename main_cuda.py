from collections import defaultdict 
import math 
from random import normalvariate 
from matplotlib import pyplot as plt
from env_cuda import Env 
import torch 
from torch.nn import functional as F 
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm 

import argparse 
from model import Model 


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
args = parser.parse_args()
writer = SummaryWriter() #创建一个TensorBoard的writer对象，用于后续记录训练过程中的各种指标和数据，以便在TensorBoard中进行可视化分析。
print(args)

device = torch.device('cuda') 

env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle) #待分析，当前看作仿真训练环境对象
if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(7+3, 6)
model = model.to(device)

if args.resume: #待分析
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr) #优化器
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01) #按余弦变化学习率

ctl_dt = 1 / 15


scaler_q = defaultdict(list) 
def smooth_dict(ori_dict): 
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i): #前2000次迭代，每250次保存一次，之后每1000次保存一次
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80) #创建进度条，用于显示训练过程中的迭代进度和相关信息。range(args.num_iters)表示迭代的总次数，ncols=80设置了进度条的宽度为80个字符。
# depths = []
# states = []
B = args.batch_size
for i in pbar:
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = [] 
    vec_to_pt_history = [] #记录无人机到最近障碍物点的向量
    act_diff_history = [] #没用到
    v_preds = [] #网络预测速度的历史记录
    vid = [] #用于保存视频帧
    v_net_feats = [] #网络相关特征，包括控制输入、局部速度和GRU隐藏状态，疑似没用到
    h = None #GRU隐藏状态

    act_lag = 1 #动作延迟一步，实际上是两步；模拟网络延迟
    act_buffer = [env.act] * (act_lag + 1) #动作填充缓冲区
    target_v_raw = env.p_target - env.p
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15) #创建一个新的tensor对象，放入cuda，后续基于该tensor创立的tensor则默认放入cuda
        zeros = torch.zeros_like(drift_av) #按照drift_av的形状创建一个全零tensor，后续可看语义
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3) #stack指定新加维度创建新的tensor，reshape指定新tensor的形状


    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15) #正态分布，均值、标准差
        depth, flow = env.render(ctl_dt) #深度图，光流（目前光流实际是None）
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt()) #获得从当前位置和当前速度预测的10个未来位置到最近障碍物点的向量，加入[10, B, 3]的历史记录

        if is_save_iter(i):
            vid.append(depth[4]) #随意取的序列

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1) #None写在哪里，就在那里新增一个长度为 1 的维度。
            # @ / torch.matmul 对三维及以上 Tensor 有固定规则：最后两个维度当作矩阵维度；前面的维度当作 batch 维度。
            # torch.squeeze() 返回一个张量，删除所有大小为 1 的维度。这里指定了dim=1，表示只删除第 1 个维度，如果该维度的大小不为 1，则不会删除。
        else:
            target_v_raw = env.p_target - env.p.detach() #切断从env.p到target_v_raw的梯度传播
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R #机体系相对于世界系的旋转矩阵，形状为[B, 3, 3]
        fwd = env.R[:, :, 0].clone() #forward,[B,3]
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1) #F.normalize(input=fwd, p=2, dim=-1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1) #机体当前前进方向水平的矩阵，也理解为机体到世界坐标系的旋转矩阵，形状为[B, 3, 3]

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True) #l2范数沿最后一维计算长度，保留维度，[B,1]方便广播
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed) #[B,3]
        state = [
            torch.squeeze(target_v[:, None] @ R, 1), #用航向系旋转，得到目标速度在机体坐标系下的表示，[B,3]
            env.R[:, 2], #等价于[:,2,:]，整数索引会吃掉一维变成[B,3]，表示机体坐标系三个轴在世界系z轴上的投影
            env.margin[:, None]] #[B,1]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v) #有里程计加入本地速度
        state = torch.cat(state, -1) #[B,10]or[B,7]

        # normalize
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02 #限制在0.3~24之间并且做翻转，深度图归一化，加入噪声
        x = F.max_pool2d(x[:, None], 4, 4) #扩充为[B,1,H,W]，做最大池化，池化核大小为4，步长为4
        act, values, h = model(x, state, h) #注意values是None，h是GRU的隐藏状态

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1) #用航向系旋转，沿最后一维拆开，大致变成世界系下（这里我认为预测量为假设机体水平时的期望速度和速度）
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1)) #[B,198]，动作、局部速度、GRU隐藏状态拼接，疑似没用到

        v_history.append(env.v)
        target_v_history.append(target_v) 

    p_history = torch.stack(p_history) #默认沿0维堆叠，[T, B, 3]
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean() #[..., 2]等价于[:, :, 2]，惩罚上天偷懒？作者到底用没用？
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history) #[T, B, 3]
    v_history_cum = v_history.cumsum(0) #沿着第0维，得到截止目前累计的速度和，仍是[T, B, 3]
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30 #最近30步的平均速度，[T-30, B, 3]
    target_v_history = torch.stack(target_v_history) #[T, B, 3]
    T, B, _ = v_history.shape #更稳？
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1) #[T-30, B]，最近30步的平均速度和目标速度的差值
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v)) #Lv
    # smooth_l1_loss(input, target) = 0.5 * (input - target)^2 if |input - target| < 1 else |input - target| - 0.5

    v_preds = torch.stack(v_preds) #[T, B, 3]
    loss_v_pred = F.mse_loss(v_preds, v_history.detach()) #预测速度损失，论文中没有显式写出

    target_v_history_norm = torch.norm(target_v_history, 2, -1) #[T,B]
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1) #相当于做点积，得到速度沿着目标速度方向的分量，[T,B]
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3 
    #默认没用到，有替代，横向速度误差的惩罚，乘3是因为mse_loss是对所有单位元平均，而表达式是对每个时刻每个batch的平均

    jerk_history = act_buffer.diff(1, 0).mul(15) #求导，沿第0维做一阶差分，乘15是因为时间间隔是1/15s
    # snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    snap_history = F.normalize(act_buffer - env.g_std, 2, -1).diff(1, 0).diff(1, 0).mul(15**2)
    #默认p=2，dim=1，所以这里原来代码有问题；
    #这里是要控制控制输入方向二阶变化率不要太快，由于要normalize所以还是需要减去重力加速度
    loss_d_acc = act_buffer.pow(2).sum(-1).mean() #La
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean() #Lj
    loss_d_snap = snap_history.pow(2).sum(-1).mean() #系数默认为0了，论文里也没提到，但可能是因为代码本身写错了误认为没用

    vec_to_pt_history = torch.stack(vec_to_pt_history) #[T,10,B,3]
    distance = torch.norm(vec_to_pt_history, 2, -1) #[T,10,B]
    distance = distance - env.margin #[T,10,B]-[B]，广播
    with torch.no_grad(): #切断到v_to_pt的梯度传播
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1) #135是因为时间间隔是1/15s，10个点间有9段，clamp_min(1)表示最小值为1
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt) #Lc其一，距离惩罚
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean() #Lc其二，碰撞惩罚，论文中忘记乘v_to_pt了
    #softplus(x) = log(1 + exp(x))，当x很大时，softplus(x)≈x；当x很小时，softplus(x)≈0

    speed_history = v_history.norm(2, -1) #[T,B]
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm) #默认没用到，纵向速度误差的惩罚，有替代

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity * loss_ground_affinity #开源代码这里是＋号，可能笔误

    if torch.isnan(loss): #注意loss还是tensor张量
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}') #进度条左侧文字
    optim.zero_grad() #清空模型参数上的.grad，否则会累加上一次的梯度
    loss.backward() #把梯度写到各个模型参数的.grad属性中
    optim.step() #根据各个模型参数的.grad属性更新模型参数
    sched.step() #更新学习率


    with torch.no_grad(): #节省显存
        avg_speed = speed_history.mean(0) #每个样本的平均速度
        success = torch.all(distance.flatten(0, 1) > 0, 0) 
        #[T,10,B]0到1间维度压平变为[T*10,B]，然后沿第0维求逻辑与，得到[B]，表示每个样本是否成功
        _success = success.sum() / B
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(), #[T,B]沿0维求最大值然后平均，平均最大速度
            'avg_speed': avg_speed.mean(), #总平均速度
            'ar': (success * avg_speed).mean()}) #成功率加权的平均速度
        log_dict = {}
        if is_save_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            #视频，[T,H,W]，放到cpu方便tensorboard日志，除以10，限制到0，1之间，再变为[1,T,1,H,W]
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend() #显示图例
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            # writer.add_video('demo', vid, i + 1, 15)
            #[N, T, C, H, W] 视频数量，帧数，通道数，图像高度，图像宽度
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear() #每25次记录一次指标平均值
