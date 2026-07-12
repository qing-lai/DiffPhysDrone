import math 
import random 
import time
import torch 
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function): #自定义前向反向传播，这个函数实际上不在真正训练中用到
    @staticmethod #要求使用静态方法：不需要 self
    def forward(ctx, x, alpha):
        ctx.alpha = alpha #保存到上下文，供 backward 反向传播时使用
        return x

    @staticmethod
    def backward(ctx, grad_output): #若前向为y=f(x)，grad_output就是dL/dy，这里给定dL/dx的实现，比如这里就是乘以衰减系数
        return grad_output * ctx.alpha, None #分别对应dL/dx,dL/dalpha
    #dL/dy未定义时会自动成0Tensor

g_decay = GDecay.apply #调用时就是GDecay.apply(x,alpha)


class RunFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next) #存tensor
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next
    # 动力学更新

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors #取tensor
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None #与前向input一一对应
    #d_act_pred传回策略网络，其余的相互传递

run = RunFunction.apply


class Env:
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device) #球
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device) #长方体
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device) #地面障碍
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device) #圆柱体
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device) #水平圆柱体
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device) #门框
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device) #风速扰动
        self.g_std = torch.tensor([0., 0, -9.80665], device=device) #重力加速度
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device) #增加屋顶障碍的偏移量
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1) #0，1/135，... ，1/15，[10,1,1]
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size] #0维重复a次，1维重复1次
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device) #光流，本项目没用到
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device) #大概是三个方向原本想取不一样的避障权重

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3) #相机系到机体系，绕y转-cam_angle，所以这里cam_angle正时相机朝上倾斜

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b #注意rand:[0,1)，B个环境，30个球，x，y，z，r，高度[-1,5)，半径[0.4,0.6)
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b #6个参数，3个中心位置，3个半长度，高度[-1,5)，半径[0.2,0.3)
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b #x,y,r，半径[0.05,0.4)
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b #x,z,r，半径[0.05,0.15)
        #x统一[0,8)，y统一[-9,9)

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan #random.random()：[0,1)随机数
        self.n_drones_per_group = random.choice([4, 8]) #随机4或8个无人机一组
        self.drone_radius = random.uniform(0.1, 0.15) #[0.1,0.15]
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        #每个随机数沿第0维重复（一组内的无人机数量）个数，对应每组无人机共享一个速度，这里隐形要求batch_size应当能被8整除，得到[B,1]
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp #[0.75,3.25)乘倍数，[B,1]
        scale = (self.max_speed - 0.5).clamp_min(1) #尺度，后续用于拉伸场景和起终点，[B,1]

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01 #随机每个无人机的推力估计误差，[B]

        roof = torch.rand((B,)) < 0.5 #随机一个是否roof的名单。[B]
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2] #若某环境（对应某batch）不生成roof，则把该环境中一半的球的x，y与竖直圆柱体重合
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2] #长方体同理
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4] #对于前面选定的roof球，抬高高度2.5，半径变大1.5
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add #对于前面选定的roof长方体，抬高高度2.5，半长度均加1.5
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        #x限制在[r+0.3/scale,8-0.3/scale-r]，防止出边界，后续还乘scale，所以0.3/scale其实是留出0.3余量
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200 #这两句是在roof对应的环境制造一个巨大的上方box作为天花板，天花板一定会高于地面1.5m([0.5,3.5))

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6 #[8,14)
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4 #[2,6)
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1 #-1是以地底-1为参考，这里整体是在地上嵌入球

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b #x、y半长[0.1,3.0),z半长[0.1,2.0)
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1 #z=-1+sz，底部刚好接触地面，顶部-1+2sz
            self.voxels = torch.cat([self.voxels, ground_voxels], 1) #[B,40,6]

        #延伸y位置
        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b #x:[3,5),y:[-1,1),z:[0,1),r:[0.5,1.0)
            p = gate[None, :, :3] #[1,B,3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1) #找到最近的障碍物表面点坐标
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1) #[B]
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50 #对[B]即每个环境判断是否门框中心与障碍物最近点，若太近则推到很远不要了
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1) #中间洞小，边框很大，强制从中间穿过，[B,4,6]

            self.voxels = torch.cat([self.voxels, gate], 1) #[B,44,6]
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale #x方向位置延伸
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)
        #x夹到[ground_balls_r_ground+0.3, 8*scale-0.3-ground_balls_r_ground]，保证地面球凸起也在场景内

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device) #名字起的很怪，“delay”越大，响应越快，而且是对加速度响应？
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device) #名字起的依然很怪，“delay”越大，yaw就转的越快

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale, #[B,1]，拉伸x，每组共享
            rd + 0.5, #每组共享,[B,1]，拉伸y，范围[0.5,1.5)
            torch.rand_like(scale) - 0.5], -1) #[B,3]，这里高度是变成随机到[-0.5,0.5]
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1 #起点
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1 #终点

        if self.random_rotation: #旋转有个好处就是可以改变无人机对立方体的大多数视角，也可以理解为防止学到世界系特征（如沿x轴，虽然我感觉没有）
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75 #[-0.75,0.75],43°
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3) #绕z轴转yaw_bias
            self.p = torch.squeeze(R @ self.p[..., None], -1) #[B,3]
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2) #transpose交换1和2维
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # scaffold 没用到，暂时不看
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2 #初始化速度
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w #风速
        self.act = torch.randn_like(self.v) * 0.1 #初始化重力与推力叠加的实际净加速度
        self.a = self.act #初始化加速度
        self.dg = torch.randn((B, 3), device=device) * 0.2 #随机扰动加速度

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5) #这里注意初始化姿态的x轴是朝着期望速度方向且有一定随机偏移，z轴朝着初始化的推力方向（由act计算出来）
        self.R_old = self.R.clone()
        self.p_old = self.p #这两个疑似本来给光流实现用的，现在并没有实现
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1 #无人机余量

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0 #drag_2[:,0]:二次阻力系数,drag_2[:,1]:一次阻力系数[0.3,0.45)，默认二次阻力项关闭
        self.z_drag_coef = torch.ones((B, 1), device=device) #z轴阻力系数，值为1

    @staticmethod #注意这里也只是演示，实际并不用到
    @torch.no_grad() #装饰器从下往上应用，大致理解为update_state_vec = staticmethod(torch.no_grad()(update_state_vec))
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5): #分别输入上一次姿态、新的实际净加速度、上一次状态推导的参考速度、yaw轴的延迟系数，最后一个不好言传
        self_forward_vec = R[..., 0] #[B,3]，矩阵R的第一列，即机体系x轴在世界系投影
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std #推力加速度，[B,3]
        thrust = torch.norm(a_thr, 2, -1, True) #[B,1]
        self_up_vec = a_thr / thrust #z轴
        forward_vec = self_forward_vec * yaw_inertia + v_pred #期望方向拉动x轴往其移动，之前的x轴向量乘5，让拉动本身不那么快，这一步是构造一个新的期望方向
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha) #延迟响应
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2] #正交化，利用点积为0
        self_forward_vec = F.normalize(forward_vec, 2, -1) #标准化
        self_left_vec = torch.cross(self_up_vec, self_forward_vec) #叉乘得y轴
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1) #总体就是更新姿态矩阵

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div #[B,3]*[10,1,1] [10,B,3]
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p #返回的是到点到各自最近点向量，[10,B,3]

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        #这么设计是为了保持方差为0.2^2
        self.p_old = self.p #没用到
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone() #没用到
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5) #更新姿态

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None): #并不使用这一版本，阻力实现有所不同，不去关心
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

