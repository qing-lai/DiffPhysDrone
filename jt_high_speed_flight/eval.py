import argparse
from datetime import datetime
import json
import math
import os
import random
import signal
import subprocess
from time import sleep, time
import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim.types import AngleLevelControllerGains, PIDGains, AngleRateControllerGains
import numpy as np
from tqdm import tqdm

class VideoRecorder:
    def __init__(self, output, w, h, fps=15, pix_fmt='rgb24') -> None:
        self.p = None
        self.output = output
        command = [
            "/usr/bin/ffmpeg",
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f'{w}x{h}',  # size of one frame
            '-pix_fmt', pix_fmt,
            '-r', f'{fps}',  # frames per second
            '-i', '-',  # The imput comes from a pipe
            # '-qp', '0',
            '-s', f'{w//2*2}x{h//2*2}',
            '-an',  # Tells FFMPEG not to expect any audio
            # '-c:v', 'h264_nvenc',
            # '-preset', 'fast',
            '-loglevel', 'error',
            '-pix_fmt', 'yuv420p'
        ]
        self.p = subprocess.Popen(command + [self.output], stdin=subprocess.PIPE)

    def add_image(self, image):
        self.p.stdin.write(image)

    def close(self):
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()

class Rate:
    def __init__(self, hz) -> None:
        self.hz = hz
        self.t0 = time()

    def sleep(self):
        while True:
            to_sleep = 1 / self.hz - time() + self.t0
            if to_sleep < 0.01:
                break
            sleep(to_sleep)
        self.t0 += max(1 / self.hz, 0.5 / self.hz - to_sleep)


import torch
import torch.nn.functional as F

from model import Model

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    From: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='exps/run1/checkpoint0004.pth')
parser.add_argument('--env', default='swap')
parser.add_argument('--target_speed', default=2, type=float, help='(m/s) real speed might be 2 m/s slower')
parser.add_argument('--margin', default=0.15, type=float, help='(m) radius of body')
parser.add_argument('--smoothness', default=0.5, type=float, help='(m) radius of body')
parser.add_argument('--clockspeed', default=0.25, type=float)
parser.add_argument('--sr', default=3, type=int)
parser.add_argument('--no_odom', default=False, action='store_true')

args = parser.parse_args()
print(args)


# hover_thr = 0.297
hover_thr = 0.5
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'exps_{args.target_speed}/{datetime_str}/'
os.makedirs(log_dir)

agents = {
    "swap": [
        ("drone_1", [[ 0,  1.2, 0], [ 6, -1.2, 0]]),
        ("drone_2", [[ 0,  0.0, 0], [ 6,  0.0, 0]]),
        ("drone_3", [[ 0, -1.2, 0], [ 6,  1.2, 0]]),
        ("drone_4", [[ 6, -1.2, 0], [ 0,  1.2, 0]]),
        ("drone_5", [[ 6,  0.0, 0], [ 0,  0.0, 0]]),
        ("drone_6", [[ 6,  1.2, 0], [ 0, -1.2, 0]]),
    ],
    "single": [
        ("drone_1", [[0, 0, -2], [20, 20, -2]]),
    ],
}[args.env]

depth_recorder = VideoRecorder(f'{log_dir}/depth.mp4', 16*args.sr, len(agents)*12*args.sr, pix_fmt='y8')

B = len(agents)
agent_names, agent_waypoints = zip(*agents)
target_pos = [w[-1] for w in agent_waypoints]
traj_history = {agent_name: [] for agent_name in agent_names}

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

device = torch.device('cuda')
model = Model(7 if args.no_odom else 10, 6).eval().to(device)
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device))

@torch.no_grad()
def main():
    h = None
    for _ in range(10):
        _, _, h = model(
            torch.zeros(B, 1, 12, 16, device=device),
            torch.zeros(B, model.v_proj.in_features, device=device),
            h)

    sleep(1)
    for agent_name, waypoints in agents:
        sleep(0.1)
        client.enableApiControl(True, agent_name)
        client.armDisarm(True, agent_name)

        # set to start position
        client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=agent_name)
        yaw = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
        start_pt = waypoints.pop(0)
        start_pt = [
            start_pt[0] + random.random() * 0.2 - 0.1,
            start_pt[1] + random.random() * 0.2 - 0.1,
            start_pt[2] + random.random() * 0.5 - 0.25,
        ]
        for _ in range(3):
            sleep(0.1)
            client.simSetVehiclePose(Pose(
                Vector3r(*start_pt),
                Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))),
                ignore_collision=True, vehicle_name=agent_name)
        # align PID parameters to px4 default
        client.setAngleRateControllerGains(AngleRateControllerGains(
            roll_gains=PIDGains(0.2, 0.01, 0.001),
            pitch_gains=PIDGains(0.2, 0.01, 0.001),
            yaw_gains=PIDGains(0.2, 0.01, 0.001),
        ))
        client.setAngleLevelControllerGains(AngleLevelControllerGains(
            roll_gains=PIDGains(2, 0, 0),
            pitch_gains=PIDGains(2, 0, 0),
            yaw_gains=PIDGains(2, 0, 0),
        ), agent_name)
    sleep(0.5)
    for agent_name, waypoints in agents:
        client.simGetCollisionInfo(agent_name)

    p_target = torch.empty((B, 3))
    last_p = torch.empty((B, 3))
    forward_vec = torch.empty((B, 3))
    v = torch.empty((B, 3))
    R = torch.empty((B, 3, 3))
    traveled_distance = [0 for _ in agents]
    traveled_time = [0 for _ in agents]
    done_flag = [False for _ in agents]
    has_collided = [set() for _ in agents]
    extra = torch.tensor([[args.margin]]).repeat(B, 1)
    for i, (agent_name, waypoints) in enumerate(agents):
        x, y, z = waypoints.pop(0)
        p_target[i] = torch.as_tensor([x, -y, -z])

        # get initial forward vector
        state = client.getMultirotorState(agent_name)
        q = state.kinematics_estimated.orientation
        p = state.kinematics_estimated.position

        q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
        last_p[i] = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
        forward_vec[i] = quaternion_to_matrix(q)[:, 0]

    pbar = tqdm()
    hidden_state = None
    rate = Rate(15 * args.clockspeed)
    t_begin_real = time()
    t_now = t_begin = state.timestamp / 1e9
    t_end = t_begin + 30
    # wind = [0, 0, 0]
    # a_set = [0, 0, 0]
    ctl_error = 0
    ctl_error = torch.randn((len(agents), 3)) * 0.17
    # wind = [0, 0, 0]
    while t_now < t_end:
        pbar.update()
        # wind = [0.95 * w + 0.05 * random.normalvariate(0, 0.2) for w in wind]
        # client.simSetWind(Vector3r(*wind))

        # take images
        depth = []
        for agent in agent_names:
            responses = client.simGetImages([
                airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPlanar, True)
            ], agent)
            depth.append(airsim.get_pfm_array(responses[0]))
        depth = np.stack(depth)
        depth_viz = np.uint8(np.clip(depth / 10 * 255, 0, 255))
        depth_recorder.add_image(depth_viz.reshape(-1, 16*args.sr))
        # cv2.imshow('depth', np.uint8(np.clip(depth / 10 * 255, 0, 255)))
        # cv2.waitKey(1)

        for i, (agent_name, waypoints) in enumerate(agents):
            state = client.getMultirotorState(agent_name)
            t_now = state.timestamp / 1e9
            p = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            _v = state.kinematics_estimated.linear_velocity
            traj_history[agent_name].append([p.x_val, p.y_val, p.z_val, q.w_val, q.x_val, q.y_val, q.z_val])
            p = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])
            duration = t_now - t_begin
            if not done_flag[i]:
                traveled_distance[i] += torch.norm(p - last_p[i]).item()
                traveled_time[i] = duration
            last_p[i] = p

            v[i] = torch.as_tensor([_v.x_val, -_v.y_val, -_v.z_val])

            q = torch.as_tensor([q.w_val, q.x_val, -q.y_val, -q.z_val])
            R[i] = quaternion_to_matrix(q)

            # step to the next checkpoint if distance < 1
            if not done_flag[i] and torch.norm(p_target[i] - p) < 1.5:
                if waypoints:
                    x, y, z = waypoints.pop(0)
                    p_target[i] = torch.as_tensor([x, -y, -z])
                else:
                    print(f"{agent_names[i]} arrived in {duration}s!")
                    done_flag[i] = True
                    client.moveToPositionAsync(*target_pos[i], 0.5, vehicle_name=agent_names[i])
                    if all(done_flag):
                        t_end = t_now + 0.5

        # target velocity points to the target (with norm bounded by target_speed)
        target_v = p_target - last_p
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        target_v = target_v / target_v_norm * target_v_norm.clamp_max(args.target_speed)

        env_R = R.clone()
        fwd = R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = fwd / torch.norm(fwd, 2, -1, keepdim=True)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        # state (in body frame): cat(velocity estimation, velocity target, rotation matrix, margin)
        state = [torch.squeeze(target_v[:, None] @ R, 1), R[:, 2], extra]
        local_v = torch.squeeze(v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # normalize depth map
        depth = torch.as_tensor(depth, device=device)[:, None]
        x = 3 / depth.clamp_(0.3, 24) - 0.6
        x = F.max_pool2d(x, (args.sr, args.sr))

        # obtain velocity setpoint and prediction from nnet
        # state = (state - states_mean) / states_std
        state = state.to(device)
        action, _, hidden_state = model(x, state, hidden_state)
        # action = action.cpu() * action_std + action_mean
        v_setpoint, v_est = (R @ action.cpu().reshape(B, 3, -1)).unbind(-1)

        # obtain acceleration setpoint
        a_setpoint = v_setpoint - v_est + ctl_error
        a_setpoint[:, 2] += 9.80665

        # convert acceleration setpoint to rpy throttle
        throttle = torch.norm(a_setpoint, 2, -1)
        up_vec = a_setpoint / throttle[..., None]
        throttle = throttle + local_v[:, 2] * local_v[:, 2].abs() * 0.01

        # forward vector is the normalized moving average of target vector
        forward_vec = env_R[..., 0] * 5 + p_target - last_p
        forward_vec[:, 2] = (forward_vec[:, 0] * up_vec[:, 0] + forward_vec[:, 1] * up_vec[:, 1]) / -up_vec[:, 2]
        forward_vec /= torch.norm(forward_vec, 2, -1, True)
        left_vec = torch.cross(up_vec, forward_vec)

        roll = torch.atan2(left_vec[:, 2], up_vec[:, 2])
        pitch = torch.asin(-forward_vec[:, 2])
        yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
        for i, (r, p, y, t) in enumerate(zip(roll.tolist(), pitch.tolist(), yaw.tolist(), throttle.tolist())):
            if done_flag[i]:
                continue
            t = t / 9.8 * hover_thr
            client.moveByRollPitchYawThrottleAsync(r, p, y, t, 0.5, agent_names[i])
            collision_info = client.simGetCollisionInfo(agent_names[i])
            if collision_info.has_collided:
                has_collided[i].add(collision_info.object_name)
                print(f"{agent_names[i]} collide with {collision_info.object_name}!")

        # 15hz rate limit
        clockspeed = duration / (time() - t_begin_real)
        rate.hz = 15 * clockspeed
        rate.sleep()

    # traveled_time = (time() - t_begin) * args.clockspeed
    with open(f"{log_dir}/log", 'w') as f:
        f.write(f'{args}\n')
        for i, (x, t) in enumerate(zip(traveled_distance, traveled_time)):
            f.write(f'ours,{args.env},{args.target_speed},{agent_names[i]},{x:.2f},{t:.2f},0,{done_flag[i]},{"_".join(has_collided[i])}\n')


if __name__ == '__main__':
    import shutil
    shutil.copy(__file__, f"{log_dir}/eval.py")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ffmpeg_p = subprocess.Popen([
        '/usr/bin/ffmpeg',
        '-f', 'x11grab',
        '-video_size', '896x504',
        '-i', ':0+512,340',
        '-c:v', 'h264_nvenc',
        '-vf', f'setpts={args.clockspeed}*PTS',
        '-loglevel', 'error',
        '-an',
        f'{log_dir}/{datetime_str}.mp4'
    ], stdin=subprocess.PIPE)

    def cleanup():
        with open(f"{log_dir}/traj_history.json", 'w') as f:
            json.dump(traj_history, f)
        ffmpeg_p.stdin.close()
        ffmpeg_p.wait()
        depth_recorder.close()

    print("start recording")

    try:
        main()
        ffmpeg_p.send_signal(signal.SIGINT)
    finally:
        cleanup()
