import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep, time

import airsim
from airsim.types import AngleLevelControllerGains, PIDGains, AngleRateControllerGains
import numpy as np
from tqdm import tqdm


class VideoRecorder:
    def __init__(self, output, w, h, fps=15, pix_fmt="rgb24") -> None:
        self.p = None
        self.output = output
        command = [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", pix_fmt,
            "-r", f"{fps}",
            "-i", "-",
            "-s", f"{w // 2 * 2}x{h // 2 * 2}",
            "-an",
            "-loglevel", "error",
            "-pix_fmt", "yuv420p",
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


PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_DIR / "DiffPhysDrone"
sys.path.insert(0, str(TRAINING_DIR))

from model import Model


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    From:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    r, i, j, k = torch.unbind(quaternions, -1)
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
parser.add_argument(
    "--resume",
    default=str(TRAINING_DIR / "checkpoint0000.pth"),
)
parser.add_argument("--vehicle_name", default="drone_1")
parser.add_argument("--target_speed", default=4, type=float)
parser.add_argument("--margin", default=0.15, type=float)
parser.add_argument("--clockspeed", default=0.25, type=float)
parser.add_argument("--duration", default=30, type=float)
parser.add_argument("--sr", default=3, type=int)
parser.add_argument("--no_odom", default=False, action="store_true")

args = parser.parse_args()
print(args)


hover_thr = 0.297
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"exps_{args.target_speed}/{datetime_str}/"
os.makedirs(log_dir)

agent_name = args.vehicle_name
B = 1
traj_history = {agent_name: []}

depth_recorder = VideoRecorder(
    f"{log_dir}/depth.mp4",
    16 * args.sr,
    12 * args.sr,
    pix_fmt="y8",
)

# Connect to the AirSim simulator.
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()

device = torch.device("cuda")
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
            h,
        )
    h = None

    sleep(1)
    client.enableApiControl(True, agent_name)
    client.armDisarm(True, agent_name)
    client.moveByVelocityAsync(
        0,
        0,
        0,
        0.5,
        vehicle_name=agent_name,
    )

    client.setAngleRateControllerGains(
        AngleRateControllerGains(
            roll_gains=PIDGains(0.2, 0.01, 0.001),
            pitch_gains=PIDGains(0.2, 0.01, 0.001),
            yaw_gains=PIDGains(0.2, 0.01, 0.001),
        ),
        agent_name,
    )
    client.setAngleLevelControllerGains(
        AngleLevelControllerGains(
            roll_gains=PIDGains(2, 0, 0),
            pitch_gains=PIDGains(2, 0, 0),
            yaw_gains=PIDGains(2, 0, 0),
        ),
        agent_name,
    )
    sleep(0.5)
    client.simGetCollisionInfo(agent_name)

    last_p = torch.empty((B, 3))
    v = torch.empty((B, 3))
    R = torch.empty((B, 3, 3))
    extra = torch.tensor([[args.margin]])
    traveled_distance = 0
    has_collided = set()

    state = client.getMultirotorState(agent_name)
    p = state.kinematics_estimated.position
    last_p[0] = torch.as_tensor([p.x_val, -p.y_val, -p.z_val])

    pbar = tqdm()
    hidden_state = None
    rate = Rate(15 * args.clockspeed)
    t_begin_real = time()
    t_now = t_begin = state.timestamp / 1e9
    t_end = t_begin + args.duration

    while t_now < t_end:
        pbar.update()

        # Take the front depth image.
        responses = client.simGetImages(
            [
                airsim.ImageRequest(
                    "front_center_custom",
                    airsim.ImageType.DepthPlanar,
                    True,
                )
            ],
            agent_name,
        )
        depth = airsim.get_pfm_array(responses[0])[None]
        depth_viz = np.uint8(np.clip(depth / 10 * 255, 0, 255))
        depth_recorder.add_image(depth_viz.reshape(12 * args.sr, 16 * args.sr))

        # Read the state and convert AirSim NED to the training coordinate system.
        state = client.getMultirotorState(agent_name)
        t_now = state.timestamp / 1e9
        p = state.kinematics_estimated.position
        q = state.kinematics_estimated.orientation
        _v = state.kinematics_estimated.linear_velocity

        traj_history[agent_name].append(
            [p.x_val, p.y_val, p.z_val, q.w_val, q.x_val, q.y_val, q.z_val]
        )

        p = torch.as_tensor([[p.x_val, -p.y_val, -p.z_val]])
        traveled_distance += torch.norm(p - last_p).item()
        last_p = p

        v[0] = torch.as_tensor([_v.x_val, -_v.y_val, -_v.z_val])
        q = torch.as_tensor([[q.w_val, q.x_val, -q.y_val, -q.z_val]])
        R[0] = quaternion_to_matrix(q)[0]

        # Single-agent deployment uses a fixed feed-forward target velocity.
        target_v = torch.tensor([[args.target_speed, 0.0, 0.0]])

        env_R = R.clone()
        fwd = R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        # State: local velocity, local target velocity, attitude and margin.
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env_R[:, 2],
            extra,
        ]
        local_v = torch.squeeze(v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # Normalize the depth map from 36x48 to the network input 12x16.
        depth = torch.as_tensor(depth, device=device)[:, None]
        x = 3 / depth.clamp_(0.3, 24) - 0.6
        x = F.max_pool2d(x, (args.sr, args.sr))

        # Network inference.
        state = state.to(device)
        action, _, hidden_state = model(x, state, hidden_state)
        v_setpoint, v_est = (R @ action.cpu().reshape(B, 3, -1)).unbind(-1)

        # Convert the acceleration setpoint to roll, pitch, yaw and throttle.
        a_setpoint = v_setpoint - v_est
        a_setpoint[:, 2] += 9.80665

        throttle = torch.norm(a_setpoint, 2, -1)
        up_vec = a_setpoint / throttle[..., None]
        throttle = throttle + local_v[:, 2] * local_v[:, 2].abs() * 0.01

        forward_vec = env_R[..., 0] * 5 + target_v
        forward_vec[:, 2] = (
            forward_vec[:, 0] * up_vec[:, 0]
            + forward_vec[:, 1] * up_vec[:, 1]
        ) / -up_vec[:, 2]
        forward_vec = F.normalize(forward_vec, 2, -1)
        left_vec = torch.cross(up_vec, forward_vec, dim=-1)

        roll = torch.atan2(left_vec[:, 2], up_vec[:, 2]).item()
        pitch = torch.asin(-forward_vec[:, 2]).item()
        yaw = torch.atan2(forward_vec[:, 1], forward_vec[:, 0]).item()
        throttle = throttle.item() / 9.8 * hover_thr

        client.moveByRollPitchYawThrottleAsync(
            roll,
            pitch,
            yaw,
            throttle,
            0.5,
            agent_name,
        )

        collision_info = client.simGetCollisionInfo(agent_name)
        if collision_info.has_collided:
            has_collided.add(collision_info.object_name)
            print(f"{agent_name} collide with {collision_info.object_name}!")
            break

        clockspeed = (t_now - t_begin) / (time() - t_begin_real)
        rate.hz = 15 * clockspeed
        rate.sleep()

    pbar.close()
    client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=agent_name)

    with open(f"{log_dir}/log", "w") as f:
        f.write(f"{args}\n")
        f.write(
            f"ours,single,{args.target_speed},{agent_name},"
            f"{traveled_distance:.2f},{t_now - t_begin:.2f},"
            f"{'_'.join(has_collided)}\n"
        )


if __name__ == "__main__":
    import shutil

    shutil.copy(__file__, f"{log_dir}/eval.py")
    ffmpeg_p = subprocess.Popen(
        [
            "/usr/bin/ffmpeg",
            "-f", "x11grab",
            "-video_size", "896x504",
            "-i", ":0+512,340",
            "-c:v", "h264_nvenc",
            "-vf", f"setpts={args.clockspeed}*PTS",
            "-loglevel", "error",
            "-an",
            f"{log_dir}/{datetime_str}.mp4",
        ],
        stdin=subprocess.PIPE,
    )

    def cleanup():
        with open(f"{log_dir}/traj_history.json", "w") as f:
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
