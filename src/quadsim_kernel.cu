#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W); //图序列
    const int u = (c % (H * W)) / W; //行
    const int v = c % W; //列
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5; //注意非整除，+0.5是为了取像素中心点，×2-1是把[0,1]映射到[-1,1]
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    //fu、fv和1构成的向量指向该像素点所对应的射线，相机头为x轴，射线向量为d0=[1,-fv,-fu]，在相机系下
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1]; //转世界系，射线
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    scalar_t min_dist = 100; //默认深度值
    scalar_t  t = (-1 - oz) / dz; //o+td=o+tRd0，oz+tdz=-1时t代表沿射线方向到地面距离
    if (t > 0) min_dist = t;

    // others //把同组无人机也作为障碍渲染进去
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group; //后面障碍物的生成根据这里使得同组一致而不是每个无人机不一样
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        // (ox + t dx - cx)^2 + (oy + t dy - cy)^2 + 4 (oz + t dz - cz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) { //这里逻辑与上面一样
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) { //(ox + t dx - cx)^2 + (oy + t dy - cy)^2 = r^2
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // voxels //判断逻辑是如果射线与voxels相交，那么x、y、z坐标相交期间应该在长方体对应范围内
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    canvas[b][u][v] = min_dist;
}

template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt, //[?,B,3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos, //[?,B,3]
    float drone_radius,
    int n_drones_per_group) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;
    // assert(j < pos.size(0));
    // assert(b < pos.size(1));

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    scalar_t min_dist = max(1e-3f, oz + 1); //地面
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f); //防止无人机与地面重叠时，求向量时出现[0,0,0]的情况

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz); //近似
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz); //？？？为什么不做归一化
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz); //？？？为什么不做归一化
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy); //？？？为什么不做归一化
            nearest_ptz = oz;
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox); //？？？为什么不做归一化
            nearest_pty = oy;
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3; //这里是为了防止无人机与障碍物重叠时，求向量时出现[0,0,0]的情况
        //设计思路：当无人机在障碍物内，最近点是无人机自己，但是训练时会得到一个0向量，这并不方便于训练，因此要设法构造出一个远离障碍物的向量，最
        //直观的方式即为——当无人机在内部，压缩长方体，简单起见构造一个max_r，如果无人机在外部那么自然任意长方形半径都小于max_r，如果无人机在内部
        //它肯定是各个方向坐标都已经在长方体范围内了，那么max_r就会大于各个方向的半径，压缩长方体，无人机在哪个方向距离长方体中心最远，距离向量就朝哪边，
        //训练时就倾向于往那一边推出
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx)); //画个图做分析即可明白
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz)); //若处于-rz和rz之间，则ptz=oz，否则ptz=cz±rz
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel( //根据深度图构建几何梯度，似乎没用到
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t unit = fov_x_half_tan / W;
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    // if ReRender.diff_kernel is None:
    //     unit = 0.637 / depth.size(3)
    //     ReRender.diff_kernel = torch.tensor([
    //         [[1, -1], [1, -1]],
    //         [[1, 1], [-1, -1]],
    //         [[unit, unit], [unit, unit]],
    //     ], device=device).mul(0.5)[:, None]
    // ddepthdyz = F.conv2d(depth, ReRender.diff_kernel, None, 2)
    // depth = ddepthdyz[:, 2:]
    // ddepthdyz = torch.cat([
    //     torch.full_like(depth, -1.),
    //     ddepthdyz[:, :2] / depth,
    // ], 1)
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
    // ddepthdyz /= ddepthdyz.norm(2, 1, True).clamp_min(8);
}

} // namespace

void render_cuda(
    torch::Tensor canvas, //[B,H,W]
    torch::Tensor flow, //没用到
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R, //相机系到世界系
    torch::Tensor R_old, //没用到
    torch::Tensor pos,
    torch::Tensor pos_old, //没用到
    float drone_radius, //没用到
    int n_drones_per_group,
    float fov_x_half_tan) {
    const int threads = 1024;
    size_t state_size = canvas.numel(); //总元素数
    const dim3 blocks((state_size + threads - 1) / threads); //block数

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
    }));
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt, //[?,B,3]
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos, //[?,B,3]，对应nearst_pt
    float drone_radius, //没用到
    int n_drones_per_group) {
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1); //共多少个位置向量，每个线程算一个
    const dim3 blocks((state_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
}
