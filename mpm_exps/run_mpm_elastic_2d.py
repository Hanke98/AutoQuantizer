import os
import time
import argparse
import sys

import numpy as np 
import taichi as ti

sys.path.append('..')
from engine.mpm_solver import MPMSolver
from mpm_config import MPMConfig
from solvers.analytical_solver import LagrangianCR, LagrangianEB

class SolverMode:
    CR = 0
    EB = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--ad',
                        action='store_true',
                        help='compute gradients')
    parser.add_argument('-r',
                        '--ranges',
                        action='store_true',
                        help='ranges')
    parser.add_argument('-e',
                        '--epsilon',
                        type=float,
                        default=0.1,
                        help='epsilon')
    parser.add_argument('-f',
                        '--forward',
                        action='store_true',
                        default=False,
                        help='dithering')
    parser.add_argument('-m',
                        '--mode',
                        type=int,
                        default=0,
                        help='mode')
    # 0 for compression rate; 1 for error bound 
    parser.add_argument('-d',
                        '--dithering',
                        action='store_true',
                        default=False,
                        help='dithering')
    args = parser.parse_args()
    print(args)
    return args


config = MPMConfig()
config.dim = 2
config.E = 5e1
config.id = 'elastic_2d_err_bound'
config.la = 1e3

bound = [3, 3]
config.bound = bound

config.output_folder = 'outputs/pics/elastic_2d/'
config.use_bls = not config.use_difftaichi
config.use_friction = False
config.use_fluid = False

config.n_particles = 80000
config.dt = 2e-4
config.n_grid = 128
config.n_timesteps = 1024 * 8 + 1
config.steps = 64
config.size = 1
config.p_rho = 1.0
dx = 1/config.n_grid
config.p_vol = (dx*0.5) ** config.dim

args = parse_args()

f64_data = 6893.409
eps = args.epsilon
tar = 1.0
if args.mode == SolverMode.CR:
    tar = eps
else:
    tar = eps * f64_data
# print(f'eps: {eps}')
print(f'tar: {tar}')


config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'F': 1})
config.quant_bits = np.array([32, ]*12)
config.quant_ranges = np.array([2.0,]*12)

dx = 1/config.n_grid

grad_folder = 'grads/elastic_2d/'
range_folder = 'ranges/elastic_2d/'
# grad_folder = 'mpm_exp/grads/rerun/elastic_2d'
# range_folder = 'mpm_exp/ranges/rerun/elastic_2d/'
# file_name = f'elastic_2d_t{config.n_timesteps}_dt{config.dt}_p{config.n_particles}_g{config.n_grid}.npy'
file_name = 'elastic_2d.npy'


ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.float64, kernel_profiler=True, print_ir=False)
# mpm = MPMSolver(config)


def init(mpm):
    mpm.initialize()
    mpm.x.fill(0)
    mpm.v.fill(0)

    mpm.rp.seed(1)

    for i in range(4):
        mpm.add_cube(((0.2 + i * 0.4)/4, (.2 + i * 0.8)/4),
                    (0.7/4, 0.7/4), config.n_particles//8)

    for i in range(4):
        mpm.add_cube(((3.1 - i * 0.2)/4, (0.2 + i * 0.8)/4),
                    (0.7/4, 0.7/4), config.n_particles//8)

def run_forward_only(mpm, ggui, save_pics=False, save_states=False):
    frame = 0
    if save_pics:
        output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
        os.makedirs(output_path, exist_ok=True)
        if ggui is not None:
            ggui.init_ggui_window(pos=(4.5, 0.5, 4.5), center=(2.0, 0.1, 2.0))
    if save_states:
        pos_output_folder = mpm.param.pos_output_folder
        os.makedirs(pos_output_folder, exist_ok=True)

    # max_frame = int(mpm.n_timestep / mpm.steps)# + 1.)
    # print(f'max_frame: {max_frame}')
    # while frame < max_frame:
    s = 0
    while s < mpm.n_timestep - 1:
        if save_pics:
            x_ = mpm.x.to_numpy(dtype=np.float32)
            x_ = x_[:, :, :2]# / 4.0
            gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0xDDDDDD, show_gui=False)
            gui.circles(x_[0], radius=2.5, color=0xED553B)
            gui.show(f'{output_path}/{frame:06d}.png')
        k = 0
        while k < mpm.steps and s < mpm.n_timestep - 1:
            if mpm.use_g2p2g:
                mpm.substep_g2p2g()
            else:
                mpm.substep(0, 0)
            k += 1
            s += 1
        print('.', end='', flush=True)
        if save_states: 
            mpm.dump_states(frame)
        frame += 1
    mpm.compute_loss()
    return mpm.loss[None]

def run_forward_with_ad_nlogn(mpm, grad_fn):
    mpm.checkpoint(mpm.logn_timestep-1, 0)
    with ti.Tape(loss=mpm.loss):
        for s in range(mpm.n_timestep - 1):
            mpm.nlogn_loop_forward(s, False)
            if (s-1) % mpm.steps == 0:
                print('.', end='', flush=True)
        mpm.compute_loss()
        print('\nforward end')
    mpm.dump_grad(0, grad_fn)
    print(f'loss= {mpm.loss[None]}', flush=True)
    return mpm.loss[None]

def run_forward_detect_ranges(mpm, fn):
    for s in range(mpm.n_timestep - 1):
        mpm.nlogn_step_forward(s)
        # mpm.detect_bound(1-s%2)
        mpm.detect_bound_kernel(1-s%2)
        if s % mpm.steps == 0:
            print('.', end='', flush=True)
    mpm.detect_bound()
    ranges = mpm.get_limits()
    print(ranges)
    np.save(fn, ranges)

def run(N=5, quant=False, save_pics=False):
    if quant:
        config.use_difftaichi=False
        config.use_bitpack=True
        config.use_quantaichi=True
    loss = []
    config.use_bls=True
    mpm = MPMSolver(config)

    total_time = 0
    for _ in range(N):
        st = time.time()
        init(mpm)
        loss.append(run_forward_only(mpm, None, save_pics, False))
        total_time += time.time() - st
    print(f'total time: {total_time:.3f} s')
    loss = np.array(loss)
    for l in loss:
        print(l)
    if args.mode == SolverMode.EB:
        d = abs(loss - f64_data) 
        print('succeed: ', np.sum(abs(d) < tar * 3) / N)
    print(f'loss: mean: {loss.mean()}, std: {loss.std()}, max: {loss.max()}, min: {loss.min()}')

def ad():
    config.use_difftaichi = True
    config.use_bls = not config.use_difftaichi
    mpm = MPMSolver(config)
    init(mpm)
    os.makedirs(grad_folder, exist_ok=True)
    grad_fn = f'{grad_folder}/{file_name}'
    run_forward_with_ad_nlogn(mpm, grad_fn)
    print(mpm.x.grad.to_numpy()[0, 0])

def detect_range():
    mpm = MPMSolver(config)
    init(mpm)
    os.makedirs(range_folder, exist_ok=True)
    fn = f'{range_folder}/{file_name}'
    run_forward_detect_ranges(mpm, fn)

def solve():
    grads = np.load(f'{grad_folder}/{file_name}')
    ranges = np.load(f'{range_folder}/{file_name}')
    ranges[:config.dim] = 1.0
    ranges[config.dim:] *= 2.0
    n_vars = grads.shape[0]

    if args.mode == SolverMode.CR:
        bits, _ = LagrangianCR(grads=grads*(ranges**2), r=eps)
        config.quant_bits = bits + 1
    else:
        bits, _ = LagrangianEB(grads=grads*(ranges**2), ref=f64_data, tol_rel=eps)
        config.quant_bits = bits + 1
    config.quant_ranges = ranges
    print('bits:', config.quant_bits)
    print(f'real compression rate: {sum(config.quant_bits)/(32*n_vars)}')

if args.ad:
    ad()
elif args.ranges:
    detect_range()
elif args.forward:
    solve()
    run(N=20, quant=True, save_pics=True)
