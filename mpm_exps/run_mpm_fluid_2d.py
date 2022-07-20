import os
import time
import sys
import argparse

sys.path.append('..')
from engine.mpm_solver import MPMSolver
from mpm_config import MPMConfig

import numpy as np 
import taichi as ti

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
    parser.add_argument('-m',
                        '--mode',
                        type=int,
                        default=0,
                        help='mode')
    # 0 for compression rate; 1 for error bound 
    args = parser.parse_args()
    print(args)
    return args

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


def run_forward_with_ad_nlogn(mpm, grads_fn):
    mpm.checkpoint(mpm.logn_timestep-1, 0)
    with ti.Tape(loss=mpm.loss):
    # if True:
        for s in range(mpm.n_timestep - 1):
            mpm.nlogn_loop_forward(s, False)
            # mpm.substep(0, 0)
            if s % mpm.steps == 0:
                print('.', end='', flush=True)
        print('')
        mpm.compute_loss()
    mpm.dump_grad(0, fn=grads_fn)
    print(f'loss= {mpm.loss[None]}', flush=True)
    return mpm.loss[None]


def run_forward_detect_ranges(mpm, range_fn):
    for s in range(mpm.n_timestep - 1):
        # mpm.nlogn_loop_forward(s, False)
        # mpm.detect_bound(1-s%2)
        mpm.substep(0, 0)
        mpm.detect_bound_kernel(0)
        if s % mpm.steps == 0:
            print('.', end='', flush=True)
    mpm.compute_loss()
    mpm.detect_bound()
    ranges = mpm.get_limits()
    print(ranges)
    np.save(range_fn, ranges)

def init(mpm):
    mpm.initialize()
    mpm.rp.seed(1)

    mpm.add_cube(((config.bound[0]) * dx, (config.bound[1]) * dx),
                (1.5/4, 1.5/4), config.n_particles)

def run(N=5, quant=False, save_pics=False):
    if quant:
        config.use_difftaichi=False
        config.use_bls=True
        config.use_bitpack=True
        config.use_quantaichi=True
    loss = []
    mpm = MPMSolver(config)
    for _ in range(N):
        init(mpm)
        loss.append(run_forward_only(mpm, None, save_pics, False))
    loss = np.array(loss)
    print(f'loss: mean: {loss.mean()}, std: {loss.std()}, max: {loss.max()}, min: {loss.min()}')

def LagrangianCR(grads, r = 0.5, a = None):
    n = len(grads)
    grads = np.array(grads)
    r = r * 32.0 -1
    if a is not None:
        assert(len(a) == n)
        a = np.array(a)
    else:
        a = np.ones(n, dtype = np.float64)
    b = a / grads
    t = np.log2(b)
    log_la = -r * 2 - t @ a/np.sum(a) + np.log2(2 * np.log(2))
    la = np.power(2, log_la)
    x = b * la
    fun = x @ grads / 12.0
    bits = np.ceil(-(t + log_la)/2.0).astype(np.int32)
    return bits, fun

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

    ranges[:2] = 1.0
    ranges[2:] *= 2.0

    n_vars = grads.shape[0]
    grads = grads/grads.min()
    # print(ranges)
    # print(grads)
    config.quant_ranges = ranges
    bits, _ = LagrangianCR(grads*ranges**2, r=eps)
    config.quant_bits = bits
    print('bits: ', config.quant_bits)
    print(f'real compression rate: {sum(config.quant_bits)/(32*n_vars)}')


args = parse_args()
assert args.mode == SolverMode.CR
eps = args.epsilon

config = MPMConfig()
config.dim = 2
config.E = 5e4
config.id = 'fluid_2d_mem_bound'
config.la = 5e2
bound = [3, 3]
config.bound = bound

config.output_folder = 'outputs/pics/fluid_2d/'

config.use_difftaichi = True
config.use_bls = not config.use_difftaichi
config.use_friction = False

config.quant_ranges = np.ones(9)
config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'J': 1})
config.quant_bits = np.ones([9]) 

config.n_particles = 10000
config.dt = 2e-4
config.n_grid = 128# * 2 
config.n_timesteps = 1024 * 16 + 1
config.steps = 128 
config.size = 1

dx = 1/config.n_grid

draw_propeller = True

ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.float64, kernel_profiler=True, print_ir=False)

grad_folder = 'grads/fluid_2d/'
range_folder = 'ranges/fluid_2d/'
file_name = f'fluid_2d_t{config.n_timesteps}_dt{config.dt}_p{config.n_particles}_g{config.n_grid}.npy'


if args.ad:
    ad()
elif args.ranges:
    detect_range()
elif args.forward:
    solve()
    run(N=20, quant=True, save_pics=False)
    # run(N=20, quant=True, save_pics=True)

