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
    parser.add_argument('-e',
                        '--epsilon',
                        type=float,
                        default=0.1,
                        help='epsilon')
    # 0 for compression rate; 1 for error bound 
    parser.add_argument('-m',
                        '--mode',
                        type=int,
                        default=0,
                        help='mode')
    # for optmality check
    parser.add_argument('-t',
                        '--task',
                        type=int,
                        default=0,
                        help='reduce_num') # 0 for reduce_all, 1 for reduce half, 2 for moving
    parser.add_argument('-b',
                        '--search-bits',
                        type=int,
                        default=1,
                        help='search_bits')
    parser.add_argument('-b2',
                        '--search-bits-2',
                        type=int,
                        default=1,
                        help='search_bits 2')
    args = parser.parse_args()
    print(args)
    return args


config = MPMConfig()
config.dim = 2
config.E = 5e1
config.id = 'elastic_oc'
config.la = 1e3

bound = [3, 3]
config.bound = bound

config.output_folder = 'outputs/pics/'
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
file_name = 'elastic_2d.npy'


ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.float64, kernel_profiler=True, print_ir=False)

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
            mpm.substep(0, 0)
            k += 1
            s += 1
            # mpm.step_collider()
        print('.', end='', flush=True)

        if save_states: 
            mpm.dump_states(frame)
        # print(f'frame: {frame}', flush=True)
        frame += 1
    mpm.compute_loss()
    return mpm.loss[None]

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

def solve():
    grads = np.load(f'{grad_folder}/{file_name}')
    ranges = np.load(f'{range_folder}/{file_name}')
    ranges[:config.dim] = 1.0
    ranges[config.dim:] *= 2.0
    n_vars = grads.shape[0]
    # print('grads:', grads)
    # print('ranges:', ranges)

    if args.mode == SolverMode.CR:
        bits, _ = LagrangianCR(grads=grads*(ranges**2), r=eps)
        config.quant_bits = bits + 1
    else:
        bits, _ = LagrangianEB(grads=grads*(ranges**2), ref=f64_data, tol_rel=eps)
        config.quant_bits = bits + 1
    config.quant_ranges = ranges
    print('bits:', config.quant_bits)
    print(f'real compression rate: {sum(config.quant_bits)/(32*n_vars)}')

def optimality_check(N=3, R=3, S=2):
    print(f'init bits: {config.quant_bits}')
    print(f'minus bits: {S}')
    a = np.arange(0, len(config.quant_bits), 1, dtype=int)

    np.random.shuffle(a)
    b = a[:R]
    for k in b:
        config.quant_bits[k] -= S
    print(f'quant bits: {config.quant_bits}')
    run(N=10, quant=True)


def optimality_check_move(S=2):
    print(f'init bits: {config.quant_bits}')
    print(f'move bits: {S}')
    for i in range(config.dim):
        config.quant_bits[i] -= S
        config.quant_bits[config.dim + i] += S

    for i in range(config.dim**2):
        config.quant_bits[config.dim * 2 + i] -= S
        config.quant_bits[config.dim * 2 + config.dim**2 + i] += S 

    print(f'quant bits: {config.quant_bits}')
    run(N=10, quant=True)


def optimality_check_move_cartesian(S1=2, S2=2):
    print(f'init bits: {config.quant_bits}')
    print(f'move bits: {S1}, {S2}')
    for i in range(config.dim):
        config.quant_bits[i] -= S1
        config.quant_bits[config.dim + i] += S1

    for i in range(config.dim**2):
        config.quant_bits[config.dim * 2 + i] -= S2
        config.quant_bits[config.dim * 2 + config.dim**2 + i] += S2

    print(f'quant bits: {config.quant_bits}')
    run(N=10, quant=True)


solve()
if args.task == 0:
    optimality_check(R=12, S=args.search_bits)
elif args.task == 1:
    optimality_check(R=6, S=args.search_bits)
elif args.task == 2:
    optimality_check_move_cartesian(args.search_bits, args.search_bits_2)
