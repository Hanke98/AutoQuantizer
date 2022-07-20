import os
import time
import argparse
import sys

import numpy as np 
import taichi as ti

sys.path.append('..')
from engine.mpm_solver import MPMSolver
from mpm_config import MPMConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dithering',
                        action='store_true',
                        default=False,
                        help='use dithering')
    parser.add_argument('-r',
                        '--ref',
                        action='store_true',
                        default=False,
                        help='reference')
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
            # TODO: restore 2d gui
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
        frame += 1
    mpm.compute_loss()
    return mpm.loss[None]


def init_dithering(mpm):
    mpm.initialize()
    mpm.x.fill(0)
    mpm.v.fill(0)

    mpm.rp.seed(1)

    for i in range(4):
        mpm.add_cube(((0.125 + i * 0.2), 0.75),
                    (0.15, 0.15), config.n_particles//4)
    config.dt = 5e-5


def run(N=5, quant=False, save_pics=False):
    if quant:
        config.use_difftaichi=False
        config.use_bls=True
        config.use_bitpack=True
        config.use_quantaichi=True
    loss = []
    mpm = MPMSolver(config)
    for _ in range(N):
        init_dithering(mpm)
        loss.append(run_forward_only(mpm, None, save_pics, False))
    loss = np.array(loss)
    for l in loss:
        print(l)


def dithering():
    if args.ref:
        print("-----------Reference---------")
        config.id = 'dithering_exp_f64_ref'
        run(N=1, quant=False, save_pics=True)
    elif args.dithering:
        print("-----------Dithering---------")
        config.use_dithering = True 
        config.id = 'dithering'
        run(N=1, quant=True, save_pics=True)
    else:
        print("---------No Dithering---------")
        config.use_dithering = False
        config.id = 'no_dithering'
        run(N=1, quant=True, save_pics=True)


args = parse_args()

use_low_res = False
config = MPMConfig()
config.dim = 2
config.E = 5e1
config.id = 50
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

range_folder = 'ranges/elastic_2d/'
file_name = f'elastic_2d.npy'
ranges = np.load(f'{range_folder}/{file_name}')
ranges[:config.dim] = 1.0
ranges[config.dim:] *= 2.0

config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'F': 1})
config.quant_bits = np.array([19.0, 19.0, 13.0, 13.0, 9.0, 10.0, 10.0, 9.0, 14.0, 14.0, 14.0, 15.0])
config.quant_ranges = ranges

ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.float64, kernel_profiler=True, print_ir=False)

dithering()

