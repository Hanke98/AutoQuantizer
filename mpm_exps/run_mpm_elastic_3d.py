import os
import time
import sys
import argparse
sys.path.append('..')
from engine.mpm_solver import MPMSolver
from engine.ggui_renderer import MPMRenderer
from engine.mesh_io import load_mesh
from mpm_config import MPMConfig
from solvers.analytical_solver import LagrangianCR

import numpy as np 
import taichi as ti

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
    parser.add_argument('-f',
                        '--forward',
                        action='store_true',
                        default=False,
                        help='dithering')
    parser.add_argument('-d',
                        '--use-dithering',
                        action='store_true',
                        default=False,
                        help='dithering')
    parser.add_argument('-p',
                        '--use-bitpack',
                        action='store_true',
                        default=False,
                        help='bitpack')
    args = parser.parse_args()
    print(args)
    return args


def add_letters(mpm, frame, s=0):
    global num_emit

    # translation = ti.Vector([1.5, 1.2, 0.006 * (frame//emit_step) + 0.1])
    translation = ti.Vector([0.90, 1.6, 0.1 + 0.006 * (frame//emit_step)])
    if num_emit == 0:
        cnt = mpm.cnt_mesh_particles_num(letters, translation=translation, sample_density=sample_density)
        num_emit = cnt * 2
        print(f'num_emit: {num_emit}')
    assert num_emit != 0
    vel = ti.Vector([0.0, -4.0, 0.0])
    mpm.last_emit_idx[None] = (frame//emit_step) * num_emit
    mpm.current_idx[None] = mpm.last_emit_idx[None]
    mpm.add_mesh(letters, velocity=vel, translation=translation, sample_density=sample_density)
    translation = ti.Vector([2.2, 1.6, 0.1 + 0.006 * (frame//emit_step)])
    mpm.add_mesh(letters, velocity=vel, translation=translation, sample_density=sample_density)
    print(f'current_idx: {mpm.current_idx[None]}')


def add_letters_ad(mpm, frame, s=0):
    global num_emit
    translation = ti.Vector([1.5, 0.8, 0.01 * (frame//emit_step) + 0.1])
    if num_emit == 0:
        cnt = mpm.cnt_mesh_particles_num(letters, translation=translation, sample_density=sample_density)
        num_emit = cnt 
        print(f'num_emit: {num_emit}')
    mpm.last_emit_idx[None] = (frame//emit_step) * num_emit
    mpm.current_idx[None] = mpm.last_emit_idx[None]
    mpm.emit_letters_one_frame()
    print(f'current_idx: {mpm.current_idx[None]}')


def emit(frame, mpm, s=0):
    print(f'emit func: {frame}')
    if frame % emit_step == 0:
        if (frame//emit_step) * num_emit + num_emit < mpm.n_particles:
            if mpm.use_difftaichi:
                add_letters_ad(mpm, frame, s)
            else:
                add_letters(mpm, frame, s)


def run_forward_only(mpm, ggui=None, save_pics=False, save_states=False, detect_range=False):
    frame = 0
    if save_pics:
        output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
        os.makedirs(output_path, exist_ok=True)
        ggui.init_ggui_window(pos=(2, 0.9, 3.5), center=(2.0, 0.7, 2.0))
    if save_states:
        pos_output_folder = mpm.param.pos_output_folder
        os.makedirs(pos_output_folder, exist_ok=True)

    # add_letters(mpm, frame)
    # print(f'num_emit: {num_emit}')
    max_frame = int(mpm.n_timestep / mpm.steps + 1.)
    # max_frame = 100
    print(f'max_frame: {max_frame}')
    # max_frame = 320
    # frame = 256
    # max_frame = 50
    # frame = 0

    while frame < max_frame:

        emit(frame, mpm)

        if save_pics:
            if mpm.dim == 3: 
                ti.sync()
                st = time.time()
                ggui.draw_ggui(mpm.x, 0, frame, output_path)
                ti.sync()
                end = time.time()
                print(f'render time: {(end - st) * 1000} ms', flush=True)
                if frame == 0:
                    # draw a second time to make the lighting right
                    # TODO: fix here!
                    ggui.draw_ggui(mpm.x, 0, frame, output_path)
            else:
                # TODO: restore 2d gui
                assert False

        st = time.time()
        for _ in range(mpm.steps):
            mpm.substep(0, 0)
            print('.', end='', flush=True)

        ti.sync()
        end = time.time()
        print('')
        print(f'simulation time: {(end - st) * 1000} ms', flush=True)
        # exit(-1)
        ti.print_kernel_profile_info()
        ti.print_memory_profile_info()

        if save_states: 
            mpm.dump_states(frame)
            mpm.checkpoint_states(frame)
        print(f'frame: {frame}', flush=True)
        # mpm.compute_loss()
        if detect_range:
            mpm.detect_bound_kernel(0)
            mpm.detect_bound()
            ranges = mpm.get_limits()
            print(ranges, flush=True)
            os.makedirs('./mpm_exp/ranges/elastic_3d/', exist_ok=True)
            np.save(f'./mpm_exp/ranges/elastic_3d/elastic_p500_{frame}.npy', ranges)
        frame += 1


def run_forward_with_ad_nlogn(mpm, save_pics, ggui=None):
    if save_pics:
        output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
        os.makedirs(output_path, exist_ok=True)
        ggui.init_ggui_window(pos=(2, 0.6, 3.5), center=(2.0, 0.5, 2.0))

    mpm.checkpoint(mpm.logn_timestep-1, 0)
    with ti.Tape(loss=mpm.loss):
    # if True:
        for s in range(mpm.n_timestep - 1):
            nlogn_step_forward(mpm, s, emit, ggui)
            # if (s-1) % mpm.steps == 0:
            #     ggui.draw_ggui(mpm.x, 0, s // mpm.steps, output_path)
            #     if s == 0:
            #         ggui.draw_ggui(mpm.x, 0, s // mpm.steps, output_path)
                # print('.', end='', flush=True)
            # print(f'step: {s}')
        mpm.compute_loss()
        print('\nforward end')
    mpm.dump_grad(0, fn='mpm_exp/grads/elastic_letters_rerun_again/elastic_rerun_test.npy')
    print(mpm.x.grad.to_numpy()[0, 0])
    print(f'loss= {mpm.loss[None]}', flush=True)
    return mpm.loss[None]


@ti.ad.grad_replaced
def nlogn_step_forward(mpm, s, emit_func=None, gui=None):
    k = s % 2
    frame = s // mpm.steps
    if emit_func is not None and s % (emit_step * mpm.steps) == 0:
        emit_func(frame, mpm, s)
    mpm.substep_difftaichi(k, 1-k)
    if mpm.use_sdf_collider:
        mpm.step_collider_difftaichi(s)

@ti.ad.grad_for(nlogn_step_forward)
def nlogn_step_forward_grad(mpm, s, emit_func=None, gui=None):
    # print(f'back prop: {s}')
    # print(f'current_idx: {mpm.current_idx[None]}')
    if s % 100 == 0:
        print(f'back prop: {s}')
        print(mpm.get_gradients())

    output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
    os.makedirs(output_path, exist_ok=True)
    w = mpm.cnt_ending_zero(s)
    mpm.rerun_from_with_emit(s, w, nlogn_step_forward, emit_func, gui)

    c = s%2
    if s != mpm.n_timestep - 2:
        mpm.set_grad(1-c)

    # nlogn_step_forward(mpm, s, emit_func, gui)
    mpm.substep_difftaichi(c, 1-c)
    # print(f'back prop: {s}')
    # print('current_idx:', mpm.current_idx[None])
    frame = s // mpm.steps
    if s % (emit_step * mpm.steps) == 0:
        if (frame//emit_step) * num_emit + num_emit < mpm.n_particles:
            mpm.last_emit_idx[None] = (max(0, frame)//emit_step) * num_emit
            mpm.current_idx[None] = (max(0, frame)//emit_step) * num_emit

    # if s > 0:
    #     print(s // mpm.steps // emit_step * num_emit, mpm.current_idx[None])

    mpm.g2p.grad(c, 1-c)
    if mpm.use_sdf_collider:
        mpm.sdf_collision_no_arg.grad()
    mpm.grid_op.grad()
    mpm.p2g.grad(c)

    if s != mpm.n_timestep - 2:
        mpm.accu_grad(c)
        # print(mpm.x.grad.to_numpy()[c])
    mpm.save_grad(c)
    if s != 0:
        mpm.clear_grad(c)

    if s % mpm.steps == 0:
        k = s % 2
        gui.draw_ggui(mpm.x, k, (s//mpm.steps), output_path)
    # print(mpm.get_gradients())

args = parse_args()

default_fp = ti.f32
use_low_res = False
if args.ad:
    default_fp = ti.f64
    use_low_res = True

ti.init(arch=ti.cuda, packed=True, device_memory_GB=18, default_fp=default_fp, kernel_profiler=True, print_ir=False)


letters = load_mesh("./inputs/auto_quant.ply", scale=(0.1, 0.1, 2), offset=(0.02, 0.02, 0.5))
num_emit = 0

config = MPMConfig()

config.dim = 3
config.E = 5e5
config.la = 5e4
config.nu = 0.2
config.g = 9.8 * 2.5
config.bound = [3, 3, 3]

config.p_rho = 1000.0
config.p_vol = (1/config.n_grid * 0.5)**config.dim
config.use_fluid = False

# for low-res
if use_low_res:
    config.n_particles = 100**3
    config.dt = 1e-4
    config.n_grid = 128
    config.n_timesteps = 1024 * 16 + 1
    emit_step = 5
    sample_density = 1
else: # for high-res
    config.n_particles = int(3e8)
    config.dt = 7.5e-5
    config.E = 5e5
    config.n_grid = 128 * 2
    config.n_timesteps = 1024 * 40 + 1
    config.steps = 128
    emit_step = 4 
    sample_density = 8


config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'F': 1})
config.output_folder = 'outputs/pics/'
config.pos_output_folder = 'outputs/pos/'
config.p_rho = 1000.0
config.p_vol = (1/config.n_grid * 0.5)**config.dim
config.id = 'exp_3d'
config.use_dithering = True 
config.use_bitpack = True

def ad():
    global emit_step, sample_density
    config.id = 'elastic_3d_grad' 
    config.use_difftaichi = True
    ggui = MPMRenderer(config.n_particles)
    ggui.init_ggui_fields_values()

    mpm = MPMSolver(config)
    _x = np.load('inputs/letters_emit_one_frame.npy').astype(np.float32)
    mpm.set_emit_data(_x)
    mpm.initialize()
    run_forward_with_ad_nlogn(mpm, True, ggui)


def detect_range():
    config.id = 'elastic_3d_range' 
    config.use_quantaichi = False
    config.use_bls = True
    config.n_particles = 500**3
    # run only 16k steps to save time, 
    # you can try longer time duriation and enlarge the ranges by a smaller factor, e.g. 1.3, 1.5;
    config.n_timesteps = 1024 * 16 
    ggui = MPMRenderer(config.n_particles)
    ggui.init_ggui_fields_values()

    mpm = MPMSolver(config)
    mpm.initialize()
    run_forward_only(mpm, ggui, True, False, True)
    

def solve():
    ranges = np.load('ranges/elastic_3d/elastic_3d.npy')
    ranges[:] *= 2.0
    ranges[:config.dim] = 4.0
    config.quant_ranges = ranges
    grads = np.load('grads/elastic_3d/elastic_3d.npy')
    eps = 0.512

    bits, _ = LagrangianCR(grads*(ranges**2), r=eps)
    config.quant_bits = bits + 1
    print(config.quant_bits)


def forward():
    global emit_step, sample_density
    config.use_quantaichi = True
    config.use_difftaichi = False
    config.use_dithering = True
    config.use_bitpack = True
    config.compute_fp = 'f32'
    config.use_bls = True
    config.use_bound_clip = True

    config.id = 'rerun_elastic_3d_after_merge5'
    
    ggui = MPMRenderer(config.n_particles)
    ggui.init_ggui_fields_values()

    mpm = MPMSolver(config)
    mpm.initialize()
    run_forward_only(mpm, ggui, True, False)

def performance_benchmark():
    global emit_step, sample_density
    config.use_quantaichi = True
    config.use_difftaichi = False
    config.use_dithering = args.use_dithering
    config.use_bitpack = args.use_bitpack 
    config.compute_fp = 'f32'
    # config.use_bls = True
    config.use_bound_clip = True
    config.n_particles = int(3e7)
    config.n_timesteps = 12800
    config.id = 'exp_test_performance'

    ggui = MPMRenderer(config.n_particles)
    ggui.init_ggui_fields_values()

    ranges = np.load('ranges/elastic_3d/elastic_3d.npy')
    ranges[:] *= 2.0
    ranges[:config.dim] = 4.0
    config.quant_ranges = ranges

    # for performance benchmark
    config.quant_bits = np.array([24.0, 23.0, 23.0, 
                                    17.0, 17.0, 15.0, 
                                    11.0, 14.0, 11.0, 13.0, 11.0, 10.0, 11.0, 12.0, 11.0, 
                                    19.0, 18.0, 18.0, 18.0, 18.0, 17.0, 17.0, 17.0, 18.0])

    mpm = MPMSolver(config)
    mpm.initialize()
    run_forward_only(mpm, ggui, True, False)

if args.ad:
    ad()
elif args.ranges:
    detect_range()
elif args.forward:
    solve()
    forward()
else:
    performance_benchmark()
