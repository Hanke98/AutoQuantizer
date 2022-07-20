import taichi as ti
import sys
import os
import time
import argparse
sys.path.append('..')
from engine.mpm_solver import MPMSolver
from engine.ggui_renderer import MPMRenderer
import numpy as np 
from mpm_config import MPMConfig
from solvers.analytical_solver import LagrangianCR

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
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()
pack_mode = False
if args.forward:
    pack_mode = True

# ti.init(arch=ti.cuda, device_memory_GB=16, default_fp = ti.float32, print_ir=True)
# ti.init(arch=ti.cuda, packed=True, device_memory_GB=8, default_fp=ti.float32, kernel_profiler=True, print_ir=False)
ti.init(arch=ti.cuda, packed=pack_mode, device_memory_GB=18, default_fp=ti.float32, kernel_profiler=True, print_ir=False)

config = MPMConfig()

def run_forward_only(mpm, ggui=None, save_pics=False, save_states=False):
    frame = 0
    if save_pics:
        output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
        os.makedirs(output_path, exist_ok=True)
        ggui.init_ggui_window()
    if save_states:
        pos_output_folder = mpm.param.pos_output_folder
        os.makedirs(pos_output_folder, exist_ok=True)
    max_frame = int(mpm.n_timestep / mpm.steps + 1.)
    print(f'max_frame: {max_frame}')
    # mpm.load_states(pos_output_folder, 220)

    while frame < max_frame:
        if save_pics:
            if mpm.dim == 3: 
                ggui.collider_theta[None] = mpm.collider_theta[None]
                ggui.collider_height[None] = mpm.collider_height[None]
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
            mpm.step_collider()
            print('.', end='', flush=True)

        ti.sync()
        end = time.time()
        print('')
        mpm.compute_loss()
        print(f'simulation time: {(end - st) * 1000} ms', flush=True)
        print(f'theta: {mpm.collider_theta[None]}')
        print(f'height: {mpm.collider_height[None]}')
        # exit(-1)
        ti.print_kernel_profile_info()
        ti.print_memory_profile_info()

        if save_states and frame % 10 == 0 and frame >= 200: 
            mpm.dump_states(frame)
            mpm.checkpoint_states(frame)
        print(f'frame: {frame}', flush=True)
        frame += 1


def ad_or_ranges():
    config.dim = 3
    config.n_particles = 100**3
    config.dt = 4e-4
    config.E = 5e4
    config.n_grid = 64 

    config.n_timesteps = 1024 * 8 + 1
    config.id = 'rerun_fluid_grads_after_merge2'
    config.la = 5e5
    config.use_sdf_collider = True

    bound = [3, 3, 3]
    config.bound = bound

    config.output_folder = 'mpm_exp/pics/'

    config.use_difftaichi = True
    config.use_bls = False
    config.use_fluid = True
    config.use_friction = False
    config.p_rho = 1000.0

    dx = 1/config.n_grid
    config.p_vol = dx ** config.dim

    config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'J': 1})
    config.use_grad_clip = True
    config.ub = [config.n_grid * 2 - bound[0],
            config.n_grid - bound[1], 
            config.n_grid * 2 - bound[2]]

    sim = MPMSolver(config)

    sim.initialize()

    ub = [config.n_grid * 2 - bound[0] - 1, 
            config.n_grid - bound[1], 
            config.n_grid * 2 - bound[2] - 1]

    sim.add_cube((config.bound[0] * dx, config.bound[1] * dx, config.bound[2] * dx),
                (ub[0] * dx, 0.2, ub[2] * dx), config.n_particles//2)
    sim.add_cube((config.bound[0] * dx, config.bound[1] * dx + 0.2, config.bound[2] * dx),
                (ub[0] * dx, 0.2, ub[2] * dx), config.n_particles//2)
    if args.ad:
        sim.run_forward_with_ad_nlogn(save_pics=False)
    elif args.ranges:
        sim.run_forward_detect_ranges()


def solve():
    ranges = np.load('ranges/fluid_3d/fluid_3d.npy')
    ranges[config.dim*2:config.dim*2+config.dim**2] *= 24
    ranges[config.dim:config.dim*2] *= 1.5
    ranges[-1] *= 1.5
    ranges[:config.dim] = 2.0
    grads = np.load('grads/fluid_3d/fluid_3d.npy')
    eps = 0.512

    bits, _ = LagrangianCR(grads*(ranges**2), r=eps)
    config.quant_bits = bits + 1
    print(config.quant_bits)
    config.ranges = ranges


def forward():
    global emit_step, sample_density
    config.use_quantaichi = True
    config.use_difftaichi = False
    config.use_fluid = True
    config.use_dithering = True
    config.use_bitpack = True
    config.use_friction = False
    config.compute_fp = 'f32'
    config.use_bls = True
    config.use_bound_clip = True
    config.id = 'fluid_3d'

    config.dim = 3
    config.n_particles = int(4e8)
    config.dt = 1e-4
    config.E = 5e4
    config.n_grid = 128 * 2 
    config.n_timesteps = 1024 * 32 + 1
    config.la = 5e5
    config.use_sdf_collider = True
    config.p_rho = 1000.0
    config.p_vol = (4 * 4 * 0.3) / config.n_particles
    config.g = 9.8
    bound = [3, 3, 3]
    config.bound = bound
    config.lb = bound
    config.ub = [config.n_grid * 4 - bound[0], config.n_grid - bound[1], config.n_grid * 4 - bound[2]]

    config.output_folder = 'outputs/pics/'
    config.pos_output_folder = '/outputs/pos/fluid_3d/'
    config.use_quantaichi = True
    config.use_bitpack = True
    config.use_bls = True
    config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'J': 1})

    solve()
    # config.quant_bits = np.array([21, 22, 21, 15, 18, 15, 12, 16, 13, 14, 13, 14, 14, 15, 13, 17]) # cr = 0.5, analytical solution rerun grad and ranges

    ranges = np.load('ranges/fluid_3d/fluid_3d.npy')
    ranges[config.dim*2:config.dim*2+config.dim**2] *= 24
    ranges[config.dim:config.dim*2] *= 1.5
    ranges[-1] *= 1.5
    ranges[:config.dim] = 4.0
    ranges[1] = 1.0
    config.quant_ranges = ranges
    
    ggui = MPMRenderer(config.n_particles, draw_propeller=True)
    ggui.init_ggui_fields_values()

    mpm = MPMSolver(config)
    mpm.initialize()
    dx = 1/config.n_grid
    mpm.add_cube(((config.lb[0]+0.5) * dx, (config.lb[1]+0.5) * dx, (config.lb[2]+0.5) * dx),
                     ((config.ub[0] - config.lb[0] - 0.5) * dx, 0.3, (config.ub[0] - config.lb[0] -0.5) * dx,), config.n_particles)
    exit(-1)
    run_forward_only(mpm, ggui, True, False)


if args.forward:
    forward()
else:
    ad_or_ranges()