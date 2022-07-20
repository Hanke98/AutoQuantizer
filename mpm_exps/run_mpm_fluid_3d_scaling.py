import os
import time
import argparse
import sys
sys.path.append('..')
from engine.mpm_solver import MPMSolver
from engine.ggui_renderer import MPMRenderer
from engine.mesh_io import load_mesh
from mpm_config import MPMConfig
from solvers.scipy_solver import ScipySolverMemMode
from solvers.analytical_solver import LagrangianCR
import json

import numpy as np 
import taichi as ti

class SolverMode:
    CR = 0
    EB = 1

class ExpTask:
    SCALING = 0
    INIT_CONDITION = 1

def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-c',
                         '--config',
                         type=str,
                         help='config files')
    parser.add_argument('-q',
                        '--quant',
                        action='store_true',
                        default=False,
                        help='use quant')
    parser.add_argument('-sp', 
                        '--save-pics', 
                        action='store_true',
                        default=False,
                        help='save pic')
    parser.add_argument('-ss', 
                        '--save-states', 
                        action='store_true',
                        default=False,
                        help='save states')
    # 0 for compression rate; 1 for error bound 
    args = parser.parse_args()
    print(args)
    return args


def parse_json(fn, c):
    with open(fn, 'r') as f:
        data = json.load(f)
        c.n_particles = data['n_particles']
        c.dt = data['dt']
        c.n_timesteps = data['steps'] * 1024 + 1
        c.n_grid = data['grids']
        c.lb = data['lb']
        c.ub = data['ub']
        c.id = data['id']
        c.steps = data['steps'] // 16 * 128
        if 'task' in data:
            c.task = data['task']
        else:
            c.task = ExpTask.SCALING


def run_forward_only(mpm, ggui, save_pics=False, save_states=False):
    frame = 0
    if save_pics:
        if mpm.param.use_quantaichi:
            output_path = mpm.param.output_folder + "/quant" + str(mpm.param.id)
        else:
            output_path = mpm.param.output_folder + "/f64_ref" + str(mpm.param.id)
        os.makedirs(output_path, exist_ok=True)
        if ggui is not None:
            ggui.init_ggui_window(pos=(1.2, 0.9, 1.2), center=(0.5, 0.2, 0.5))
    if save_states:
        if mpm.param.use_quantaichi:
            pos_output_folder = f'{mpm.param.pos_output_folder}/quant_config{mpm.param.id}'
        else:
            pos_output_folder = f'{mpm.param.pos_output_folder}/f64_config{mpm.param.id}'
        os.makedirs(pos_output_folder, exist_ok=True)

    # max_frame = int(mpm.n_timestep / mpm.steps)# + 1.)
    # print(f'max_frame: {max_frame}')
    # while frame < max_frame:
    s = 0
    while s < mpm.n_timestep - 1:
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
                x_ = mpm.x.to_numpy(dtype=np.float32)
                x_ = x_[:, :, :2]# / 4.0
                gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0xDDDDDD, show_gui=False)
                gui.circles(x_[0], radius=2.5, color=0xED553B)
                gui.show(f'{output_path}/{frame:06d}.png')
                # assert False

        # st = time.time()
        # for k in range(mpm.steps):
        k = 0
        while k < mpm.steps and s < mpm.n_timestep - 1:
            if (config.id == 5) and s % emit_step == 0:
                mpm.emit_water([0.5, 0.3, 0.2], 0.04, [0.0, 0.0, 2.0], 1000)
                mpm.emit_water([0.5, 0.3, 0.8], 0.04, [0.0, 0.0, -2.0], 1000)
            if (config.id == 6) and s % emit_step == 0:
                mpm.emit_water([0.5, 0.3, 0.3], 0.04, [0.0, 0.0, 3.0], 1000)
                mpm.emit_water([0.5, 0.3, 0.7], 0.04, [0.0, 0.0, -3.0], 1000)
            if mpm.use_g2p2g:
                mpm.substep_g2p2g()
            else:
                mpm.substep(0, 0)
            k += 1
            s += 1
            # mpm.step_collider()
        print('.', end='', flush=True)
        # ti.sync()
        # end = time.time()
        # print('')
        # print(f'simulation time: {(end - st) * 1000} ms', flush=True)
        # print(f'theta: {mpm.collider_theta[None]}')
        # print(f'height: {mpm.collider_height[None]}')
        # exit(-1)
        # ti.print_kernel_profile_info()
        # ti.print_memory_profile_info()

        if save_states: 
            mpm.dump_states(frame, pos_output_folder)
        # print(f'frame: {frame}', flush=True)
        frame += 1
    mpm.compute_loss()
    return mpm.loss[None]

def run_forward_with_ad_nlogn(mpm, grad_fn):
    # if save_pics:
    #     output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
    #     os.makedirs(output_path, exist_ok=True)
    #     ggui.init_ggui_window(pos=(2, 0.6, 3.5), center=(2.0, 0.5, 2.0))

    mpm.checkpoint(mpm.logn_timestep-1, 0)
    with ti.Tape(loss=mpm.loss):
    # if True:
        for s in range(mpm.n_timestep - 1):
            # nlogn_step_forward(mpm, s, None, None)
            mpm.nlogn_loop_forward(s, False)
            if (s-1) % mpm.steps == 0:
            #     ggui.draw_ggui(mpm.x, 0, s // mpm.steps, output_path)
            #     if s == 0:
            #         ggui.draw_ggui(mpm.x, 0, s // mpm.steps, output_path)
                print('.', end='', flush=True)
            # print(f'step: {s}')
        mpm.compute_loss()
        print('\nforward end')
    mpm.dump_grad(0, grad_fn)
    print(f'loss= {mpm.loss[None]}', flush=True)
    return mpm.loss[None]

@ti.ad.grad_replaced
def nlogn_step_forward(mpm, s, emit_func=None, gui=None):
    k = s % 2
    frame = s // mpm.steps
    # if emit_func is not None and s % (emit_step * mpm.steps) == 0:
    #     emit_func(frame, mpm)
    mpm.substep_difftaichi(k, 1-k)
    # mpm.intro_quantization_errors(1-k)
    if mpm.use_sdf_collider:
        mpm.step_collider_difftaichi(s)

@ti.ad.grad_for(nlogn_step_forward)
def nlogn_step_forward_grad(mpm, s, emit_func=None, gui=None):
    if s % 100 == 0:
        print(f'back prop: {s}', flush=True)
    # output_path = mpm.param.output_folder + "/exp_" + str(mpm.param.id)
    # os.makedirs(output_path, exist_ok=True)
    w = mpm.cnt_ending_zero(s)
    mpm.rerun_from(s, w, nlogn_step_forward, emit_func, gui)

    c = s%2
    if s != mpm.n_timestep - 2:
        mpm.set_grad(1-c)

    nlogn_step_forward(mpm, s)
    # mpm.nlogn_step_forward(s)

    mpm.g2p.grad(c, 1-c)
    if mpm.use_sdf_collider:
        mpm.sdf_collision_no_arg.grad()
    mpm.grid_op.grad()
    mpm.p2g.grad(c)

    if s != mpm.n_timestep - 2:
        mpm.accu_grad(c)
    mpm.save_grad(c)
    if s != 0:
        mpm.clear_grad(c)

def run_forward_detect_ranges(mpm, fn):
    for s in range(mpm.n_timestep - 1):
        nlogn_step_forward(mpm, s, None, None)
        # mpm.detect_bound(1-s%2)
        mpm.detect_bound_kernel(1-s%2)
        if s % mpm.steps == 0:
            print('.', end='', flush=True)
    mpm.detect_bound()
    ranges = mpm.get_limits()
    print(ranges)
    np.save(fn, ranges)

def init(mpm):
    mpm.initialize()
    mpm.x.fill(0)
    mpm.v.fill(0)

    # mpm.rp.seed(1)
    
    if mpm.param.task == ExpTask.SCALING:
        # scaling exp
        mpm.add_cube(((config.lb[0]+0.5) * dx, (config.lb[1]+0.5) * dx, (config.lb[2]+0.5) * dx),
                     ((config.ub[0] - config.lb[0] + 1) * dx, 0.2, 0.3), config.n_particles)

    elif mpm.param.task == ExpTask.INIT_CONDITION:
        if config.id == 0:
            # initial condition exp 0:
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                        ((config.ub[0] - config.lb[0] + 1) * 0.5 * dx, 0.2, 0.3), config.n_particles)

        elif config.id == 1:
            # initial condition exp 1:
            mpm.add_cube((((config.lb[0] + config.ub[0]) * 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                         ((config.ub[0] - config.lb[0] + 1) * 0.5 * dx, 0.2, 0.3), config.n_particles)
        
        elif config.id == 2:
            # initial condition exp 2:
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                         ((config.ub[0] - config.lb[0] + 1) * dx, 0.3, 0.3), config.n_particles)
        
        elif config.id == 3:
            # initial condition exp 3:
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                        ((config.ub[0] - config.lb[0] + 1) * dx, 0.2, 0.3), config.n_particles //2)
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.ub[2] - 0.3 * config.n_grid - 0.5) * dx),
                        ((config.ub[0] - config.lb[0] + 1) * dx, 0.2, 0.3), config.n_particles //2)
        elif config.id == 4:
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                     ((config.ub[0] - config.lb[0] + 1) * dx, 0.2, 0.3), config.n_particles)
        elif config.id == 5 or config.id ==6:
            pass
        elif config.id == 7:
            mpm.add_cube(((config.lb[0] + 0.5) * dx, (config.lb[1] + 0.5) * dx, (config.lb[2] + 0.5) * dx),
                     ((config.ub[0] - config.lb[0] + 1) * dx, 0.2, 0.3), config.n_particles)
        else:
            assert False
    else:
        assert False

def run(N=5, quant=False, save_pics=False, save_states=False):
    if quant:
        config.use_difftaichi=False
        config.use_bitpack=True
        config.use_quantaichi=True
    config.use_bls=True
    ggui=None
    if save_pics:
        ggui = MPMRenderer(config.n_particles, draw_propeller=False, reduce_num=1)
    loss = []
    mpm = MPMSolver(config)
    for _ in range(N):
        init(mpm)
        loss.append(run_forward_only(mpm, ggui, save_pics, save_states))
        print(loss[-1])
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
    # file_name = f'elastic_3d_t{config.n_timesteps}_dt{config.dt}_p{config.n_particles}_g{config.n_grid}.npy'
    fn = f'{range_folder}/{file_name}'
    run_forward_detect_ranges(mpm, fn)

def solve():
    grads = np.load(f'{grad_folder}/{file_name}')
    range_file = 'ranges/fluid_3d_scaling/fluid_3d_t65537_dt5e-05_p5000000_g256.npy'
    ranges = np.load(f'{range_file}')
    ranges[:config.dim] = 1.0
    ranges[config.dim:] *= 2.0
    n_vars = grads.shape[0]
    print(f'n_vars: {n_vars}')

    config.quant_ranges = ranges
    
    assert args.mode == SolverMode.CR
    if config.id < 6:
        grads = grads/grads.min()
        solver = ScipySolverMemMode(n_vars, a=[1], eps=eps, grad=grads)
        solver.set_range(ranges)
        solver.run()
        config.quant_bits = solver.get_bits()
    else:
        g = grads * ranges**2
        bits, _ = LagrangianCR(grads=g, r=eps)
        config.quant_bits = bits + 1

    print('bits: ', config.quant_bits)
    print('ranges: ', config.quant_ranges)
    print(f'real compression rate: {sum(config.quant_bits)/(32*n_vars)}')


config = MPMConfig()
config.dim = 3
bound = [3,] * config.dim
config.bound = bound

config.lb = [15, 3, 3]
config.ub = [48, 60, 60]

config.use_bls = not config.use_difftaichi
config.use_friction = False
config.use_fluid = True

args = parse_args()

f64_data = 0.06709209246760206
eps = args.epsilon

if args.config is not None:
    parse_json(args.config, config)

config.la = 5e5
if config.id == 4 or config.id == 7:
    config.la = 2e6

print(f'particles: {config.n_particles}')
print(f'dt: {config.dt}')
print(f'n_timesteps: {config.n_timesteps}')
print(f'steps: {config.steps}')
print(f'grids: {config.n_grid}')
print(f'lb: {config.lb}')
print(f'ub: {config.ub}')
print(f'id: {config.id}')
print(f'la: {config.la}')
print(f'eps: {eps}')
# print(f'tar: {tar}')

if config.task == ExpTask.SCALING:
    config.output_folder = 'outputs/scaling_pics/'
    config.pos_output_folder = f'/data/auto_quantizer_output/mpm_exps/rerun2_scaling/'
else:
    config.output_folder = 'outputs/init_condition_pics/'
    config.pos_output_folder = f'/data/auto_quantizer_output/mpm_exps/rerun2_init_condition/'
emit_step = 30

config.E = 5e4
config.p_rho = 1000.0
config.p_vol = (1/config.n_grid * 0.5)**config.dim
config.size = 1.0
config.g = 9.8

config.quantized_properties = dict({'x': 1, 'v': 1, 'C': 1, 'J': 1})

dx = 1/config.n_grid
print(f'dx: {dx}')

ref_steps = 1024 * 16 + 1
ref_dt = 2e-4
ref_particles = 100**3 // 5
ref_grids = 64

grad_folder = 'grads/fluid_3d_scaling/'
range_folder = 'ranges/fluid_3d_scaling/'
file_name = f'fluid_3d_t{ref_steps}_dt{ref_dt}_p{ref_particles}_g{ref_grids}.npy'

ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.float64, kernel_profiler=True, print_ir=False)


# ad()
# exit(-1)
# detect_range()
# exit(-1)
solve()

quant = False
save_pics = False
save_states = False
N_run = 5
if args.quant:
    quant = True
else:
    N_run = 1
if args.save_pics:
    save_pics = True
    N_run = 1
if args.save_states:
    save_states = True
    N_run = 1

run(N=N_run, quant=quant, save_pics=save_pics, save_states=save_states)

