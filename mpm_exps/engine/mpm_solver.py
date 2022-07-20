import taichi as ti
import time
import os
import numpy as np
from engine.mesh_io import load_mesh_vertices_and_indices, write_point_cloud_packed
from engine.mesh_io import load_particles_only
from engine.voxelizer import Voxelizer
from engine.ggui_renderer import MPMRenderer
from engine.random_pool import RandomPool
from engine.mesh_io import load_piont_cloud_packed


class ParticleStates:
    Deactivated=0
    Activated=1

@ti.data_oriented
class MPMSolver:
    def __init__(self, mpm_param):
        self.param = mpm_param
        self.dim = mpm_param.dim
        self.n_timestep = int(mpm_param.n_timesteps)
        self.n_particles, self.n_grid = int(mpm_param.n_particles), mpm_param.n_grid
        self.dx = 1/self.n_grid
        self.inv_dx = 1./self.dx
        self.grid_size = 4096

        self.p_rho = mpm_param.p_rho 
        self.p_vol = mpm_param.p_vol

        self.p_mass = self.p_vol * self.p_rho
        self.la = mpm_param.la
        
        self.E = mpm_param.E
        self.nu = mpm_param.nu
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.la_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters
        if mpm_param.softer:
            self.mu_0 *= 0.3
            self.la_0 *= 0.3
        print(f'la: {self.la_0}')
        print(f'mu: {self.mu_0}')

        self.dt = mpm_param.dt
        self.steps = mpm_param.steps
        self.current_step = 0
        self.clear_step = 32
        self.g = mpm_param.g

        self.use_g2p2g = mpm_param.use_g2p2g
        self.use_quantaichi = mpm_param.use_quantaichi
        self.use_hand_scheme = mpm_param.use_hand_scheme
        self.use_difftaichi = mpm_param.use_difftaichi
        print(f'use_difftaichi: {self.use_difftaichi}')
        self.use_bls = mpm_param.use_bls
        self.use_sdf_collider = mpm_param.use_sdf_collider
        print(f'use_sdf_collider: {self.use_sdf_collider}')
        self.use_bitpack = mpm_param.use_bitpack
        print(f'use_bitpack: {self.use_bitpack}')
        self.use_fluid = mpm_param.use_fluid
        print(f'use_fluid: {self.use_fluid}')
        self.use_dithering = mpm_param.use_dithering
        print(f'use_dithering: {self.use_dithering}')
        self.use_grad_clip = mpm_param.use_grad_clip
        print(f'use_grad_clip: {self.use_grad_clip}')
        self.use_height_loss = mpm_param.use_height_loss
        self.hand_scheme_exp = mpm_param.hand_scheme_exp
        self.use_bound_clip = mpm_param.use_bound_clip

        self.use_friction = mpm_param.use_friction
        self.refine_ranges = mpm_param.refine_ranges # for the experiment of comparison with hand tune scheme
        self.unbounded = False

        if self.use_quantaichi or self.hand_scheme_exp:
            bits = self.reshape_params(self.param.quant_bits)
            ranges = self.reshape_params(self.param.quant_ranges)
            self.bits = bits
            self.ranges = ranges
        self.init_sigma()

        self.init_particle_fields()
        self.init_sparse_grids()
        if self.use_difftaichi:
            self.init_difftaichi()

        self.sdf_offset = self.offset
        # self.init_sdf_collider()
        self.sdf_offset = [0, 0, 0]
        self.grid_postprocess = []
        if self.use_sdf_collider:
            self.init_sdf_collider()
            self.init_sdf_value()
            if self.use_difftaichi:
                self.grid_postprocess.append(self.sdf_collision_no_arg)
            else:
                self.grid_postprocess.append(self.sdf_collision)
        self.size = mpm_param.size
        self.bound = mpm_param.bound
        self.lb = [b + o for b, o in zip(self.bound, self.offset)]
        self.ub = [self.n_grid * self.size - b for b, o in zip(self.bound, self.offset)]

        if mpm_param.ub is not None:
            self.ub = mpm_param.ub
        if mpm_param.lb is not None:
            self.lb = mpm_param.lb
        print(f'ub: {self.ub}')

        self.g2p2g_allowed_cfl = 0.9

        self.init_limits()

        self.emit_data = ti.Vector.field(3, dtype=float, shape=63856)

        if self.use_fluid == False and self.dim == 3:
            self.voxelizer_super_sample = 2
            self.voxelizer = Voxelizer(res=((self.n_grid,)*3), dx=self.dx, super_sample=self.voxelizer_super_sample)
        if self.dim == 2:
            self.rp = RandomPool(self.n_particles * 3)

        # self.draw_propeller = False
        # self.init_ggui()
        # self.ggui_initialized = False
    
    @ti.kernel
    def set_emit_data(self, data: ti.ext_arr()):
        for i in range(63856):
            for k in ti.static(range(self.dim)):
                self.emit_data[i][k] = data[i, k]

    @ti.kernel
    def emit_letters_one_frame(self):
        for i in self.emit_data:
            self.x[0, i+self.current_idx[None]] = self.emit_data[i]
            self.v[0, i+self.current_idx[None]] = [0.0, -2.0, 0.0]
        self.current_idx[None] += 63856

    def init_difftaichi(self):
        self.logn_timestep = int(np.ceil(np.log2(self.n_timestep))+1)
        N_step = 2

        self.current_idx_stash = ti.field(dtype=int)
        self.last_emit_idx_stash = ti.field(dtype=int)
        self.x_stash = ti.Vector.field(self.dim, dtype=float)
        self.v_stash = ti.Vector.field(self.dim, dtype=float)
        self.C_stash = ti.Matrix.field(self.dim, self.dim, dtype=float)
        if ti.static(self.use_fluid):
            self.J_stash = ti.field(dtype=float)
        else:
            self.F_stash = ti.Matrix.field(self.dim, self.dim, dtype=float)

        particles_stash_frames = ti.root.dense(ti.i, self.logn_timestep-1)
        particles_stash = particles_stash_frames.dense(ti.j, self.n_particles)
        if ti.static(self.use_fluid):
            particles_stash.place(self.x_stash, self.v_stash, self.J_stash, self.C_stash)
        else:
            particles_stash.place(self.x_stash, self.v_stash, self.F_stash, self.C_stash)
        
        particles_stash_frames.place(self.current_idx_stash, self.last_emit_idx_stash)

        self.grad_x = ti.Vector.field(self.dim, dtype=float)
        self.grad_v = ti.Vector.field(self.dim, dtype=float)
        self.grad_C = ti.Matrix.field(self.dim, self.dim, dtype=float)
        if ti.static(self.use_fluid):
            self.grad_J = ti.field(dtype=float)
        else:
            self.grad_F = ti.Matrix.field(self.dim, self.dim, dtype=float)

        grads_tmp = ti.root.dense(ti.i, self.n_particles)
        if ti.static(self.use_fluid):
            grads_tmp.place(self.grad_x, self.grad_v, self.grad_J, self.grad_C)
        else:
            grads_tmp.place(self.grad_x, self.grad_v, self.grad_F, self.grad_C)

        self.grad_sum_x = ti.Vector.field(self.dim, dtype=ti.f64)
        self.grad_sum_v = ti.Vector.field(self.dim, dtype=ti.f64)
        self.grad_sum_C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64)
        if ti.static(self.use_fluid):
            self.grad_sum_J = ti.field(dtype=ti.f64)
        else:
            self.grad_sum_F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64)
        self.grad_sum = ti.root.dense(ti.i, self.n_particles)
        if ti.static(self.use_fluid):
            self.grad_sum.place(self.grad_sum_x, self.grad_sum_v, self.grad_sum_J, self.grad_sum_C)
        else:
            self.grad_sum.place(self.grad_sum_x, self.grad_sum_v, self.grad_sum_F, self.grad_sum_C)

        ti.root.lazy_grad()
        self.last_step = (self.n_timestep - 1) % 2

    def init_particle_fields(self):
        N = 2 if self.use_difftaichi else 1
        self.particles = ti.root.dense(ti.i, N).dense(ti.j, self.n_particles)
        # self.particles = ti.root.dense(ti.i, self.n_particles)
        if self.use_quantaichi:
            self.init_quantaichi()
        elif self.use_hand_scheme:
            self.init_hand_scheme()
        else:
            self.x = ti.Vector.field(self.dim, dtype=float)
            self.v = ti.Vector.field(self.dim, dtype=float)
            self.C = ti.Matrix.field(self.dim, self.dim, dtype=float)
            if ti.static(self.use_fluid):
                self.J = ti.field(dtype=float)
                self.particles.place(self.x, self.v, self.C, self.J)
            else:
                self.F = ti.Matrix.field(self.dim, self.dim, dtype=float)
                self.particles.place(self.x, self.v, self.C, self.F)

        self.current_idx = ti.field(dtype=int, shape=())
        self.loss = ti.field(dtype=float, shape=())
        self.last_emit_idx = ti.field(dtype=int, shape=())
        self.vol_avg = ti.field(dtype=float, shape=())
        self.vel_avg = ti.Vector.field(self.dim, dtype=float, shape=())
        self.x_avg = ti.Vector.field(self.dim, dtype=float, shape=())

    def init_sparse_grids(self):
        indices = ti.ijk if self.dim == 3 else ti.ij
        grid_block_size = 128
        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            self.leaf_block_size = 4

        def block_component(block, indices, c):
            block.dense(indices, self.leaf_block_size).place(c)

        self.offset = [0,] * self.dim 


        if not self.use_g2p2g:
            self.grid_m = ti.field(dtype=float)
            self.grid_v = ti.Vector.field(self.dim, dtype=float)

            self.grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
            self.block = self.grid.pointer(indices, grid_block_size // self.leaf_block_size)

            block_component(self.block, indices, self.grid_m)
            for i in range(self.dim):
                block_component(self.block, indices, self.grid_v.get_scalar_field(i))

            # for auto diff 
            if self.use_difftaichi:
                self.grid_v1 = ti.Vector.field(self.dim, dtype=float)
                for i in range(self.dim):
                    block_component(self.block, indices, self.grid_v1.get_scalar_field(i))
                if self.use_sdf_collider:
                    self.grid_v2 = ti.Vector.field(self.dim, dtype=float)
                    for i in range(self.dim):
                        block_component(self.block, indices, self.grid_v2.get_scalar_field(i))

            self.pid = ti.field(dtype=ti.i32)
            self.block.dynamic(ti.axes(self.dim), 1024 * 32, 
                                chunk_size=self.leaf_block_size**self.dim * 8).place(self.pid)
        else:
            self.grid_v = []
            self.grid_m = []
            self.grids = []
            self.pid = []
            self.pid_block = []

            self.curr_grid_idx = 0
            
            for _ in range(2):
                grid_m = ti.field(dtype=float)
                grid_v = ti.Vector.field(self.dim, dtype=float)
                grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
                block = grid.pointer(indices, grid_block_size // self.leaf_block_size)
                block_component(block, indices, grid_m)
                for i in range(self.dim):
                    block_component(block, indices, grid_v.get_scalar_field(i))
                pid = ti.field(dtype=ti.i32)
                block.dynamic(ti.axes(self.dim), 1024 * 32,
                                chunk_size=self.leaf_block_size**self.dim * 8).place(pid)
                # pid_block = block.pointer(indices, self.leaf_block_size)
                # pid_block.dynamic(ti.axes(self.dim), 1024 * 256, 
                                                            # chunk_size=8).place(pid)
                self.pid.append(pid)
                self.grid_v.append(grid_v)
                self.grid_m.append(grid_m)
                self.grids.append(grid)
                # self.pid_block.append(pid_block)
            # print(f'grid_m.shape: {self.grid_m[0].shape}')

    def init_sdf_collider(self):
        indices = ti.ijk
        grid_block_size = 128
        self.collider_sdf = ti.field(dtype=float)
        self.collider_sdf_normal = ti.Vector.field(3, dtype=float)

        grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
        block = grid.pointer(indices, grid_block_size // self.leaf_block_size)
        block.dense(indices, self.leaf_block_size).place(self.collider_sdf, self.collider_sdf_normal)

        self.collider_theta = ti.field(dtype=float, shape=())
        self.collider_omega = ti.field(dtype=float, shape=())
        self.collider_height = ti.field(dtype=float, shape=())
    
    def init_sdf_value(self):
        if not self.use_difftaichi:
            _sdf = np.load('inputs/propeller_2x2_larger.npz')['sdf']
            _sdf_normal = np.load('inputs/propeller_2x2_larger_normal.npz')['sdf']
        else:
            _sdf = np.load('inputs/propeller_higher.npz')['sdf']
            _sdf_normal = np.load('inputs/propeller_higher_normal.npz')['sdf']
        _sdf_max = _sdf.max() 
        print(f'sdfmax: {_sdf_max}')

        shape = _sdf.shape

        @ti.kernel
        def _init_sdf_value(data: ti.ext_arr(), normal: ti.ext_arr()):
            for i, j, k in ti.ndrange((0, shape[0]), (0, shape[1]), (0, shape[2])):
                if data[i, j, k] < _sdf_max:
                    idx = ti.Vector([i + self.sdf_offset[0], j + self.sdf_offset[1], k + self.sdf_offset[2]])
                    self.collider_sdf[idx] = data[i, j, k]
                    for l in ti.static(range(3)):
                        self.collider_sdf_normal[idx][l] = normal[i, j, k, l]

        _init_sdf_value(_sdf, _sdf_normal)
        del _sdf
        del _sdf_normal
        # self.collider_height[None] =  0.2867415249347687
        # self.collider_height[None] = -0.12

    def init_hand_scheme(self):
        compute_fp = ti.f32
        if not self.refine_ranges:
            cft = ti.quant.fixed(20, range=2.0, signed=False, compute=compute_fp, use_dithering=self.use_dithering)
            # cft = ti.quant.fixed(20, range=1.0, signed=False, compute=compute_fp, use_dithering=self.use_dithering)
            self.x = ti.Vector.field(self.dim, dtype=cft) # position
            self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(0), self.x.get_scalar_field(1))
        
            cft = ti.quant.float(exp=6, frac=10, signed=True, compute=compute_fp)
            # cft = ti.quant.fixed(16, range=18.0, compute=compute_fp, use_dithering=self.use_dithering)
            self.v = ti.Vector.field(self.dim, dtype=cft) # velocity
            self.particles.bit_struct(num_bits=32).place(self.v.get_scalar_field(0), self.v.get_scalar_field(1))

            cft = ti.quant.fixed(16, range=4.0, compute=compute_fp, use_dithering=self.use_dithering)
            self.F = ti.Matrix.field(2, 2, dtype=cft) # deformation gradient
            self.particles.bit_struct(num_bits=32).place(self.F.get_scalar_field(0, 0), self.F.get_scalar_field(0, 1))
            self.particles.bit_struct(num_bits=32).place(self.F.get_scalar_field(1, 0), self.F.get_scalar_field(1, 1))
        else:
            cft = ti.quant.fixed(20, range=1.0, signed=False, compute=compute_fp, use_dithering=self.use_dithering)
            self.x = ti.Vector.field(self.dim, dtype=cft) # position
            self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(0), self.x.get_scalar_field(1))
        
            cft = ti.quant.fixed(16, range=18.0, compute=compute_fp, use_dithering=self.use_dithering)
            self.v = ti.Vector.field(self.dim, dtype=cft) # velocity
            self.particles.bit_struct(num_bits=32).place(self.v.get_scalar_field(0), self.v.get_scalar_field(1))

            F_types = []
            for i in range(4):
                r = self.ranges[2][i]
                _type = ti.quant.fixed(16, range=r, compute=compute_fp, use_dithering=self.use_dithering)
                F_types.append(_type)
            self.F = ti.Matrix.field(2, 2, dtype=np.reshape(F_types, (self.dim, self.dim))) # deformation gradient
            self.particles.bit_struct(num_bits=32).place(self.F.get_scalar_field(0, 0), self.F.get_scalar_field(0, 1))
            self.particles.bit_struct(num_bits=32).place(self.F.get_scalar_field(1, 0), self.F.get_scalar_field(1, 1))

        self.C = ti.Matrix.field(self.dim, self.dim, float)
        self.particles.place(self.C)
        print("use_hand_quantaichi")

    def init_quantaichi(self):
        print(f'use quant!')

        bits = self.bits
        ranges = self.ranges

        use_dithering = self.param.use_dithering
        compute_fp = ti.f64
        if self.param.compute_fp == 'f32':
            compute_fp = ti.f32

        x_types = []
        v_types = []
        C_types = []
        F_types = []
        idx = 0
        for b, r in zip(bits[idx], ranges[idx]):
            _type = ti.quant.fixed(frac=int(b), range=r, signed=True, compute=compute_fp, use_dithering=use_dithering)
            x_types.append(_type)
        self.x = ti.Vector.field(self.dim, x_types)
        idx += 1

        for b, r in zip(bits[idx], ranges[idx]):
            _type = ti.quant.fixed(frac=int(b), range=r, signed=True, compute=compute_fp, use_dithering=use_dithering)
            v_types.append(_type)
        self.v = ti.Vector.field(self.dim, v_types)
        idx += 1 

        if not self.use_g2p2g:
            if "C" in self.param.quantized_properties:
                for b, r in zip(bits[idx], ranges[idx]):
                    _type = ti.quant.fixed(frac=int(b), range=r, signed=True, compute=compute_fp, use_dithering=use_dithering)
                    C_types.append(_type)
                self.C = ti.Matrix.field(self.dim, self.dim, np.reshape(C_types, (self.dim,self.dim)))
                idx += 1

        if not self.use_fluid:
            for b, r in zip(bits[idx], ranges[idx]):
                _type = ti.quant.fixed(frac=int(b), range=r, signed=True, compute=compute_fp, use_dithering=use_dithering)
                F_types.append(_type)
            self.F = ti.Matrix.field(self.dim, self.dim, np.reshape(F_types, (self.dim, self.dim)))
            idx += 1
        # self.C = ti.Matrix.field(self.dim, self.dim, float)
        else:
            J_type = ti.quant.fixed(frac=bits[idx], range=ranges[idx], signed=True, compute=compute_fp, offset=1.0, use_dithering=use_dithering)
            self.J = ti.field(dtype=J_type)

        if not self.use_bitpack:
            if self.use_fluid:
                self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(0), self.x.get_scalar_field(1), self.x.get_scalar_field(2))
                self.particles.bit_struct(num_bits=64).place(self.v.get_scalar_field(0), self.v.get_scalar_field(1), self.v.get_scalar_field(2))
                if not self.use_g2p2g:
                    self.particles.bit_struct(num_bits=64).place(self.C.get_scalar_field(0, 0), self.C.get_scalar_field(0, 1), 
                                                                    self.C.get_scalar_field(0, 2), self.C.get_scalar_field(1, 0)) 
                    self.particles.bit_struct(num_bits=64).place(self.C.get_scalar_field(1, 1), self.C.get_scalar_field(1, 2), 
                                                                    self.C.get_scalar_field(2, 0), self.C.get_scalar_field(2, 1))
                    self.particles.bit_struct(num_bits=32).place(self.C.get_scalar_field(2, 2), self.J)
            else:
                if self.dim == 2:
                    self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(0), self.x.get_scalar_field(1), self.v.get_scalar_field(0))
                    self.particles.bit_struct(num_bits=64).place(self.v.get_scalar_field(1), self.C.get_scalar_field(0, 0), self.C.get_scalar_field(0, 1), self.C.get_scalar_field(1, 0), self.C.get_scalar_field(1, 1))
                    self.particles.bit_struct(num_bits=64).place(self.F.get_scalar_field(0, 0), self.F.get_scalar_field(0, 1), self.F.get_scalar_field(1, 0))
                    self.particles.bit_struct(num_bits=32).place(self.F.get_scalar_field(1, 1))
                else:
                    self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(0), self.x.get_scalar_field(1))
                    self.particles.bit_struct(num_bits=64).place(self.x.get_scalar_field(2), self.v.get_scalar_field(0), self.v.get_scalar_field(1))
                    if not self.use_g2p2g:
                        self.particles.bit_struct(num_bits=64).place(self.v.get_scalar_field(2), self.C.get_scalar_field(0, 0), self.C.get_scalar_field(0, 1), 
                                                                        self.C.get_scalar_field(0, 2), self.C.get_scalar_field(1, 0)) 
                        self.particles.bit_struct(num_bits=64).place(self.C.get_scalar_field(1, 1), self.C.get_scalar_field(1, 2), 
                                                                        self.C.get_scalar_field(2, 0), self.C.get_scalar_field(2, 1), self.C.get_scalar_field(2, 2))
                        self.particles.bit_struct(num_bits=64).place( self.F.get_scalar_field(0, 0), 
                                                                    self.F.get_scalar_field(0, 1),  self.F.get_scalar_field(0, 2))
                        self.particles.bit_struct(num_bits=64).place(self.F.get_scalar_field(1, 0), self.F.get_scalar_field(1, 1),
                                                                        self.F.get_scalar_field(1, 2)) 
                        self.particles.bit_struct(num_bits=64).place( self.F.get_scalar_field(2, 0), self.F.get_scalar_field(2, 1), self.F.get_scalar_field(2, 2))
                    else:
                        self.particles.bit_struct(num_bits=64).place(self.F.get_scalar_field(0, 0), self.F.get_scalar_field(0, 1), self.F.get_scalar_field(0, 2))
                        self.particles.bit_struct(num_bits=64).place(self.F.get_scalar_field(1, 0), self.F.get_scalar_field(1, 1), self.F.get_scalar_field(1, 2))
                        self.particles.bit_struct(num_bits=64).place(self.F.get_scalar_field(2, 0), self.F.get_scalar_field(2, 1), self.F.get_scalar_field(2, 2))

        else:
            if self.use_fluid:
                self.particles.bit_pack().place(self.x, self.v, self.C, self.J)
            else:
                if 'C' in self.param.quantized_properties:
                    self.particles.bit_pack().place(self.x, self.v, self.C, self.F)
                else:
                    self.particles.bit_pack().place(self.x, self.v, self.F)
        if 'C' not in self.param.quantized_properties:
            self.C = ti.Matrix.field(self.dim, self.dim, float)
            self.particles.place(self.C)

    def init_sigma(self):
        self.sigma_vec = ti.Vector.field(self.dim, dtype=float, shape=(2))
        self.sigma_mat = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=(2))

    @ti.kernel
    def set_sigma_vec(self, r: ti.template(), b: ti.template(), idx: ti.i32):
        for i in ti.static(range(self.dim)):
            self.sigma_vec[idx][i] = r[i] / (2 ** (b[i]-1)) / (2 * ti.sqrt(3))
    
    @ti.kernel
    def set_sigma_mat(self, r: ti.template(), b: ti.template(), idx: ti.i32):
        for i in ti.static(range(self.dim)):
            for j in ti.static(range(self.dim)):
                self.sigma_mat[idx][i, j] = r[i, j] / (2 ** (b[i, j]-1)) / (2 * ti.sqrt(3))

    def create_r_and_b(self, bits, ranges, idx):
        if self.dim == 2:
            b = ti.Vector([bits[idx][0], bits[idx][1]])
            r = ti.Vector([ranges[idx][0], ranges[idx][1]])
        else:
            b = ti.Vector([bits[idx][0], bits[idx][1], bits[idx][2]])
            r = ti.Vector([ranges[idx][0], ranges[idx][1], bits[idx][2]])
        return r, b
    
    def create_r_and_b_mat(self, bits, ranges, idx):
        if self.dim == 2:
            r = ti.Matrix([[ranges[idx][0], ranges[idx][1]], 
                            [ranges[idx][2], ranges[idx][3]]])
            b = ti.Matrix([[bits[idx][0], bits[idx][1]], 
                            [bits[idx][2], bits[idx][3]]])
            return r, b
        else:
            r = ti.Matrix([[ranges[idx][0], ranges[idx][1], ranges[idx][2]], 
                            [ranges[idx][3], ranges[idx][4], ranges[idx][5]],
                            [ranges[idx][6], ranges[idx][7], ranges[idx][8]]])
            b = ti.Matrix([[bits[idx][0], bits[idx][1], bits[idx][2]], 
                            [bits[idx][3], bits[idx][4], bits[idx][5]],
                            [bits[idx][6], bits[idx][7], bits[idx][8]]])
            return r, b

    def set_bits_and_ranges(self, bits, ranges):
        for i in range(len(bits)):
            bits[i] = max(min(bits[i], 32), 1)
        bits = self.reshape_params(bits)
        ranges = self.reshape_params(ranges)
        self.bits = bits
        self.ranges = ranges
        idx = 0
        if 'x' in self.param.quantized_properties:
            r, b = self.create_r_and_b(bits, ranges, idx)
            self.set_sigma_vec(r, b, 0)
            idx += 1
        if 'v' in self.param.quantized_properties:
            r, b = self.create_r_and_b(bits, ranges, idx)
            self.set_sigma_vec(r, b, 1)
            idx += 1
        if 'C' in self.param.quantized_properties:
            r, b = self.create_r_and_b_mat(bits, ranges, idx)
            # self.set_sigma_mat(r, b, 0)
            idx += 1
        if 'F' in self.param.quantized_properties:
            r, b = self.create_r_and_b_mat(bits, ranges, idx)
            # self.set_sigma_mat(r, b, 1)
            idx += 1

    def reshape_params(self, ranges):
        r = []
        if 'x' in self.param.quantized_properties:
            r.append(np.array(ranges[:self.dim]))
        if 'v' in self.param.quantized_properties:
            r.append(np.array(ranges[self.dim : self.dim + self.dim]))
        if 'C' in self.param.quantized_properties:
            mat = ranges[self.dim*2: self.dim * 2 + self.dim ** 2]
            r.append(np.array(mat))
        if 'J' in self.param.quantized_properties:
            r.append(np.array(ranges[-1:]))
        if 'F' in self.param.quantized_properties:
            r.append(np.array(ranges[-self.dim**2:]))
        return r

    def init_limits(self):
        # TODO: use taichi kernel to do this
        self.v_limits = np.ones(self.dim) * -np.inf
        self.C_limits = np.ones((self.dim, self.dim)) * -np.inf
        self.F_limits = np.ones((self.dim, self.dim)) * -np.inf
        self.C_limits = self.C_limits.flatten()
        self.F_limits = self.F_limits.flatten()
        self.J_limits = np.ones(1) * -np.inf

        self.v_max = ti.Vector.field(self.dim, dtype=float, shape=())
        self.v_min = ti.Vector.field(self.dim, dtype=float, shape=())
        self.C_max = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=())
        self.C_min = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=())
        self.J_max = ti.field(dtype=float, shape=())
        self.J_min = ti.field(dtype=float, shape=())
        self.F_max = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=())
        self.F_min = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=())
        
        self.v_max.fill(-np.inf)
        self.v_min.fill(np.inf)
        self.C_max.fill(-np.inf)
        self.C_min.fill(np.inf)
        self.J_max[None] = -np.inf
        self.J_min[None] = np.inf
        self.F_max.fill(-np.inf)
        self.F_min.fill(np.inf)
 
    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            if ti.static(self.use_fluid):
                self.J[0, i] = 1.0
            else:
                self.F[0, i] = ti.Matrix.diag(self.dim, 1.0)
            self.C[0, i] = ti.Matrix.zero(float, self.dim, self.dim)
        self.current_idx[None] = 0
        self.loss[None] = 0.0
        self.vol_avg[None] = 0.0
        self.vel_avg[None].fill(0.0) 
        self.x_avg[None].fill(0.0) 

    @ti.kernel
    def build_pid(self, s: ti.i32):
        ti.block_dim(64)
        # for p in range(self.n_particles):
        for p in range(self.current_idx[None]):
            base = int(ti.floor(self.x[s, p] * self.inv_dx - 0.5))
            # pid grandparent is `block`
            base_pid = ti.rescale_index(self.grid_m, self.pid.parent(2), base)
            # for k in ti.static(range(3)):
                # base_pid[k] = ti.max(ti.min(base_pid[k], 128), 0)
            ti.append(self.pid.parent(), base_pid, p)

    @ti.kernel
    def p2g_bls(self, s: ti.i32, t: ti.i32):
        ti.block_dim(256)
        if ti.static(self.use_bls):
            for i in ti.static(range(self.dim)):
                ti.block_local(self.grid_v.get_scalar_field(i))
            ti.block_local(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(int)
            Im = ti.rescale_index(self.pid, self.grid_m, I)
            for D in ti.static(range(self.dim)):
                # for block shared memory: hint compiler that there is a connection between `base` and loop index `I`
                base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)

            fx = self.x[s, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            # c_trace = self.C[s, p].trace()
            # self.J[t, p] = (1. + self.dt * c_trace) * self.J[s, p]

            cauchy = self.compute_cauchy(s, p)
            # cauchy = ti.Matrix.diag(self.dim, self.la * (self.J[s, p] - 1.0))
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + self.p_mass * self.C[s, p]

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[s, p] +
                                                        affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    def substep(self, s, t=None):
        self.grid.deactivate_all()
        self.build_pid(s)
        self.p2g_bls(s, t)
        self.grid_op()
        for p in self.grid_postprocess:
            p(self.grid_v)
        self.g2p_bls(s, t)

    @ti.kernel
    def grid_op(self):
        # for i, j in grid_v:
        grid_v_in = ti.static(self.grid_v) 
        grid_v_out = ti.static(self.grid_v1 if self.use_difftaichi else self.grid_v) 

        for I in ti.grouped(self.grid_m):
            inv_m = 1 / (self.grid_m[I] + 1e-10)
            v_out = inv_m * grid_v_in[I]
            v_out[1] -= self.dt * self.g

            if ti.static(self.use_friction):
                n = ti.Vector([0, 1, 0])
                if I[1] < self.lb[1] and v_out[1] < 0:
                    v = v_out
                    normal_component = n.dot(v)
                    vt_damp = 0.02
                    v = (v - n * normal_component) * vt_damp 
                #     # friction = 0.5
                #     # v = v * 0.02
                #     # v = v.normalized() * ti.max(0, v.norm() + ti.min(normal_component, 0) * friction)
                #     # if normal_component < 0 and v.norm() > 1e-30:
                #     # apply friction here
                #         # v = v.normalized() * ti.max(0, v.norm() + normal_component * friction)
                    v_out = v

            for k in ti.static(range(self.dim)):
                if I[k] < self.lb[k] and v_out[k] < 0:
                    v_out[k] = 0
                # if I[k] > self.n_grid - self.bound[k] and v_out[k] > 0:
                if I[k] > self.ub[k] and v_out[k] > 0:
                    v_out[k] = 0
            grid_v_out[I] = v_out

    @ti.kernel
    def g2p_bls(self, s: ti.i32, t: ti.i32):
        grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v1)
        ti.block_dim(256)
        if ti.static(self.use_bls):
            for i in ti.static(range(self.dim)):
                ti.block_local(grid_v.get_scalar_field(i))
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(int)
            Im = ti.rescale_index(self.pid, self.grid_m, I)
            for D in ti.static(range(self.dim)):
                # for block shared memory: hint compiler that there is a connection between `base` and loop index `I`
                base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
            fx = self.x[s, p] * self.inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, self.dim)
            new_C = ti.Matrix.zero(float, self.dim, self.dim)

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = grid_v[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            new_x = self.x[s, p] + self.dt * new_v

            if ti.static(self.use_bound_clip):
                for k in ti.static(range(self.dim)):
                    new_x[k] = ti.max(ti.min(new_x[k], self.ub[k] * self.dx), self.lb[k] * self.dx)

            if ti.static(self.use_bitpack):
                if ti.static(self.use_fluid):
                    new_J = (1. + self.dt * new_C.trace()) * self.J[s, p]
                    if ti.static(self.dim == 2):
                        ti.bit_pack_assign(self.x.parent(), [t, p], [new_x[0], new_x[1],
                                                                new_v[0], new_v[1], 
                                                                new_C[0, 0], new_C[0, 1],
                                                                new_C[1, 0], new_C[1, 1],
                                                                new_J])
                    else:
                        ti.bit_pack_assign(self.x.parent(), [t, p], [new_x[0], new_x[1], new_x[2],
                                                                new_v[0], new_v[1], new_v[2], 
                                                                new_C[0, 0], new_C[0, 1], new_C[0, 2],
                                                                new_C[1, 0], new_C[1, 1], new_C[1, 2],
                                                                new_C[2, 0], new_C[2, 1], new_C[2, 2],
                                                                new_J])
                else: 

                    new_F = (ti.Matrix.identity(float, self.dim) + self.dt * new_C) @ self.F[s, p]
                    # J = new_F.determinant()
                    # new_F = ti.Matrix([[J, 0.0], [0.0, 1.0]])
                    # sqrtJ = ti.sqrt(J)
                    # new_F = ti.Matrix.diag(self.dim, sqrtJ)
                    if ti.static(self.hand_scheme_exp):
                        ti.bit_pack_assign(self.x.parent(), [t, p], [new_x[0], new_x[1],
                                                                new_v[0], new_v[1], 
                                                                new_F[0, 0], new_F[0, 1], 
                                                                new_F[1, 0], new_F[1, 1]])
                        self.C[s, p] = new_C
                    elif ti.static(self.dim == 2):
                        ti.bit_pack_assign(self.x.parent(), [t, p], [new_x[0], new_x[1],
                                                                new_v[0], new_v[1], 
                                                                new_C[0, 0], new_C[0, 1], 
                                                                new_C[1, 0], new_C[1, 1], 
                                                                new_F[0, 0], new_F[0, 1], 
                                                                new_F[1, 0], new_F[1, 1]])
                    else:
                        ti.bit_pack_assign(self.x.parent(), [t, p], [new_x[0], new_x[1], new_x[2],
                                                                new_v[0], new_v[1], new_v[2], 
                                                                new_C[0, 0], new_C[0, 1], new_C[0, 2],
                                                                new_C[1, 0], new_C[1, 1], new_C[1, 2],
                                                                new_C[2, 0], new_C[2, 1], new_C[2, 2],
                                                                new_F[0, 0], new_F[0, 1], new_F[0, 2],
                                                                new_F[1, 0], new_F[1, 1], new_F[1, 2],
                                                                new_F[2, 0], new_F[2, 1], new_F[2, 2]])
            else:
                self.x[t, p] = new_x
                self.v[t, p] = new_v
                self.C[t, p] = new_C
                if ti.static(self.use_fluid):
                    self.J[t, p] = (1. + self.dt * new_C.trace()) * self.J[s, p]
                else:
                    new_F = (ti.Matrix.identity(float, self.dim) + self.dt * new_C) @ self.F[s, p]
                    # J = new_F.determinant()
                    # new_F = ti.Matrix([[J, 0.0], [0.0, 1.0]])
                    # sqrtJ = ti.sqrt(J)
                    # new_F = ti.Matrix.diag(self.dim, sqrtJ)
                    # self.F[t, p] = (ti.Matrix.identity(float, self.dim) + self.dt * new_C) @ self.F[s, p]
                    self.F[t, p] = new_F

    @ti.func
    def particle_sdf(self, new_v, x):
        center = (x * self.inv_dx + 0.5).cast(int)
        idx = self.rot_scene(center)
        dist = self.sdf(idx)
        ret_x = x
        if dist < 0:
            normal = self.sdf_grad(idx)
            theta = self.collider_theta[None]
            vx = ti.cos(theta) * normal[0] + ti.sin(theta) * normal[2]
            vz = -ti.sin(theta) * normal[0] + ti.cos(theta) * normal[2]
            normal[0] = vx
            normal[2] = vz
            new_v = new_v - 2 * normal.dot(new_v) * normal
            # ret_x = x + normal * (ti.abs(dist) + 1e-4 + self.dx)
            x_ = x + normal * (self.dx * 2)
            for k in ti.static(range(self.dim)):
                x_[k] = ti.max(ti.min(x_[k], self.dx * (self.n_grid-self.bound[k])), self.dx * self.bound[k])
            ret_x = x_
        return dist, new_v, ret_x

    @ti.kernel
    def clear_grids(self, grid_v: ti.template(), grid_m: ti.template(), pid: ti.template()):
        for I in ti.grouped(grid_m):
            grid_v[I] = [0.0, 0.0, 0.0]
            grid_m[I] = 0
            # idx = ti.rescale_index(grid_v, pid.parent(2), I)
            # ti.deactivate(pid.parent(), idx)

    def stencil_range(self):
        return ti.ndrange(*((3,) * self.dim))

    def substep_difftaichi(self, s, t):
        self.grid.deactivate_all()
        self.p2g(s)
        self.grid_op()
        if self.use_sdf_collider:
            self.sdf_collision_no_arg()
        self.g2p(s, t)

    @ti.kernel
    def p2g(self, s: ti.i32):
        # grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v1)
        ti.block_dim(256)
        if ti.static(self.use_bls):
            for i in ti.static(range(self.dim)):
                ti.block_local(self.grid_v.get_scalar_field(i))
        for p in range(self.current_idx[None]):
            base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[s, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            cauchy = self.compute_cauchy(s, p)
            # cauchy = ti.Matrix.diag(self.dim, self.la * (self.J[s, p] - 1.0))
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + self.p_mass * self.C[s, p]

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[s, p] +
                                                        affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def g2p(self, s: ti.i32, t: ti.i32):
        # grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v1)
        # grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v2)
        grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v1 if not self.use_sdf_collider else self.grid_v2)
        ti.block_dim(256)
        for p in range(self.current_idx[None]):
            base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[s, p] * self.inv_dx - ti.cast(base, float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, self.dim)
            new_C = ti.Matrix.zero(float, self.dim, self.dim)

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = grid_v[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            new_x = self.x[s, p] + self.dt * new_v
            # for k in ti.static(range(self.dim)):
            #     new_x[k] = ti.max(ti.min(new_x[k], self.ub[k] * self.dx), self.lb[k] * self.dx)

            self.x[t, p] = new_x  # advection
            self.v[t, p] = new_v
            self.C[t, p] = new_C
            if ti.static(self.use_fluid):
                self.J[t, p] = (1. + self.dt * new_C.trace()) * self.J[s, p]
            else:
                self.F[t, p] = (ti.Matrix.identity(float, self.dim) + self.dt * new_C) @ self.F[s, p]

    def run_forward_only(self, save_pics=False, save_states=False):
        frame = 0
        if save_pics:
            output_path = self.param.output_folder + "/exp_" + str(self.param.id)
            os.makedirs(output_path, exist_ok=True)
            self.init_ggui_window()
        if save_states:
            pos_output_folder = self.param.pos_output_folder
            os.makedirs(pos_output_folder, exist_ok=True)

        max_frame = int(self.n_timestep / self.steps + 1.)
        print(f'max_frame: {max_frame}')
        while frame < max_frame:
            if save_pics:
                if self.dim == 3: 
                    ti.sync()
                    st = time.time()
                    self.draw_ggui(0, frame, output_path)
                    ti.sync()
                    end = time.time()
                    print(f'render time: {(end - st) * 1000} ms', flush=True)
                    if frame == 0:
                        # draw a second time to make the lighting right
                        # TODO: fix here!
                        self.draw_ggui(0, frame, output_path)
                else:
                    x_ = self.x.to_numpy(dtype=np.float32)
                    x_ = x_[:, :, :2]
                    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41, show_gui=False)
                    gui.circles(x_[0], radius=2.5, color=0xED553B)
                    gui.show(f'{output_path}/{frame:06d}.png')


            st = time.time()
            for k in range(self.steps):
                self.substep(0, 0)
                self.step_collider()
                print('.', end='', flush=True)
            ti.sync()
            end = time.time()
            print('')
            print(f'simulation time: {(end - st) * 1000} ms', flush=True)
            print(f'theta: {self.collider_theta[None]}')
            print(f'height: {self.collider_height[None]}')
            # exit(-1)
            ti.print_kernel_profile_info()
            ti.print_memory_profile_info()

            if save_states: 
                # if frame == 60:
                self.dump_states(frame)
                # self.dump_grid(frame, self.grid_m[self.curr_grid_idx])
            print(f'frame: {frame}', flush=True)

            frame += 1
        self.compute_loss()
        return self.loss[None]

    def run_forward_with_ad_nlogn(self, save_pics=False, ggui=None):
        self.checkpoint(self.logn_timestep-1, 0)
        if save_pics:
            output_path = self.param.output_folder + "/exp_" + str(self.param.id)
            os.makedirs(output_path, exist_ok=True)
        with ti.Tape(loss=self.loss):
            for s in range(self.n_timestep - 1):
                self.nlogn_loop_forward(s, save_pics)
                if s % self.steps == 0:
                    print('.', end='', flush=True)
            self.compute_loss()
        self.dump_grad(0)
        print(f'loss= {self.loss[None]}', flush=True)
        return self.loss[None]

    def run_forward_detect_ranges(self):
        for s in range(self.n_timestep - 1):
            self.nlogn_loop_forward(s, intro_err=False, save_pics=False)
            # self.detect_bound(1-s%2)
            self.detect_bound_kernel(1-s%2)
            if s % self.steps == 0:
                print('.', end='', flush=True)
                self.detect_bound()
                print(self.get_limits())
        self.detect_bound()

    def detect_bound_kernel(self, s: ti.i32):
        self.update_limits_vector(self.v, self.v_max, self.v_min, s)
        self.update_limits_mat(self.C, self.C_max, self.C_min, s)
        if self.use_fluid:
            self.update_limits_scalar(self.J, self.J_max, self.J_min, s)
        else:
            self.update_limits_mat(self.F, self.F_max, self.F_min, s)

    def detect_bound(self, s=0):
        self.v_limits = self.update_limits(self.v_max, self.v_min)
        self.C_limits = self.update_limits(self.C_max, self.C_min)
        if self.use_fluid:
            self.J_limits = self.update_limits(self.J_max, self.J_min)
        else:
            self.F_limits = self.update_limits(self.F_max, self.F_min)

    @ti.kernel
    def update_limits_scalar(self, fields: ti.template(), fields_max: ti.template(),
                                fields_min: ti.template(), s: ti.i32):
        for i in range(self.current_idx[None]):
            ti.atomic_max(fields_max[None], fields[s, i])
            ti.atomic_min(fields_min[None], fields[s, i])

    @ti.kernel
    def update_limits_vector(self, fields: ti.template(), fields_max: ti.template(), 
                                fields_min: ti.template(), s: ti.int32):
        for i in range(self.current_idx[None]):
            ti.atomic_max(fields_max[None], fields[s, i])
            ti.atomic_min(fields_min[None], fields[s, i])

    @ti.kernel
    def update_limits_mat(self, fields: ti.template(), fields_max: ti.template(), 
                            fields_min: ti.template(), s:ti.i32):
        for i in range(self.current_idx[None]):
            ti.atomic_max(fields_max[None], fields[s, i])
            ti.atomic_min(fields_min[None], fields[s, i])

    def update_limits(self, fields_max, fields_min, center=0.0):
        fields_max_np = fields_max.to_numpy().flatten() - center
        fields_min_np = fields_min.to_numpy().flatten() - center
        _max = np.max(np.vstack([np.abs(fields_max_np), np.abs(fields_min_np)]), axis=0)
        return _max

    def get_limits(self):
        l = []
        l.append([2,]*self.dim)
        l.append(self.v_limits.tolist())
        l.append(self.C_limits.flatten().tolist())
        if self.use_fluid:
            l.append(self.J_limits.tolist())
        else:
            l.append(self.F_limits.tolist())
        l = np.hstack(l)
        return np.array(l)

    def compute_loss(self):
        if not self.use_height_loss:
            self.compute_ek()
        else:
            self.compute_x_avg()
            self.compute_height_loss()
        print('Total energy: ', self.loss[None], flush=True)

    
    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0.0    

    @ti.func
    def compute_cauchy(self, s, p):
        cauchy = ti.Matrix.zero(float, self.dim, self.dim)
        if ti.static(self.use_fluid):
            J = self.J[s, p]
            cauchy = ti.Matrix.diag(self.dim, self.la * (J - 1.0))
        else:
            F = self.F[s, p]
            J = F.determinant()

            if ti.static(self.dim == 2):
                R, S = ti.polar_decompose(F)
                # cauchy = 2 * self.mu_0 * (F - U @ V.transpose()) @ F.transpose() + \
                cauchy = 2 * self.mu_0 * (F - R) @ F.transpose() + \
                             ti.Matrix.identity(float, self.dim) * self.la_0 * J * (J - 1)
            else:
                cauchy = self.mu_0 * (F @ F.transpose() - ti.Matrix.identity(float, self.dim)) +\
                                 self.la_0 * ti.log(J)

        return cauchy
    
    @ti.kernel
    def compute_vel_avg(self):
        N = ti.static(self.last_step if ti.static(self.use_difftaichi) else self.n_timestep-1)
        for p in range(self.current_idx[None]):
            self.vel_avg[None] += self.v[N, p]

    @ti.kernel
    def compute_vel_avg_loss(self):
        self.loss[None] += (self.vel_avg[None] ).norm_sqr() 

    @ti.kernel
    def compute_ek(self):
        N = ti.static(self.last_step if ti.static(self.use_difftaichi) else 0)
        for p in range(self.current_idx[None]):
            # if self.particle_states[self.last_step, p] == ParticleStates.Activated:
            if ti.static(self.dim == 3):
                self.loss[None] += 0.5 * self.p_mass * self.v[N, p].norm_sqr()
                self.loss[None] += self.g * self.p_mass * self.x[N, p][1]
            else:
                self.loss[None] += self.v[N, p].norm_sqr()

    @ti.kernel
    def compute_x_avg(self):
        N = ti.static(self.last_step if ti.static(self.use_difftaichi) else self.n_timestep-1)
        for p in range(self.current_idx[None]):
            self.x_avg[None] += self.x[N, p]
    
    @ti.kernel
    def compute_x_avg_loss(self):
        # N = ti.static(self.last_step if ti.static(self.use_difftaichi) else self.n_timestep-1)
        # for p in range(self.current_idx[None]):
            # self.x_avg[None] += self.x[N, p]
        self.loss[None] += (self.x_avg[None] / self.n_particles).norm_sqr()
    
    @ti.kernel
    def compute_height_loss(self):
        self.loss[None] += (self.x_avg[None][1] / self.n_particles)
    
    @ti.kernel
    def compute_vol(self):
        N = ti.static(self.last_step if ti.static(self.use_difftaichi) else self.n_timestep-1)
        for p in range(self.current_idx[None]):
            self.vol_avg[None] += (self.J[N, p] - 1) ** 2
    
    @ti.kernel
    def compute_vol_loss(self):
        # self.loss[None] += ((self.vol_avg[None] - ti.cast(self.n_particles, float)) / ti.cast(self.n_particles, float) )**2
        self.loss[None] += self.vol_avg[None] / self.n_particles

    @ti.kernel
    def compute_ep(self):
        N = ti.static(self.last_step if ti.static(self.use_difftaichi) else self.n_timestep-1)
        for p in range(self.current_idx[None]):
            # e = self.p_vol * self.la * (self.J[self.last_step, p] - 1) ** 2
            F = self.F[N, p]
            J = F.determinant()
            I1 = (F @ F.transpose()).trace()
            e = 0.5 * self.mu_0 * (I1 - 2) - self.mu_0 * ti.log(J) + 0.5 * self.la_0 * ti.log(J)**2
            self.loss[None] += e * self.p_vol
    
    @ti.kernel
    def checkpoint(self, c: ti.i32, odd: ti.i32):
        # print('checkpoint to {}'.format(c))
        for p in range(self.n_particles):
            self.x_stash[c, p] = self.x[odd, p]
            self.v_stash[c, p] = self.v[odd, p]
            self.C_stash[c, p] = self.C[odd, p]
            if ti.static(self.use_fluid):
                self.J_stash[c, p] = self.J[odd, p]
            else:
                self.F_stash[c, p] = self.F[odd, p]
        self.current_idx_stash[c] = self.current_idx[None]
        self.last_emit_idx_stash[c] = self.last_emit_idx[None]
    
    @ti.kernel
    def load_checkpoint(self, w: ti.i32):
        for p in range(self.n_particles):
            self.x[0, p] = self.x_stash[w, p]
            self.v[0, p] = self.v_stash[w, p]
            # self.J[0, p] = self.J_stash[w, p]
            self.C[0, p] = self.C_stash[w, p]
            if ti.static(self.use_fluid):
                self.J[0, p] = self.J_stash[w, p]
            else:
                self.F[0, p] = self.F_stash[w, p]
        self.current_idx[None] = self.current_idx_stash[w]
        self.last_emit_idx[None] = self.last_emit_idx_stash[w]

    def rerun_from(self, N, w, gui=None):
        # print('s: ', N, 'w: ', w)
        # print(f'load checkpoint: {min([w+1, self.logn_timestep-1])}')
        self.load_checkpoint(min([w+1, self.logn_timestep-1]))
        self.checkpoint(min(w, self.logn_timestep-1), 0)

        c = w
        st = N-N % (1 << (w+1))
        # if st < N:
            # print(f'rerun from: {st} to {N}')
        for s in range(st, N):
            self.nlogn_loop_forward(s)
            if c and (s+1) % (1 << (c-1)) == 0:
                c -= 1
                self.checkpoint(c, 1-s % 2)
                # print(f'checkpointing at frame {s}, stash to {c}')
        assert c == 0
    
    def rerun_from_with_emit(self, N, w, forward_func, emit_func, gui):
        self.load_checkpoint(min([w+1, self.logn_timestep-1]))
        self.checkpoint(min(w, self.logn_timestep-1), 0)

        c = w
        st = N-N % (1 << (w+1))
        if st < N:
            print(f'rerun from: {st} to {N}')
        for s in range(st, N):
            forward_func(self, s, emit_func, gui)
            if c and (s+1) % (1 << (c-1)) == 0:
                c -= 1
                self.checkpoint(c, 1-s % 2)
        assert c == 0

    @ti.ad.grad_replaced
    def nlogn_loop_forward(self, s, intro_err=False, save_pics=False):
        k = s % 2
        self.substep_difftaichi(k, 1-k)
        if self.use_sdf_collider:
            self.step_collider_difftaichi(s)

    def nlogn_step_forward(self, s, emit_func=None):
        k = s % 2
        if emit_func is not None:
            emit_func(s, self)
        self.substep_difftaichi(k, 1-k)
        if self.use_sdf_collider:
            self.step_collider_difftaichi(s)

    @ti.kernel
    def set_current_idx(self, frame: ti.i32, num_emit: ti.i32):
        self.current_idx[None] = frame * num_emit

    @ti.ad.grad_for(nlogn_loop_forward)
    def nlogn_loop_forward_grad(self, s, intro_err, save_pics=False):
        st = time.time()
        if s % 100 == 0:
            print(f'back prop frame: {s}', flush=True)

        w = self.cnt_ending_zero(s)
        self.rerun_from(s,w)

        c = s%2
        if s != self.n_timestep - 2:
            self.set_grad(1-c)

        self.nlogn_loop_forward(s)

        self.g2p.grad(c, 1-c)
        if self.use_sdf_collider:
            self.sdf_collision_no_arg.grad()
        self.grid_op.grad()
        self.p2g.grad(c)

        if s != self.n_timestep - 2:
            self.accu_grad(1-c)
        self.save_grad(c)
        if s != 0:
            self.clear_grad(c)

    # @ti.kernel
    def cnt_ending_zero(self, s: ti.i32) -> ti.i32:
        cnt = 0
        s = s+1
        while s % 2 == 0:
            cnt += 1
            s = s >> 1
        return cnt

    @ti.kernel
    def set_grad(self, odd: ti.i32):
        for p in range(self.n_particles):
            self.x.grad[odd, p] = self.grad_x[p]
            self.v.grad[odd, p] = self.grad_v[p]
            self.C.grad[odd, p] = self.grad_C[p]
            if ti.static(self.use_fluid):
                self.J.grad[odd, p] = self.grad_J[p]
            else:
                self.F.grad[odd, p] = self.grad_F[p]


    @ti.kernel
    def save_grad(self, odd: ti.i32):
        for p in range(self.n_particles):
            self.grad_x[p] = self.x.grad[odd, p]
            self.grad_v[p] = self.v.grad[odd, p]
            self.grad_C[p] = self.C.grad[odd, p]
            if ti.static(self.use_fluid):
                self.grad_J[p] = self.J.grad[odd, p]
            else:
                self.grad_F[p] = self.F.grad[odd, p]

    @ti.kernel
    def accu_grad(self, odd: ti.i32):
        delta_g = 0.1
        for p in range(self.n_particles):
            for k in ti.static(range(self.dim)):
                g_curr = self.x.grad[odd, p][k]
                if ti.static(self.use_grad_clip):
                    g_next = self.x.grad[1-odd, p][k]
                    if g_curr - g_next > delta_g:
                        g = g_next + delta_g
                        self.x.grad[odd, p][k] = g
                        # self.grad_x[p][k] = g
                        # self.grad_sum_x[p][k] += ti.cast(g, ti.f64) ** 2
                    elif g_next - g_curr > delta_g:
                        g = g_next - delta_g
                        self.x.grad[odd, p][k] = g
                        # self.grad_x[p][k] = g
                        # self.grad_sum_x[p][k] += ti.cast(g, ti.f64) ** 2
                    else:
                        self.grad_sum_x[p][k] += ti.cast(g_curr, ti.f64) ** 2
                else:
                    self.grad_sum_x[p][k] += ti.cast(g_curr, ti.f64) ** 2

        for p in range(self.n_particles):
            for k in ti.static(range(self.dim)):
                self.grad_sum_v[p][k] += ti.cast(self.v.grad[odd, p][k], ti.f64) ** 2

        if ti.static(self.use_fluid):
            for p in range(self.n_particles):
                self.grad_sum_J[p] += ti.cast(self.J.grad[odd, p], ti.f64) ** 2
        else:
            for p in range(self.current_idx[None]):
                for k in ti.static(range(self.dim)):
                    for l in ti.static(range(self.dim)):
                        self.grad_sum_F[p][k, l] += ti.cast(self.F.grad[odd, p][k, l], ti.f64) ** 2

        for p in range(self.n_particles):
            for k in ti.static(range(self.dim)):
                for l in ti.static(range(self.dim)):
                    self.grad_sum_C[p][k, l] += ti.cast(self.C.grad[odd, p][k, l], ti.f64) ** 2

    def get_gradients(self):
        grads = []
        if 'x' in self.param.quantized_properties:
            # x_grad = np.square(self.x.grad.to_numpy()).sum(axis=0)
            # x_grad = np.abs(self.x.grad.to_numpy())[0]
            x_grad = self.grad_sum_x.to_numpy().sum(axis=0, keepdims=True)
            # print('x_grad: ', x_grad)
            grads.append(x_grad)
        if 'v' in self.param.quantized_properties:
            # v_grad = np.square(self.v.grad.to_numpy()).sum(axis=0)
            # v_grad = np.abs(self.v.grad.to_numpy())[0]
            v_grad = self.grad_sum_v.to_numpy().sum(axis=0, keepdims=True)
            # print('shape v: ', v_grad.shape)
            grads.append(v_grad)
        if 'C' in self.param.quantized_properties:
            C_grad = self.grad_sum_C.to_numpy().sum(axis=0, keepdims=True)
            C_grad = C_grad.reshape((1, self.dim**2))
            grads.append(C_grad)
        if 'J' in self.param.quantized_properties:
            J_grad = self.grad_sum_J.to_numpy().sum(axis=0, keepdims=True)
            J_grad =  np.reshape(J_grad, (1, 1))
            grads.append(J_grad)
        if 'F' in self.param.quantized_properties:
            F_grad = self.grad_sum_F.to_numpy().sum(axis=0, keepdims=True)
            F_grad = F_grad.reshape((1, self.dim**2))
            # F_grad = np.abs(self.F.grad.to_numpy())[0].reshape((self.n_particles, 4))
            grads.append(F_grad)
        grads = np.hstack(grads)
        # print(grads)
        return grads

    @ti.kernel
    def clear_grad(self, s:ti.i32):
        for i in range(self.n_particles):
            self.x.grad[s, i] = ti.Vector.zero(float, self.dim)
            self.v.grad[s, i] = ti.Vector.zero(float, self.dim)
            if ti.static(self.use_fluid):
                self.J.grad[s, i] = 0.0
            else:
                self.F.grad[s, i] = ti.Matrix.zero(float, self.dim, self.dim)
            self.C.grad[s, i] = ti.Matrix.zero(float, self.dim, self.dim)
            self.x.grad[1-s, i] = ti.Vector.zero(float, self.dim)
            self.v.grad[1-s, i] = ti.Vector.zero(float, self.dim)
            if ti.static(self.use_fluid):
                self.J.grad[1-s, i] = 0.0
            else:
                self.F.grad[1-s, i] = ti.Matrix.zero(float, self.dim, self.dim)
            self.C.grad[1-s, i] = ti.Matrix.zero(float, self.dim, self.dim)
        for I in ti.grouped(self.grid_m):
            self.grid_m.grad[I] = 0.0
            self.grid_v.grad[I].fill(0.0)
            self.grid_v1.grad[I].fill(0.0)
            if ti.static(self.use_sdf_collider):
                self.grid_v2.grad[I].fill(0.0)

    def dump_grad(self, s, fn=None):
        grads = self.get_gradients()
        grads = np.hstack(grads)
        print(grads)


    def add_cube(self, low_corner, box_size, num):
        if (len(low_corner) == 3):
            lc = ti.Vector([low_corner[0], low_corner[1], low_corner[2]])
            bs = ti.Vector([box_size[0], box_size[1], box_size[2]])
            self.fill_in_box(lc, bs, num)
        else:
            lc = ti.Vector([low_corner[0], low_corner[1]])
            bs = ti.Vector([box_size[0], box_size[1]])
            self.fill_in_box_2duniform(lc, bs, num)
    
    @ti.kernel
    def pos_from_numpy(self, pos: ti.ext_arr(), num: ti.i32):
        for i in range(num):
            for k in ti.static(range(self.dim)):
                self.x[0, i][k] = pos[i, k]
        self.current_idx[None] += num
    
    def add_ball(self, center, radius, num):
        c = ti.Vector([*center])
        r = radius
        self.fill_in_ball(c, r, num)

    def add_mesh(self,
                 triangles,
                 material=None,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None,
                 translation=None,
                 s=0):
        assert self.dim == 3
        if sample_density is None:
            sample_density = 2 ** self.dim

        if translation is None:
            translation = ti.Vector([0.0, 0.0, 0.0])

        if velocity is None:
            velocity = ti.Vector([0, 0, 0])

        self.voxelizer.voxelize(triangles)
        # t = time.time()
        self.add_particles_from_voxels(sample_density, translation, velocity, s)

    def add_particles_from_voxels(self, sample_density, translation, velocity, s):
        self.seed_from_voxels(sample_density, self.voxelizer_super_sample, translation, velocity, s)
        ti.sync()
    
    def cnt_mesh_particles_num(self,
                                triangles,
                                material=None,
                                color=0xFFFFFF,
                                sample_density=None,
                                velocity=None,
                                translation=None):
        assert self.dim == 3
        if sample_density is None:
            sample_density = 2 ** self.dim

        if translation is None:
            translation = ti.Vector([0.0, 0.0, 0.0])

        if velocity is None:
            velocity = ti.Vector([0, 0, 0])

        self.voxelizer.voxelize(triangles)
        cnt = self.cnt_emit_num(sample_density, self.voxelizer_super_sample, translation, velocity)
        return cnt

    @ti.kernel
    def cnt_emit_num(self, sample_density: ti.i32,
                         voxelizer_super_sample:ti.i32, 
                         translation: ti.template(), 
                         vel: ti.template()) -> ti.i32:
        cnt = 0
        for i, j, k in ti.grouped(self.voxelizer.voxels):
            if self.voxelizer.voxels[i, j, k] > 0:
                cnt += sample_density + 1  
        return cnt
    
    @ti.kernel
    def seed_from_voxels(self, sample_density: ti.i32,
                         voxelizer_super_sample:ti.i32, 
                         translation: ti.template(), 
                         vel: ti.template(), s: ti.i32):
        for i, j, k in ti.grouped(self.voxelizer.voxels):
            # inside = 1
            # for d in ti.static(range(3)):
                # inside = inside and -self.grid_size // 2 + self.padding <= i and i < self.grid_size // 2 - self.padding
            if self.voxelizer.voxels[i, j, k] > 0:
                # cnt += 1
                # print('i, j, k: ', i, j ,k)
                # s = sample_density / voxelizer_super_sample ** self.dim
                for l in range(sample_density + 1):
                    # if ti.random() + l < s:
                    p = ti.atomic_add(self.current_idx[None], 1)
                    # x = ti.Vector([
                    #     self.voxelizer.voxels_random[i, j, k, l * self.dim] + i,
                    #     self.voxelizer.voxels_random[i, j, k, l * self.dim + 1] + j,
                    #     self.voxelizer.voxels_random[i, j, k, l * self.dim + 2] + k,
                    # ]) * (self.dx)
                    x = ti.Vector([
                        ti.random() + i,
                        ti.random() + j,
                        ti.random() + k
                    ]) * (self.dx)# / voxelizer_super_sample)
                    self.x[s, p] = x + translation
                    self.v[s, p] = vel

    @ti.func
    def sdf_grad(self, I):
        return self.collider_sdf_normal[I]

    @ti.func
    def sdf(self, I):
        ret = 1.0
        if self.collider_sdf[I] < -1e-5:
            ret = self.collider_sdf[I]
        return ret

    @ti.func
    def get_center_idx(self):
        # return ti.Vector([self.grid_size // 2, self.n_grid//2, self.grid_size//2])
        return ti.Vector([self.n_grid * 2 // 2, self.n_grid//2, self.n_grid * 2//2])

    @ti.kernel
    def sdf_collision(self, grid_v: ti.template()):
        # for I in ti.grouped(self.collider_sdf):
        for I in ti.grouped(grid_v):
            # d = self.sdf(I)
            idx = self.rot_scene(I)
            d = self.sdf(idx)
            # c = ti.Vector([self.n_grid//2, I[1], self.n_grid//2])
            c = self.get_center_idx()
            c[1] = I[1]
            if d < 0:
                g   = self.sdf_grad(idx)
                theta = self.collider_theta[None]
                # theta = 0.0#self.collider_theta[None]
                vx = ti.cos(theta) * g[0] + ti.sin(theta) * g[2]
                vz = -ti.sin(theta) * g[0] + ti.cos(theta) * g[2]
                normal = ti.Vector([vx, g[1], vz])

                omega = ti.Vector([0.0, self.collider_omega[None], 0.0])
                collider_v = omega.cross((idx - c).cast(float) * self.dx)
                # collider_v = omega.cross((I - c).cast(float) * self.dx)
                v_rel = grid_v[I] - collider_v
                grid_v[I] = collider_v + (v_rel - ti.min(v_rel.dot(normal), 0.0) * normal)
                
                # idx = I#self.rot_scene(I)
                # idx_v = ti.rescale_index(grid_v, self.block, idx)
                # if ti.is_active(self.block, idx_v):
                # grid_v[I] = grid_v[I] - 2 * normal.dot(grid_v[I]) * normal
                # else:
                    # print("visit inactivated grid_v")

    @ti.kernel
    def sdf_collision_no_arg(self):
        grid_v = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v1)
        grid_v_out = ti.static(self.grid_v if not self.use_difftaichi else self.grid_v2)
        for I in ti.grouped(grid_v):
            # d = self.sdf(I)
            idx = self.rot_scene(I)
            d = self.sdf(idx)
            # c = ti.Vector([self.n_grid//2, I[1], self.n_grid//2])
            c = self.get_center_idx()
            c[1] = I[1]
            if d < 0:
                g   = self.sdf_grad(idx)
                theta = self.collider_theta[None]
                # theta = 0.0#self.collider_theta[None]
                vx = ti.cos(theta) * g[0] + ti.sin(theta) * g[2]
                vz = -ti.sin(theta) * g[0] + ti.cos(theta) * g[2]
                normal = ti.Vector([vx, g[1], vz])

                omega = ti.Vector([0.0, self.collider_omega[None], 0.0])
                collider_v = omega.cross((I - c).cast(float) * self.dx) # from the branch of fluid forward 
                # collider_v = omega.cross((idx - c).cast(float) * self.dx)
                v_rel = grid_v[I] - collider_v
                grid_v_out[I] = collider_v + (v_rel - ti.min(v_rel.dot(normal), 0.0) * normal)
            else:
                grid_v_out[I] = grid_v[I]


    @ti.func
    def rot_scene(self, I):
        # C = ti.Vector([self.n_grid//2, self.n_grid//2, self.n_grid//2])
        # C = ti.Vector([self.grid_size // 2, self.n_grid//2, self.grid_size//2])
        C = self.get_center_idx()
        # _offset = I - C
        idx = I
        if I[0] >= self.n_grid * 2:
            idx[0] = I[0] - self.n_grid * 2
        if I[2] >= self.n_grid * 2:
            idx[2] = I[2] - self.n_grid * 2

        _offset = idx - C
        theta = -self.collider_theta[None]
        # theta = self.collider_theta[None]
        x = ti.cos(theta) * _offset[0] + ti.sin(theta) * _offset[2] + C[0] + 0.5
        z = -ti.sin(theta) * _offset[0] + ti.cos(theta) * _offset[2] + C[2] + 0.5
        y = I[1] - int(self.collider_height[None] / self.dx + 0.5)
        # y = I[1] + int(self.collider_height[None] / self.dx + 0.5)
        x_int = ti.cast(x, ti.int32)
        z_int = ti.cast(z, ti.int32)
        y_int = ti.cast(y, ti.int32)
        # y_int = ti.max(y_int, 0)
        # y_int = ti.min(y_int, self.n_grid-3)
        x_int = ti.max(ti.min(x_int, self.ub[0]), self.bound[0])
        y_int = ti.max(ti.min(y_int, self.ub[1]), self.bound[1])
        z_int = ti.max(ti.min(z_int, self.ub[2]), self.bound[2])
        return ti.Vector([x_int, y_int, z_int])

    @ti.kernel
    def step_collider(self):
        self.collider_omega[None] = 5.0 #/ (self.dt / 1e-4)
        self.collider_theta[None] += self.collider_omega[None] * self.dt
        self.collider_height[None] += 0.1 * self.dt
        # self.collider_height[None] = ti.min(self.collider_height[None], -0.016)

    @ti.kernel
    def step_collider_difftaichi(self, step: ti.i32):
        self.collider_omega[None] = 5.0
        self.collider_theta[None] = self.collider_omega[None] * self.dt * step
        self.collider_height[None] = 0.15 * self.dt * step # for range and ad
        # self.collider_height[None] = 0.1 * self.dt * step

    @ti.kernel
    def fill_in_box_2duniform(self, low_corner: ti.template(), box_size: ti.template(), num: ti.i32):
        n = ti.cast(ti.sqrt(num), ti.i32)
        st = self.current_idx[None]
        for i in range(n):
            for j in range(n):
                self.x[0, st + i * n + j][0] = low_corner[0] + 1.0/n * j * box_size[0]
                self.x[0, st + i * n + j][1] = low_corner[1] + 1.0/n * i * box_size[1]
                self.v[0, st + i * n + j].fill(0)
                                        
        self.current_idx[None] += n * n

    @ti.kernel
    def fill_in_box(self, low_corner: ti.template(), box_size: ti.template(), num: ti.i32):
        st = self.current_idx[None]
        for i in range(num):
            for k in ti.static(range(self.dim)):
                self.x[0, i + st][k] = low_corner[k] + ti.random() * box_size[k]
                self.v[0, i + st][k] = 0.0
            
            if ti.static(self.use_sdf_collider):
                base = (self.x[0, i + st] / self.dx + 0.5).cast(int)
                idx = self.rot_scene(base)
                # idx = base
                if self.sdf(idx) < 0:
                    g = self.sdf_grad(idx)
                    self.x[0, i + st] += g * (ti.random() * 3 + 2) * self.dx
                    # self.x[0, i + st][1] += 0.5#g * (ti.random() * 3 + 2) * self.dx
        self.current_idx[None] += num

    @ti.kernel
    def fill_in_ball(self, center: ti.template(), radius: ti.f32, num: ti.i32):
        st = self.current_idx[None]
        if st + num < self.n_particles:
            for i in range(num):
                self.x[i + st] = center + self.random_point_in_unit_sphere() * radius
            self.current_idx[None] += num

    @ti.kernel
    def fill_in_ball_with_vel(self, center: ti.template(), radius: ti.f32, vel: ti.template(), num: ti.i32):
        st = self.current_idx[None]
        if st + num < self.n_particles:
            for i in range(num):
                self.x[0, i + st] = center + self.random_point_in_unit_sphere_in_xy() * radius
                self.v[0, i + st] = vel
            self.current_idx[None] += num

    @ti.func
    def random_point_in_unit_sphere_in_xy(self):
        ret = ti.Vector.zero(float, self.dim)
        theta = ti.random() * np.pi * 2.0
        r = ti.random()
        _x = ti.cos(theta) * r
        _y = ti.sin(theta) * r
        ret[0] = _x 
        ret[1] = _y 
        return ret

    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(ti.f32, n=self.dim)
        while True:
            for i in ti.static(range(self.dim)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    def emit_water(self, center, radius, velocity, num):
        cent = ti.Vector([center[0], center[1], center[2]])
        vel = ti.Vector([velocity[0], velocity[1], velocity[2]])
        self.fill_in_ball_with_vel(cent, radius, vel, num)

    @ti.func
    def seed_particle(self, x, lc, bs):
        for k in ti.static(range(self.dim)):
            x[k] = lc[k] + ti.random() * bs[k]

    def intro_quantization_errors(self, f):
        if 'x' in self.param.quantized_properties:
            self.intro_error_vec_kernel(self.x, self.x, f, 0)
            # self.add_error_vec_kernel(self.x, self.x, f, 0)
        if 'v' in self.param.quantized_properties:
            self.intro_error_vec_kernel(self.v, self.v, f, 1)
            # self.add_error_vec_kernel(self.v, self.v, f, 1)
        if 'C' in self.param.quantized_properties:
            self.intro_error_mat(self.C, self.C, f, 0)
            # self.add_error_mat_kernel(self.C, self.C, f, 0)
        if 'F' in self.param.quantized_properties:
            self.intro_error_mat(self.F, self.F, f, 1)
            # self.add_error_mat_kernel(self.F, self.F, f, 1)

    @ti.kernel
    def copy_particles_pos_for_ggui(self, pos_idx: ti.i32):
        N = ti.static(self.reduced_rate if self.reduced_ggui_particles else 1)
        for i in range(self.n_ggui_particles):
            self.x_for_ggui[i] = ti.cast(self.x[pos_idx, N * i], ti.f32)

    def init_ggui(self):
        self.ground = ti.Vector.field(3, dtype=ti.f32, shape=(4))
        self.ground_indices = ti.field(dtype=int, shape=(6))
        self.reduced_ggui_particles = self.n_particles > 100**3
        self.reduced_rate = 100 if self.reduced_ggui_particles else 1
        self.n_ggui_particles = self.n_particles // self.reduced_rate
        self.x_for_ggui = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ggui_particles)
        if self.draw_propeller:
            self.init_propeller_mesh()
        self.init_ground_mesh()

    @ti.kernel
    def init_ground_mesh(self):
        scale = 10.0
        self.ground[0] = ti.Vector([-1.0, -0.0, -1.0]) * scale
        self.ground[1] = ti.Vector([1.0, -0.0, -1.0]) * scale
        self.ground[2] = ti.Vector([1.0, -0.0, 1.0]) * scale
        self.ground[3] = ti.Vector([-1.0, -0.0, 1.0]) * scale

        self.ground_indices[0] = 0
        self.ground_indices[1] = 1
        self.ground_indices[2] = 2
        self.ground_indices[3] = 0
        self.ground_indices[4] = 2
        self.ground_indices[5] = 3

    
    def init_propeller_mesh(self):
        vert, indi = load_mesh_vertices_and_indices('propeller_2x2.ply')
        N_vert = vert.shape[0]
        N_indi = indi.shape[0]
        self.collider_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_vert)
        self.collider_vertices_draw = ti.Vector.field(3, dtype=ti.f32, shape=N_vert * 4)
        self.collider_indices = ti.field(dtype=int, shape=N_indi)
        self.collider_indices_draw = ti.field(dtype=int, shape=N_indi * 4)

        @ti.kernel
        def set_data(vert: ti.ext_arr(), indi: ti.ext_arr()):
            for i in range(N_vert):
                for j in ti.static(range(3)):
                    self.collider_vertices[i][j] = vert[i, j] 
            for i in range(N_indi):
                self.collider_indices[i] = int(indi[i])
            for i in ti.static(range(4)):
                for j in range(N_indi):
                    self.collider_indices_draw[i * N_indi + j] = self.collider_indices[j] + N_vert * i 

        set_data(vert, indi)

    def init_ggui_window(self):
        if not self.ggui_initialized:
            camera = ti.ui.make_camera()
            # cam_pos = (2.5, 0.7, 2.5)
            cam_pos = (4.5, 0.5, 4.5)
            camera.position(*(cam_pos))
            # camera.lookat(1.0, 0.1, 1.0)
            camera.lookat(2.0, 0.1, 2.0)
            camera.up(0.0, 1.0, 0.0)
            camera.fov(55)
            self.window = ti.ui.Window('3D MPM Fluid Simulator', (1280, 720), show_window = False)
            self.scene = ti.ui.Scene()
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((0.9, 0.9, 0.9))
            self.scene.set_camera(camera)
            self.scene.ambient_light((0.1, 0.1, 0.1))
    
    def draw_ggui(self, pos_idx, frame, output_path):
        self.scene.point_light(pos=(2.0, 5.5, 2.0), color=(2, 2, 2))
        self.scene.mesh(self.ground, 
                    indices=self.ground_indices,
                    color=(0.5, 0.5, 0.5),
                    two_sided=True)
        self.copy_particles_pos_for_ggui(pos_idx)
        self.scene.particles(self.x_for_ggui, 0.01, color=(.5, .5, 1.0))
        if self.draw_propeller:
            for i in range(2):
                for j in range(2):
                    propeller_pos = ti.Vector([ (i * 2.0) + 1.0, 0, (j * 2.0) + 1.0])
                    self.move_propeller_mesh(propeller_pos, self.collider_vertices.shape[0] * (2 * i + j))
            self.scene.mesh(self.collider_vertices_draw, self.collider_indices_draw, color=(0.1, 0.1, 0.1))
        self.canvas.scene(self.scene)
        png_name = f'{output_path}/{frame:06d}.png'
        self.window.write_image(png_name)

    @ti.kernel
    def move_propeller_mesh(self, pos: ti.template(), st: ti.i32):
        height = self.collider_height[None]
        theta = self.collider_theta[None]
        C = ti.cos(theta)
        S = ti.sin(theta)
        rot_mat = ti.Matrix([[C, 0, S], [0, 1, 0], [-S, 0, C]])
        for v in self.collider_vertices:
            self.collider_vertices_draw[st + v] = rot_mat @ (self.collider_vertices[v] - ti.Vector([1.0, self.collider_vertices[v][1], 1.0])) \
                                                        + ti.Vector([0.0, self.collider_vertices[v][1], 0.0]) + pos
            self.collider_vertices_draw[st + v][1] = height + self.collider_vertices[v][1]

    def dump_grid(self, frame, grid_m):
        pos_output_folder = self.param.pos_output_folder
        os.makedirs(pos_output_folder, exist_ok=True)
        m = grid_m.to_numpy(np.float16)
        np.savez_compressed(f'{pos_output_folder}/density_{frame:06d}.npz', m=m)
    
    def checkpoint_states(self, frame):
        # if frame % 10 == 0 and frame >= 200:
        pos_output_folder = self.param.pos_output_folder
        # v_ = self.v.to_numpy(dtype=np.float32)[0]
        v_ =self.fetch_data_np(self.v)
        np.save(f'{pos_output_folder}/v.npy', v_)
        # C_ = self.C.to_numpy(dtype=np.float32)[0]
        C_ = self.fetch_mat_data_np(self.C)
        np.save(f'{pos_output_folder}/C.npy', C_)
        # F_ = self.F.to_numpy(dtype=np.float32)[0]
        # F_ = self.fetch_mat_data_np(self.F)
        # np.save(f'{pos_output_folder}/F.npy', F_)
        J_ = self.fetch_scalar_data_np(self.J)
        np.save(f'{pos_output_folder}/J.npy', J_)

    def dump_states(self, frame):
        pos_output_folder = self.param.pos_output_folder
        os.makedirs(pos_output_folder, exist_ok=True)
        data = self.fetch_data_np(self.x)
        write_point_cloud_packed(f'{pos_output_folder}/data_{frame:06d}.ply', data, self.n_particles)

    def fetch_data_np(self, fields, parts=10):
        N = self.n_particles // parts
        data = np.zeros([self.n_particles, self.dim], np.float32)
        for i in range(parts):
            x = np.zeros([N, self.dim])
            self.fetch_data(N, i * N, x, fields)
            data[i*N:(i+1)*N, :] = x
        return data
    
    def fetch_scalar_data_np(self, fields, parts=10):
        N = self.n_particles // parts
        data = np.zeros([self.n_particles], np.float32)
        for i in range(parts):
            x = np.zeros([N])
            self.fetch_scalar_data(N, i * N, x, fields)
            data[i*N:(i+1)*N] = x
        return data

    def fetch_mat_data_np(self, fields):
        N = self.n_particles // 10
        data = np.zeros([self.n_particles, self.dim, self.dim], np.float32)
        for i in range(10):
            x = np.zeros([N, self.dim, self.dim])
            self.fetch_mat_data(N, i * N, x, fields)
            data[i*N:(i+1)*N, :, :] = x
        return data

    @ti.kernel
    def fetch_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            for k in ti.static(range(self.dim)):
                # data_np[i, k] = self.x[0, st + i][k]
                data_np[i, k] = fields[0, st + i][k]
    
    @ti.kernel
    def fetch_scalar_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            data_np[i] = fields[0, st + i]

    @ti.kernel
    def fetch_mat_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            for j in ti.static(range(self.dim)):
                for k in ti.static(range(self.dim)):
                    data_np[i, j, k] = fields[0, st + i][j, k]

    def push_data_np(self, fields, data):
        N = self.n_particles // 10
        # data = np.zeros([self.n_particles, self.dim], np.float32)
        for i in range(10):
            d = data[i*N:(i+1)*N, :]
            self.push_data(N, i * N, d, fields)
        return data

    def push_mat_data_np(self, fields, data):
        N = self.n_particles // 10
        for i in range(10):
            d = data[i*N:(i+1)*N, :, :]
            self.push_mat_data(N, i * N, d, fields)
        return data

    def push_scalar_data_np(self, fields, data):
        N = self.n_particles // 10
        for i in range(10):
            d = data[i*N:(i+1)*N]
            self.push_scalar_data(N, i * N, d, fields)
        return data

    @ti.kernel
    def push_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            for k in ti.static(range(self.dim)):
                # data_np[i, k] = self.x[0, st + i][k]
                fields[0, st + i][k] = data_np[i, k]

    @ti.kernel
    def push_scalar_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            fields[0, st + i] = data_np[i]

    @ti.kernel
    def push_mat_data(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr(), fields: ti.template()):
        for i in range(n):
            for j in ti.static(range(self.dim)):
                for k in ti.static(range(self.dim)):
                    fields[0, st + i][j, k] = data_np[i, j, k]

    def load_states(self, folder, frame):
        pos_fn = f'{folder}/data_{frame:06d}.ply'
        print('load x')
        x_ = load_particles_only(pos_fn)
        print('push x')
        self.push_data_np(self.x, x_)
        print('load v')
        v_ = np.load(f'{folder}/v.npy')
        print('push v')
        self.push_data_np(self.v, v_)
        print('load C')
        C_ = np.load(f'{folder}/C.npy')
        print('push C')
        self.push_mat_data_np(self.C, C_)
        if not self.use_fluid:
            print('load F')
            F_ = np.load(f'{folder}/F.npy')
            print('push F')
            self.push_mat_data_np(self.F, F_)
            self.current_idx[None] = 295280208
        else:
            print('load J')
            J_ = np.load(f'{folder}/J.npy')
            print('push J')
            self.push_scalar_data_np(self.J, J_)
            self.current_idx[None] = self.n_particles
        print('finished')

    @ti.kernel
    def disturbx(self, h: ti.f32):
        self.x[0, 0][1] += h
    
    @ti.kernel
    def set_particle_pos(self, n: ti.i32, st: ti.i32, data_np: ti.ext_arr()):
        for i in range(n):
            for k in ti.static(range(self.dim)):
                self.x[0, st + i][k] = data_np[i, k] 
    
    def init_particles_from_file(self, fn):
        data = load_piont_cloud_packed(fn)
        N = self.n_particles // 10
        for i in range(10):
            self.set_particle_pos(N, i * N, data[i * N: (i+1)*N])
        self.current_idx[None] = self.n_particles
