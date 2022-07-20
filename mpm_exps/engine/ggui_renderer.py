import taichi as ti
from engine.mesh_io import load_mesh_vertices_and_indices


@ti.data_oriented
class MPMRenderer():
    def __init__(self, n_particles, draw_propeller=False, reduce_num=100):
        self.n_particles = n_particles
        self.draw_propeller = draw_propeller
        self.ggui_initialized = False
        self.reduce_num = reduce_num
        self.init_ggui_fields()

    def init_ggui_fields(self):
        self.ground = ti.Vector.field(3, dtype=ti.f32, shape=(4))
        self.ground_indices = ti.field(dtype=int, shape=(6))

        if self.draw_propeller:
            self.init_propeller_mesh_fields()
        self.reduced_ggui_particles = self.n_particles > 100**3
        self.reduced_rate = self.reduce_num if self.reduced_ggui_particles else 1
        self.n_ggui_particles = self.n_particles // self.reduced_rate
        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.n_ggui_particles)

        self.bounds_particles_lines = ti.Vector.field(3, dtype=ti.f32, shape=10*12)

    def init_ggui_fields_values(self):
        if self.draw_propeller:
            self.init_propeller_mesh_data()
        self.init_ground_mesh_data()
        self.init_bounds_particles_lines()

    @ti.kernel
    def init_ground_mesh_data(self):
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

    def init_propeller_mesh_fields(self):
        vert, indi = load_mesh_vertices_and_indices('inputs/propeller_2x2.ply')
        N_vert = vert.shape[0]
        N_indi = indi.shape[0]
        self.collider_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_vert)
        self.collider_vertices_draw = ti.Vector.field(3, dtype=ti.f32, shape=N_vert * 4)
        self.collider_indices = ti.field(dtype=int, shape=N_indi)
        self.collider_indices_draw = ti.field(dtype=int, shape=N_indi * 4)
        self.collider_height = ti.field(dtype=float, shape=())
        self.collider_theta = ti.field(dtype=float, shape=())

    def init_propeller_mesh_data(self):
        vert, indi = load_mesh_vertices_and_indices('inputs/propeller_2x2.ply')
        N_vert = vert.shape[0]
        N_indi = indi.shape[0]
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

    @ti.kernel
    def init_bounds_particles_lines(self):
        L = 4
        N = 10
        st = 0
        for i in range(N):
            self.bounds_particles_lines[i + st] = [i/N * L, 0, 0]
            self.bounds_particles_lines[i + st + N] = [i/N * L, 0, L]
            self.bounds_particles_lines[i + st + N * 2] = [i/N * L, L, L]
            self.bounds_particles_lines[i + st + N * 3] = [i/N * L, L, 0]
        st += N * 4
        for i in range(N):
            self.bounds_particles_lines[i + st] = [0, i/N * L, 0]
            self.bounds_particles_lines[i + st + N] = [0, i/N * L, L]
            self.bounds_particles_lines[i + st + N * 2] = [L, i/N * L, L]
            self.bounds_particles_lines[i + st + N * 3] = [L, i/N * L, 0]
        st += N * 4
        for i in range(N):
            self.bounds_particles_lines[i + st] = [0, 0, i/N * L]
            self.bounds_particles_lines[i + st + N] = [0, L, i/N * L]
            self.bounds_particles_lines[i + st + N * 2] = [L, L, i/N * L]
            self.bounds_particles_lines[i + st + N * 3] = [L, 0, i/N * L]

    @ti.kernel
    def move_propeller_mesh(self, pos: ti.template(), st: ti.i32):
        height = self.collider_height[None]
        theta = self.collider_theta[None]
        # height = 0.0
        # theta = 0.0
        C = ti.cos(theta)
        S = ti.sin(theta)
        rot_mat = ti.Matrix([[C, 0, S], [0, 1, 0], [-S, 0, C]])
        for v in self.collider_vertices:
            self.collider_vertices_draw[st + v] = rot_mat @ (self.collider_vertices[v] - ti.Vector([1.0, self.collider_vertices[v][1], 1.0])) \
                                                        + ti.Vector([0.0, self.collider_vertices[v][1], 0.0]) + pos
            self.collider_vertices_draw[st + v][1] = height + self.collider_vertices[v][1]

    @ti.kernel
    def copy_particles_pos_for_ggui(self, x: ti.template(), pos_idx: ti.i32):
        N = ti.static(self.reduced_rate if self.reduced_ggui_particles else 1)
        for i in range(self.n_ggui_particles):
            self.particle_pos[i] = ti.cast(x[pos_idx, N * i], ti.f32)

    def init_ggui_window(self, pos=None, center=None):
        if not self.ggui_initialized:
            camera = ti.ui.make_camera()
            # for fluid demo
            cam_pos = (4.5, 0.5, 4.5) if pos is None else pos
            camera.position(*(cam_pos))
            cam_center = (2.0, 0.1, 2.0) if center is None else center
            camera.lookat(*cam_center) 

            # for elastic demo
            # cam_pos = (2, 0.6, 3.5) if pos is None else pos
            # camera.position(*(cam_pos))
            # cam_center = (2, 0.5, 2.0) if center is None else center
            # camera.lookat(*cam_center)

            camera.up(0.0, 1.0, 0.0)
            camera.fov(55)
            self.window = ti.ui.Window('3D MPM Fluid Simulator', (1280, 720), show_window = False)
            self.scene = ti.ui.Scene()
            self.canvas = self.window.get_canvas()
            self.canvas.set_background_color((0.9, 0.9, 0.9))
            self.scene.set_camera(camera)
            self.scene.ambient_light((0.1, 0.1, 0.1))

    def draw_ggui(self, x, pos_idx, frame, output_path):
        self.scene.point_light(pos=(2.0, 5.5, 2.0), color=(2, 2, 2))
        self.scene.mesh(self.ground, 
                    indices=self.ground_indices,
                    color=(0.5, 0.5, 0.5),
                    two_sided=True)
        self.copy_particles_pos_for_ggui(x, pos_idx)
        # self.scene.particles(self.particle_pos, 0.01, color=(.5, .5, 1.0)) # for fluid demo
        if self.draw_propeller:
            for i in range(2):
                for j in range(2):
                    propeller_pos = ti.Vector([ (i * 2.0) + 1.0, 0, (j * 2.0) + 1.0])
                    self.move_propeller_mesh(propeller_pos, self.collider_vertices.shape[0] * (2 * i + j))
            self.scene.mesh(self.collider_vertices_draw, self.collider_indices_draw, color=(0.1, 0.1, 0.1))
        self.scene.particles(self.particle_pos, 0.005, color=(.5, .5, 1.0)) # for elastic demo
        self.scene.particles(self.bounds_particles_lines, 0.01, color=(1.0, .5, 1.0))
        self.canvas.scene(self.scene)
        png_name = f'{output_path}/{frame:06d}.png'
        self.window.write_image(png_name)
