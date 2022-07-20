from plyfile import PlyData
import numpy as np

def load_particles_only(fn):
    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        assert len(x) == len(y) == len(z)
        pos = np.vstack([x[:], y[:], z[:]]).T
        return pos

def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    return triangles


def load_mesh_vertices_and_indices(fn):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    vertices = np.vstack([x, z, y]).T
    vertices[:, -1] *= -1
    indices = np.vstack(elements['vertex_indices']).flatten()
    return vertices, indices


def write_point_cloud(fn, pos_and_color):
    num_particles = len(pos_and_color)
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
property float vel_x 
property float vel_y 
property float vel_z
end_header
"""
        f.write(str.encode(header))
        f.write(pos_and_color.tobytes())


def write_point_cloud_packed(fn, pos, num_particles):
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(pos.tobytes())


def load_particles_only(fn):
    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        assert len(x) == len(y) == len(z)
        pos = np.vstack([x[:], y[:], z[:]]).T
        return pos

# def load_particles(fn, frame, emit_step=4, num_emit=4762584):
def load_particles(fn, frame, all_particles=False, emit_step=5, num_emit=8466816):

    colors = np.array([[0x00, 0x66, 0x33], 
                        [0xCC, 0xCC, 0x33], 
                        [0xCC, 0x99, 0x33]], dtype=np.uint8)
    # colors = np.array([[0xCC, 0xCC, 0x33],
    #                     [0x00, 0x66, 0x33], 
    #                     [0xCC, 0x99, 0x33]], dtype=np.uint8)

    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        assert len(x) == len(y) == len(z)

        # N = (frame // emit_step + 1) * num_emit if not all_particles else len(x)
        # if N > len(x):
            # N = (len(x)//num_emit) * num_emit
        # N = min(N, len(x))
        N = frame * 256 // emit_step * num_emit if not all_particles else len(x)
        print(f'N: {N}')

        pos = np.vstack([x[:N], y[:N], z[:N]]).T
        color = np.ones_like(pos, dtype=np.uint8)

        if not all_particles:
            # emit_times = frame // emit_step + 1
            # for k in range(emit_times):
            #     color[k*num_emit:(k+1)*num_emit] = colors[k%3]
            color[:] = colors[0]
        else:
            color[:] = colors[0]

        # pos = np.vstack([x, -z, y]).T
    return pos, color

def load_piont_cloud_packed(fn):
    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        return np.vstack([x, y, z]).T
