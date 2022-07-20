import taichi as ti


@ti.test(require=ti.extension.quant_basic, debug=True)
def test_fixed_scalar():
    cft = ti.quant.fixed(frac=5, range=2, compute=ti.f64, use_dithering=True)
    x = ti.field(dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x)

    up = ti.field(dtype=int, shape=())
    down = ti.field(dtype=int, shape=())

    @ti.kernel
    def foo():
        x[None] = 0.7
        x[None] = x[None] + 0.4

    @ti.kernel
    def run():
        if x[None] == 1.125:
            up[None] += 1
        elif x[None] == 1.0:
            down[None] += 1

    for _ in range(10000):
        foo()
        run()

    ratio = up[None] / down[None]
    assert abs(ratio - 4) < 0.1
