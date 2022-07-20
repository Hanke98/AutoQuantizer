import math

from pytest import approx
from taichi.lang import expr, impl

import taichi as ti


@ti.test(require=ti.extension.quant_basic)
def test_fxp_offset():
    cft = ti.quant.fixed(frac=29, range=2, offset=1.0)
    x = ti.field(dtype=cft)

    ti.root.bit_struct(num_bits=32).place(x)

    @ti.kernel
    def foo():
        x[None] = 0.7
        # print(x[None])
        x[None] = x[None] + 0.4


    foo()
    assert x[None] == approx(1.1, 1e-4)
    x[None] = 0.64
    assert x[None] == approx(0.64, 1e-4)
    x[None] = 0.66
    assert x[None] == approx(0.66, 1e-4)
    x[None] = 2.8
    assert x[None] == approx(2.8, 1e-4)

# test_fxp_offset()