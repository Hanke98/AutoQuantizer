import taichi as ti

import numpy as np
import pytest
from pytest import approx

import taichi as ti


@ti.test(require=ti.extension.quant_basic, debug=True, default_fp=ti.f32)
def test_assign():
    x_type = ti.quant.fixed(frac=21, range=2)
    y_type = ti.quant.int(15)
    x = ti.Vector.field(3, dtype=x_type)
    y = ti.field(dtype=y_type)

    ti.root.dense(ti.i, 2).bit_pack().place(x, y)

    @ti.kernel
    def assign():
        ti.bit_pack_assign(x.parent(), 0, [1.5, 1.2, 1.3, 15])
        ti.bit_pack_assign(x.parent(), 1, [-1.1, 1.3, -1.2, 16])

    def verify():
        assert x[0][0] == approx(1.5, 1e-4)
        assert x[0][1] == approx(1.2, 1e-4)
        assert x[0][2] == approx(1.3, 1e-4)
 
        assert x[1][0] == approx(-1.1, 1e-4)
        assert x[1][1] == approx(1.3, 1e-4)
        assert x[1][2] == approx(-1.2, 1e-4)

        assert y[0] == 15
        assert y[1] == 16
    
    assign()
    verify()


@ti.test(require=ti.extension.quant_basic, debug=True, default_fp=ti.f32)
def test_assign_split_exactly_on_64():
    x_type = ti.quant.fixed(frac=16, range=2)
    y_type = ti.quant.int(15)
    x = ti.Vector.field(4, dtype=x_type)
    y = ti.field(dtype=y_type)

    ti.root.dense(ti.i, 2).bit_pack().place(x, y)

    @ti.kernel
    def assign():
        ti.bit_pack_assign(x.parent(), 0, [1.5, 1.2, 1.3, 1.6, 15])
        ti.bit_pack_assign(x.parent(), 1, [-1.1, 1.3, -1.2, -1.6, 16])

    def verify():
        assert x[0][0] == approx(1.5, 1e-4)
        assert x[0][1] == approx(1.2, 1e-4)
        assert x[0][2] == approx(1.3, 1e-4)
        assert x[0][3] == approx(1.6, 1e-4)
 
        assert x[1][0] == approx(-1.1, 1e-4)
        assert x[1][1] == approx(1.3, 1e-4)
        assert x[1][2] == approx(-1.2, 1e-4)
        assert x[1][3] == approx(-1.6, 1e-4)

        assert y[0] == 15
        assert y[1] == 16
    
    assign()
    verify()
