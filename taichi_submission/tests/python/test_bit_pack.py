import numpy as np
from pytest import approx

import taichi as ti


@ti.test(require=ti.extension.quant_basic, debug=True)
def test_simple_array():
    ci13 = ti.quant.int(13, True)
    cu19 = ti.quant.int(19, False)

    x = ti.field(dtype=ci13)
    y = ti.field(dtype=cu19)

    N = 12

    ti.root.dense(ti.i, N).bit_pack().place(x, y)

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = -2**i
            y[i] = 2**i - 1

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == -2**i
            assert y[i] == 2**i - 1

    set_val()
    verify_val()

    # Test bit_pack SNode read and write in Python-scope by calling the wrapped, untranslated function body
    set_val.__wrapped__()
    verify_val.__wrapped__()


# TODO: remove excluding of ti.metal
@ti.test(require=ti.extension.quant_basic, exclude=[ti.metal], debug=True)
def test_custom_int_load_and_store():
    cix = ti.quant.int(31, True)
    cuy = ti.quant.int(29, False)
    ciz = ti.quant.int(8, True)

    x = ti.field(dtype=cix)
    y = ti.field(dtype=cuy)
    z = ti.field(dtype=ciz)

    test_case_np = np.array(
        [[2**30 - 1, 2**29 - 1, -(2**7)], [2**30 - 1, 2**29 - 1, -(2**7 - 1)],
         [0, 0, 0], [123, 4567, 8], [10, 31, 11]],
        dtype=np.int32)

    ti.root.bit_pack().place(x, y, z)
    test_case = ti.Vector.field(3, dtype=ti.i32, shape=len(test_case_np))
    test_case.from_numpy(test_case_np)

    @ti.kernel
    def set_val(idx: ti.i32):
        x[None] = test_case[idx][0]
        y[None] = test_case[idx][1]
        z[None] = test_case[idx][2]

    @ti.kernel
    def verify_val(idx: ti.i32):
        assert x[None] == test_case[idx][0]
        assert y[None] == test_case[idx][1]
        assert z[None] == test_case[idx][2]

    for idx in range(len(test_case_np)):
        set_val(idx)
        verify_val(idx)

    # Test bit_pack SNode read and write in Python-scope by calling the wrapped, untranslated function body
    for idx in range(len(test_case_np)):
        set_val.__wrapped__(idx)
        verify_val.__wrapped__(idx)

@ti.test(require=[ti.extension.quant_basic, ti.extension.sparse], debug=True)
def test_bit_pack_struct_for():
    block_size = 16
    N = 64
    cell = ti.root.pointer(ti.i, N // block_size)
    fixed32 = ti.quant.fixed(frac=32, range=1024)

    x = ti.field(dtype=fixed32)
    cell.dense(ti.i, block_size).bit_pack().place(x)

    for i in range(N):
        if i // block_size % 2 == 0:
            x[i] = 0

    @ti.kernel
    def assign():
        for i in x:
            x[i] = ti.cast(i, float)

    assign()

    for i in range(N):
        if i // block_size % 2 == 0:
            assert x[i] == approx(i, abs=1e-3)
        else:
            assert x[i] == 0
