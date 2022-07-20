import taichi as ti
# ti.init(arch=ti.cpu, print_ir=False, advanced_optimization = False, debug = True)
ti.init(arch=ti.cuda, print_ir=False, advanced_optimization = False, debug = True)
# ti.init(advanced_optimization = False, quant_opt_atomic_demotion = False)
# cxt=ti.quant.fixed(signed = False, frac = 20, range=10.0)
# cyt=ti.quant.fixed(signed = False, frac = 20, range=10.0)
# czt=ti.quant.fixed(signed = False, frac = 24, range=10.0)
cxt=ti.quant.int(signed = False, bits = 20)
cyt=ti.quant.int(signed = False, bits = 20)
czt=ti.quant.int(signed = False, bits = 26)

cwt=ti.quant.int(signed = False, bits = 25)
cut=ti.quant.int(signed = False, bits = 23)
# cxt=float
# cyt=float
# czt=float

x=ti.field(dtype=cxt)
y=ti.field(dtype=cyt)
z=ti.field(dtype=czt)
x1=ti.field(dtype=cxt)
y1=ti.field(dtype=cyt)
z1=ti.field(dtype=czt)
x2=ti.field(dtype=cxt)
y2=ti.field(dtype=cyt)
z2=ti.field(dtype=czt)
# u=ti.field(dtype=cut)
# w1=ti.field(dtype=cwt)
# w=ti.field(dtype=cwt)



t1 = ti.root.dense(ti.i,(100)).bit_pack()
# t1 = ti.root.dense(ti.i,(100)).bit_struct(64)
# t1.place(x,y,z)
t1.place(x,y,z,x1,y1,z1,x2,y2,z2)
# t1.place(x1,y1,z1)
# t1.place(x,y,w,w1,u)
# t1.place(x,y)
# t1.place(x,w)
# ti.root.dense(ti.i,(100)).bit_struct(64).place(x,y,z)
@ti.kernel
def run_loop():
    for i in range(10):
        # x [i] = 1 * i
        # y [i] = 2 * i
        # z [i] = 0x2000001 + i
        x2[i] = 1 * i 
        y2[i] = 2 * i
        z2[i] = 3 * i
        # w [i] = 3
        # w1[i] = 4
        # u [i] = 20
        # x [i]= 1 
        # y [i]= 2 
        # z [i]= 3 
        # x1[i]= 4 - 5
        # y1[i]= 5 - 5
        # z1[i]= 6 - 5
    # print(x[0])
    # print(u[0])
def verify():
    for i in range(10):
        # print(x[i])
        # print(u[i])
        # print(x[i],y[i],w[i])
        # print(x[i],y[i])
        # print(x[i],y[i],w[i])
        # print(x[i],w[i])
        # print(w1[i],u[i])
        # print(x[i],y[i],z[i])
        # print(x[i],y[i],z[i],w[i])
        # print(x1[i],y1[i],z1[i])
        print(x2[i],y2[i],z2[i])

@ti.kernel
def run():
    x [0] = 1 
    y [0] = 6
    # w [0] = 5 
    z [0] = 0x2000001
    # w1[0] = 1 
    # u [0] = 200
    # y[0] = x[0]

    # x[0] = 1.0
    # y[0] = 2.0
    # z[0] = 3.0
    # x1[0]= 4.0
    # y1[0]= 4.0
    # z1[0]= 4.0


    # print(x[0],y[0],z[0])
    # print(x1[0],y1[0],z1[0])
# run()
# print(x[0],y[0],z[0])
# print(x[0],y[0])
# print(w1[0],u[0])

run_loop()
verify()
# print(x[0])