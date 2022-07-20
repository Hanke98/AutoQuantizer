
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T',
                        '--target-exp',
                        type=int,
                        default=0,
                        help='target-exp')
    return parser.parse_args()

def run_error_bound():
    script = 'run_mpm_elastic_2d.py'
    log_file = 'log_elastic_2d_eb'

    for i in range(1, 5):
        eps = 10**(-i)
        cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -f -e {eps} -m {1} > {log_file}{eps}'
        print(cmd)
        os.system(cmd)

def run_mem_bound():
    script = 'run_mpm_fluid_2d.py'
    log_file = 'log_fluid_2d_mc'

    mr = [0.4, 0.5, 0.6]
    for i in range(0, 3):
        eps = mr[i]
        cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -f -e {eps} -m {0} > {log_file}{eps}'
        print(cmd)
        os.system(cmd)

def run_dithering_exp():
    script = 'run_mpm_dithering_exp.py'
    log_file = 'log_dithering_exp'

    cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -r  > {log_file}_f64'
    os.system(cmd)
    cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -d  > {log_file}_with_dithering'
    os.system(cmd)
    cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script}  > {log_file}_without_dithering'
    os.system(cmd)

# optimality_check
def run_error_bound_oc():
    log_file = 'log_oc_2d'
    script = 'run_mpm_optimality_check.py'
    for k in range(1, 4):
        for t in range(0, 2):
            eps = 10**-2
            b = k
            reduce_str = 'reduce_' + 'all' if t == 0 else 'half'
            cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -e {eps} -m {1} -t {t} -b {b} > {log_file}_{eps}_{reduce_str}_-{b}_bits'
            print(cmd)
            os.system(cmd)

# optimality_check_catesian
def run_error_bound_oc_cartesian():
    log_file = 'log_rerun4_oc_catesian_move'
    script = 'run_mpm_optimality_check.py'
    for S1 in range(-3, 4):
        for S2 in range(-3, 4):
            eps = 10**-2
            cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} -e {eps} -t 2 -m {1} -b {S1} -b2 {S2} > {log_file}_{eps}_{S1}_{S2}_bits'
            print(cmd)
            os.system(cmd)


args = parse_args()
if args.target_exp == 0:
    run_error_bound()
elif args.target_exp == 1:
    run_mem_bound()
elif args.target_exp == 2:
    run_dithering_exp()
elif args.target_exp == 3:
    run_error_bound_oc()
    run_error_bound_oc_cartesian()