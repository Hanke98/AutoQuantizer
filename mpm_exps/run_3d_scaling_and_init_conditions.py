import os



script = 'run_mpm_fluid_3d_scaling.py'

def run_init_condition(save_pics, save_states, quant):
    log_file = 'log_init_condition'

    for i in range(0, 8):
        save_pic_str = ' -sp ' if save_pics else ' '
        save_states_str = ' -ss ' if save_states else ' '
        quant_str = '-q' if quant else ' '
        log_file_str = (log_file + '_quant') if quant else log_file
        config_str = f'-c configs/init_condition/config{i}.json'
        record_error_str = '' if save_pics or save_states else '_error'
        eps = 0.5 if i <= 5 else 0.3
        cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} {quant_str} -e {eps} -m 0 {config_str}'\
              f'{save_states_str} {save_pic_str} > {log_file_str}_config{i}{record_error_str}'
        print(cmd)
        os.system(cmd)


def run_scaling(save_pics, save_states, quant):
    log_file = 'log_scaling'
    quant_str = '-q' if quant else ' '
    save_pic_str = ' -sp ' if save_pics else ' '
    save_states_str = ' -ss ' if save_states else ' '
    log_file_str = (log_file + '_quant') if quant else log_file
    record_error_str = '' if save_pics or save_states else '_error'
    eps = 0.5
    for i in range(3, 4):
        config_str = f'-c configs/scaling/config{i}.json'
        cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {script} {quant_str} -e {eps} -m 0 {config_str}'\
              f'{save_states_str} {save_pic_str} > {log_file_str}_config{i}{record_error_str}'
        print(cmd)
        os.system(cmd)



def run_rendering(quant):
    render_script= 'run_render_particles.py'
    log_file = 'log_render_rerun2_init'
    for i in range(5, 6):
        quant_str = '-q' if quant else ' '
        log_file_str = (log_file + '_quant') if quant else log_file + '_f64'
        cmd = f'CUDA_VISIBLE_DEVICES=\"0\" python -u {render_script} {quant_str} -i {i} > {log_file_str}_config{i}'
        print(cmd)
        os.system(cmd)

run_init_condition(True, False, True)
run_scaling(True, False, True)