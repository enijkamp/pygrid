import os

import shutil
import argparse
import itertools
import random

import torch
import torch.utils.data

import pygrid


def train(opt_override, output_dir, logger, return_dict):

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', default='exp_id')
    parser.add_argument('--seed', type=int, default=1, help='manual seed, default=%(default)s')
    parser.add_argument('--device', type=int, default=0, metavar='S', help='device id, default=%(default)s')

    parser.add_argument('--param1', type=float, default=1.0, help='parameter 1, default=%(default)s')
    parser.add_argument('--param2', type=bool, default=True, help='parameter 2, default=%(default)s')

    # preamble
    opt = parser.parse_args()
    opt = pygrid.overwrite_opt(opt, opt_override)
    if torch.cuda.is_available():
        logger.info('setting cuda device to id {} (total of {} cuda devices)'.format(torch.cuda.current_device(), torch.cuda.device_count()))
    pygrid.set_cudnn()
    pygrid.set_gpu(opt.device)
    device = get_device(opt.device)
    opt.seed = set_seed(opt.seed)
    logger.info(opt)

    # run
    # TODO add your training loop here
    with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
        print('param1={}, param2={}'.format(opt.param1, opt.param2), file=f)

    x = torch.tensor(opt.param1, requires_grad=True).to(device)
    metric = torch.autograd.grad(torch.sin(x), x)[0].item()

    # return
    return_dict['stats'] = {'metric': metric}
    logger.info('done')


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device(device):
    return 'cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu'


def create_opts():
    # TODO add your enumeration of parameters here
    param1 = [float(p1) for p1 in range(10)]
    param2 = [True, False]

    args_list = [param1, param2]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'param1': args[0],
            'param2': args[1]
        }
        # TODO add your result metric here
        opt_result = {'metric': 0.0}
        opt_list += [{**opt_job, **opt_args, **opt_result}]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['metric'] = job_stats['metric']


def main(device_ids):
    # set devices
    pygrid.fill_queue(device_ids)

    # set opts
    get_opts_filename = lambda exp: '{}.csv'.format(exp)
    write_opts = lambda opts: pygrid.write_opts(opts, get_opts_filename(exp_id))
    read_opts = lambda: pygrid.read_opts(get_opts_filename(exp_id))

    exp_id = pygrid.get_exp_id(__file__)
    output_dir = pygrid.get_output_dir(exp_id)
    if not os.path.exists(get_opts_filename(exp_id)):
        write_opts(create_opts())
    pygrid.write_opts(pygrid.reset_job_status(read_opts()), get_opts_filename(exp_id))

    # set logging
    logger = pygrid.setup_logging('main', output_dir, console=True)
    logger.info('available devices {}'.format(device_ids))

    # run
    pygrid.run_jobs(logger, exp_id, output_dir, len(device_ids), train, read_opts, write_opts, update_job_result)
    shutil.copyfile(get_opts_filename(exp_id), os.path.join(output_dir, get_opts_filename(exp_id)))
    logger.info('done')


if __name__ == '__main__':
    # TODO enumerate gpu devices here
    gpu_ids = [gpu for gpu in range(1)]
    main(gpu_ids)
