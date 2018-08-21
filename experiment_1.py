import os
import random

import shutil
import datetime
import logging
import sys
import argparse
import csv
import itertools

import queue
import threading
import concurrent.futures
import multiprocessing

import torch
import torch.utils.data


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
    opt = overwrite_opt(opt, opt_override)
    set_cudnn()
    set_gpu(opt.device)
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


def get_device(device):
    return 'cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def set_gpu(device):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


def overwrite_opt(opt, opt_override):
    for (k, v) in opt_override.items():
        setattr(opt, k, v)
    return opt


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


free_devices_lock = threading.Lock()
free_devices = queue.Queue()


def fill_queue(device_ids):
    [free_devices.put_nowait(device_id) for device_id in device_ids]


def allocate_device():
    try:
        free_devices_lock.acquire()
        return free_devices.get()
    finally:
        free_devices_lock.release()


def free_device(device):
    try:
        free_devices_lock.acquire()
        return free_devices.put_nowait(device)
    finally:
        free_devices_lock.release()


def run_job(logger, opt, output_dir):
    update_job_status(opt['job_id'], 'running')
    device_id = allocate_device()
    opt_override = {'device': device_id}
    opt = {**opt, **opt_override}
    logger.info('new job: job_id={}, device_id={}'.format(opt['job_id'], opt['device']))
    try:
        output_dir_thread = os.path.join(output_dir, str(opt['job_id']))
        if not os.path.exists(output_dir_thread):
            os.mkdir(output_dir_thread)
        copy_source(__file__, output_dir_thread)
        logger_thread = setup_logging('job{}'.format(opt['job_id']), output_dir_thread, console=False)

        logger.info("spawning process: job_id={}, device_id={}".format(opt['job_id'], opt['device']))

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=train, args=(opt, output_dir_thread, logger_thread, return_dict))
        p.start()
        p.join()

        logger.info('finished process: job_id={}, device_id={}'.format(opt['job_id'], opt['device']))

        return return_dict['stats']
    finally:
        free_device(device_id)


job_file_lock = threading.Lock()


def update_job_status(job_id, job_status):
    try:
        job_file_lock.acquire()

        opts = read_opts()
        opt = next(opt for opt in opts if opt['job_id'] == job_id)
        opt['status'] = job_status
        write_opts(opts)
    except Exception:
        logging.exception('exception in update_job_status()')
    finally:
        job_file_lock.release()


def run_jobs(logger, exp_id, opt_list, output_dir, workers):
    opt_open = [opt for opt in opt_list if opt['status'] == 'open']
    logger.info('scheduling {} open of {} total jobs'.format(len(opt_open), len(opt_list)))
    logger.info('starting thread pool with {} workers'.format(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        def adjust_opt(opt):
            opt_override = {'exp_id': '{}_{}'.format(exp_id, opt['job_id'])}
            return {**opt, **opt_override}

        futures = {executor.submit(run_job, logger, adjust_opt(opt), output_dir): opt for opt in opt_open}

        for future in concurrent.futures.as_completed(futures):
            opt = futures[future]
            try:
                stats = future.result()
                logger.info('finished job future: job_id={}'.format(opt['job_id']))
                update_job_result(opt, stats)
                update_job_status(opt['job_id'], 'finished')
            except Exception:
                logger.exception('exception in run_jobs()')
                update_job_status(opt['job_id'], 'fail')


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_float(value):
    try:
        float(value)
        return not is_int(value)
    except ValueError:
        return False


def is_bool(value):
    return value.upper() in ['TRUE', 'FALSE']


def cast_str(value):
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    if is_bool(value):
        return bool(value)
    return value


def get_exp_id():
    return os.path.splitext(os.path.basename(__file__))[0]


def get_opts_filename():
    return '{}.csv'.format(get_exp_id())


def write_opts(opt_list, filename=get_opts_filename()):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = [key for key in opt_list[0]]
        writer.writerow(header)
        for opt in opt_list:
            writer.writerow([opt[k] for k in header])


def read_opts(filename=get_opts_filename()):
    opt_list = []
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for values in reader:
            opt = {}
            for i, field in enumerate(header):
                opt[field] = cast_str(values[i])
            opt_list += [opt]
    return opt_list


def reset_job_status(opts_list):
    for opt in opts_list:
        if opt['status'] == 'running':
            opt['status'] = 'open'


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
    write_opts(opt_list)


def update_job_result(job_opt, job_stats):
    try:
        job_file_lock.acquire()

        opts = read_opts()
        opt = next(opt for opt in opts if opt['job_id'] == job_opt['job_id'])

        # TODO add your result metric here
        opt['metric'] = job_stats['metric']

        write_opts(opts)
    finally:
        job_file_lock.release()


def main(device_ids):
    fill_queue(device_ids)
    output_dir = get_output_dir(get_exp_id())
    if not os.path.exists(get_opts_filename()):
        create_opts()
    logger = setup_logging('main', output_dir, console=True)
    logger.info('available devices {}'.format(device_ids))
    opt_list = read_opts(get_opts_filename())
    reset_job_status(opt_list)
    write_opts(opt_list)
    run_jobs(logger, get_exp_id(), opt_list, output_dir, workers=len(device_ids))
    shutil.copyfile(get_opts_filename(), os.path.join(output_dir, get_opts_filename()))
    logger.info('done')


if __name__ == '__main__':
    # TODO enumerate gpu or cpu devices here
    #cpu_ids = [cpu for cpu in range(32)]
    gpu_ids = [gpu for gpu in range(1)]
    main(gpu_ids)
