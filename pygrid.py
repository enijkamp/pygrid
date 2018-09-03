import os

import shutil
import datetime
import logging
import sys
import csv

import queue
import threading
import concurrent.futures
import multiprocessing

import torch
import torch.utils.data


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def set_gpu(device):
    torch.cuda.set_device(device)


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


job_file_lock = threading.Lock()


def update_job_status(job_id, job_status, read_opts, write_opts):
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


def update_job_result_file(update_job_result, job_opt, job_stats, read_opts, write_opts):
    try:
        job_file_lock.acquire()

        opts = read_opts()
        target_opt = next(opt for opt in opts if opt['job_id'] == job_opt['job_id'])
        update_job_result(target_opt, job_stats)

        write_opts(opts)
    finally:
        job_file_lock.release()


run_job_lock = threading.Lock()


def run_job(logger, opt, output_dir, train):

    device_id = allocate_device()
    opt_override = {'device': device_id}
    opt = {**opt, **opt_override}
    logger.info('new job: job_id={}, device_id={}'.format(opt['job_id'], opt['device']))
    try:
        logger.info("spawning process: job_id={}, device_id={}".format(opt['job_id'], opt['device']))

        try:
            output_dir_thread = os.path.join(output_dir, str(opt['job_id']))
            if not os.path.exists(output_dir_thread):
                os.mkdir(output_dir_thread)
            copy_source(__file__, output_dir_thread)
            logger_thread = setup_logging('job{}'.format(opt['job_id']), output_dir_thread, console=False)

            run_job_lock.acquire()
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=train, args=(opt, output_dir_thread, logger_thread, return_dict))
            p.start()
        finally:
            run_job_lock.release()

        p.join()

        logger.info('finished process: job_id={}, device_id={}'.format(opt['job_id'], opt['device']))

        return return_dict['stats']
    finally:
        free_device(device_id)


def run_jobs(logger, exp_id, output_dir, workers, train_job, read_opts, write_opts, update_job_result):
    opt_list = read_opts()
    opt_open = [opt for opt in opt_list if opt['status'] == 'open']
    logger.info('scheduling {} open of {} total jobs'.format(len(opt_open), len(opt_list)))
    logger.info('starting thread pool with {} workers'.format(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        def adjust_opt(opt):
            opt_override = {'exp_id': '{}_{}'.format(exp_id, opt['job_id'])}
            return {**opt, **opt_override}

        def do_run_job(opt):
            update_job_status(opt['job_id'], 'running', read_opts, write_opts)
            return run_job(logger, adjust_opt(opt), output_dir, train_job)

        futures = {executor.submit(do_run_job, opt): opt for opt in opt_open}

        for future in concurrent.futures.as_completed(futures):
            opt = futures[future]
            try:
                stats = future.result()
                logger.info('finished job future: job_id={}'.format(opt['job_id']))
                update_job_result_file(update_job_result, opt, stats, read_opts, write_opts)
                update_job_status(opt['job_id'], 'finished', read_opts, write_opts)
            except Exception:
                logger.exception('exception in run_jobs()')
                update_job_status(opt['job_id'], 'fail', read_opts, write_opts)


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
        return value.upper() == 'TRUE'
    return value


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def overwrite_opt(opt, opt_override):
    for (k, v) in opt_override.items():
        setattr(opt, k, v)
    return opt


def write_opts(opt_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        header = [key for key in opt_list[0]]
        writer.writerow(header)
        for opt in opt_list:
            writer.writerow([opt[k] for k in header])


def read_opts(filename):
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
    return opts_list
