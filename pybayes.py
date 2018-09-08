import sys
import logging

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class Acquisiton(object):
    def __init__(self, kind, kappa=2.576, xi=0.0):
        self.kappa = kappa
        self.xi = xi
        self.kind = kind

    def __call__(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        return norm.cdf(z)


class Points:

    def __init__(self, bounds):
        self.bounds = np.array(list(bounds.values()), dtype=np.float)
        self.keys = list(bounds.keys())
        self.dim = len(self.keys)

        self._Xarr = None
        self._Yarr = None

        self._length = 0

        self._Xview = None
        self._Yview = None

        self._cache = {}

    @property
    def X(self):
        return self._Xview

    @property
    def Y(self):
        return self._Yview

    def __contains__(self, x):
        return self._hashable(x) in self._cache

    def __len__(self):
        return self._length

    def append(self, x, y):
        assert x not in self
        assert all(self.bounds[i][0] <= x[i] <= self.bounds[i][1] for i in range(len(self.bounds)))

        if self._length >= self._n_alloc_rows:
            self._allocate((self._length + 1) * 2)

        x = np.asarray(x).ravel()

        self._cache[self._hashable(x)] = y

        self._Xarr[self._length] = x
        self._Yarr[self._length] = y

        self._length += 1

        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    def _allocate(self, num):
        assert not num <= self._n_alloc_rows

        _Xnew = np.empty((num, self.bounds.shape[0]))
        _Ynew = np.empty(num)

        if self._Xarr is not None:
            _Xnew[:self._length] = self._Xarr[:self._length]
            _Ynew[:self._length] = self._Yarr[:self._length]
        self._Xarr = _Xnew
        self._Yarr = _Ynew

        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    @property
    def _n_alloc_rows(self):
        return 0 if self._Xarr is None else self._Xarr.shape[0]

    def max_point(self):
        return {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, self.X[self.Y.argmax()]))}

    @staticmethod
    def _hashable(x):
        return tuple(map(float, x))


def maximize(f, points, n_iter, acq, gp, callback, random_state=np.random.RandomState()):
    y_max = points.Y.max()

    gp.fit(points.X, points.Y)

    result = {'max': {'max_val': None, 'max_params': None}, 'all': {'values': [], 'params': []}}

    for i in range(n_iter):
        x_max = arg_max_acq(acq=acq, gp=gp, y_max=y_max, bounds=points.bounds, random_state=random_state)

        while x_max in points:
            x_max = random_points(points.bounds, 1, random_state)[0]

        y = f(x_max)
        points.append(x_max, y)
        callback(i, points.keys, x_max, y)
        gp.fit(points.X, points.Y)

        result['max'] = points.max_point()
        result['all']['values'].append(y)
        result['all']['params'].append(dict(zip(points.keys, x_max)))

        if points.Y[-1] > y_max:
            y_max = points.Y[-1]

    return result


def arg_max_acq(acq, gp, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
    # random
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0]))
    ys = acq(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # optimize
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -acq(x.reshape(1, -1), gp=gp, y_max=y_max), x_try.reshape(1, -1), bounds=bounds, method='L-BFGS-B')

        if not res.success:
            continue

        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def wrap_f(f, param_keys):
    def f_closure(x):
        x = np.asarray(x).ravel()
        params = dict(zip(param_keys, x))
        return f(**params)
    return f_closure


def random_points(bounds, num, random_state=np.random.RandomState()):
    dim = len(bounds)
    data = np.empty((num, dim))
    for i, (lower, upper) in enumerate(bounds):
        data.T[i] = random_state.uniform(lower, upper, size=num)
    return data


def sort_params(params):
    return [params[i] for i in sorted(range(len(params)), key=params.__getitem__)]


def get_sizes(params):
    return [max(len(ps), 7) for ps in params]


def print_header(logger, params):
    params = sort_params(params)
    sizes = get_sizes(params)

    log_str = '{:>{}} {:>{}}'.format('Step', 5, 'Value', 10)
    for param, size in zip(params, sizes):
        log_str += ('{0:>{1}}'.format(param, size + 2))
    logger.info(log_str)


def print_step(logger, iter, params, x, y):
    params = sort_params(params)
    sizes = get_sizes(params)

    log_str = '{:>5d}'.format(iter)
    log_str += (' {: >10.5f}'.format(y))
    for x_i, size in zip(x, sizes):
        log_str += (' {0: >{1}.{2}f}'.format(x_i, size + 2, min(size - 3, 6 - 2)))
    logger.info(log_str)


def create_print_step(logger):
    return lambda i, params, x, y: print_step(logger, i, params, x, y)


def create_console_logger():
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    logger = create_console_logger()

    random_state = np.random.RandomState()
    points = Points(bounds={'x': (-10, +10), 'y': (-1, +1)})
    f = lambda x, y: -np.power(x, 2.0) + y
    f_wrap = wrap_f(f, points.keys)
    [points.append(x, f_wrap(x)) for x in random_points(points.bounds, 5)]

    print_header(logger, points.keys)
    acq = Acquisiton(kind='ucb')
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=random_state)
    res = maximize(f_wrap, points=points, n_iter=5, acq=acq, gp=gp, callback=create_print_step(logger))
    logger.info('{}'.format(res))
