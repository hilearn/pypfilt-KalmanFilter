import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import pypfilt
import scipy
import os
import argparse
import sys
import pandas as pd
import convert_csv_netcdf as cnv
from netCDF4 import Dataset


def generate_pypfilt_obs_data(data: list,
                              source='Any',
                              unit='states',
                              period=1):
    obs = []
    for i, ret in enumerate(data):
        obs.append(
            {'date': float(i),
             'value': ret,
             'unit': unit,
             'period': period,
             'source': source})
    return obs


def extract_timeseries_from_csv(csv_path, variable):
    data_frame = pd.read_csv(csv_path)
    var = data_frame[variable]
    return np.array(var)


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help="data file path")
    parser.add_argument('-v', '--variable',
                        help='variable on which particle filter will be done',
                        required=True)
    args = parser.parse_args(*argument_array)
    return args


def kalmanfilter_randomwalk(obs, sigma_w=10., sigma_r=10.):
    kalman = KalmanFilter(obs.shape[1], 1)
    kalman.x = np.zeros((kalman.dim_x, 1))
    kalman.P = np.eye(kalman.dim_x) * 1 ** 2
    kalman.Q = (np.eye(kalman.dim_x) * sigma_w ** 2)
    kalman.R = (np.eye(kalman.dim_z) * sigma_r ** 2)
    kalman.F = np.eye(kalman.dim_x)

    kalman.predict()

    states = [kalman.x[:, 0]]

    for r in obs:
        # prediction
        kalman.update(r, H=np.eye(kalman.dim_x))
        kalman.predict()
        states.append(kalman.x[:, 0])

    states = np.array(states)

    return states


class RandomWalk(pypfilt.Model):
    def init(self, params, vec):
        rnd = params['resample']['rnd']
        size = vec[..., 0].shape
        vec[..., 0] = params['prior']['states'](rnd, size)

    def state_size(self):
        return 1

    def priors(self, params):
        return {
            'states': lambda r, size=None: np.zeros(size)
        }

    def update(self, params, t, dt, is_fs, prev, curr):
        curr[:] = [np.random.normal(pre, params['obs']['sigma_w'])
                   for pre in prev]

    def describe(self):
        return [
            ('states', False, -1e5, 1e5),
        ]


def log_llhd(params, obs_list, curr, prev_dict, weights):
    log_llhd = np.zeros(curr.shape[:-1])
    for o in obs_list:
        if o['unit'] == 'states':
            states_dist = scipy.stats.norm(loc=curr[..., 0],
                                           scale=params['obs']['sigma_r'])
            log_llhd += states_dist.pdf(o['value'])
        else:
            raise ValueError('invalid observation')
    return log_llhd


def make_randomwalk_params(px_count, ret=0.25, sigma_r=1., sigma_w=10.):
    model = RandomWalk()
    time_scale = pypfilt.Scalar()
    params = pypfilt.default_params(model, time_scale, px_count=px_count)
    params['steps_per_unit'] = 1
    params['log_llhd_fn'] = log_llhd
    params['obs'] = {'sigma_r': sigma_r, 'ret': ret, 'sigma_w': sigma_w}
    params['resample']['prng_seed'] = 0
    params['resample']['threshold'] = 0.5
    params['out_dir'] = '.'
    params['tmp_dir'] = '.'
    return params


def bootstrap_forecast_randomwalk(obs,
                                  sigma_w=10.,
                                  sigma_r=1.,
                                  t0=0.0,
                                  t1=100.0):
    fs_times = list(range(1, len(obs)))
    fs_times = [float(time) for time in fs_times]
    params = make_randomwalk_params(px_count=4000,
                                    sigma_r=sigma_r,
                                    sigma_w=sigma_w)

    summary = pypfilt.summary.HDF5(params, obs, first_day=True)
    summary.add_tables(
        pypfilt.summary.ModelCIs(probs=[0], name=u'model_cints'),
        pypfilt.summary.Obs())
    results = pypfilt.run(params,
                          t0,
                          t1,
                          [obs],
                          summary,
                          save_when=fs_times,
                          save_to='output')

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parse_args()

    # getting data path and variable for particle filtering
    data_path = os.path.abspath(args.data)
    variable = args.variable

    # extracting data
    data = extract_timeseries_from_csv(data_path, variable)

    # generating data
    cnv.convert_csv_to_netcdf(csv_path=data_path,
                              netcdf_path='input.nc',
                              var=variable)
    libbi_data_path = os.path.abspath('input.nc')
    pypfilt_data = generate_pypfilt_obs_data(list(data))
    kalman_data = data.reshape(data.shape[0], 1)

    # implementing particle filtering
    state_cints = bootstrap_forecast_randomwalk(pypfilt_data,
                                                sigma_w=0.2,
                                                sigma_r=0.1,
                                                t1=float(data.shape[0]))
    states = kalmanfilter_randomwalk(kalman_data, sigma_w=0.2, sigma_r=0.1)

    # running libbi
    x = os.system('cd ./libbi && ./filter.sh {0} {1} && cd ../'.format(libbi_data_path, data.shape[0])) # noqa
    if x:
        sys.exit(x)

    libbi_filtered_grp = Dataset('filtered.nc', 'r')
    libbi_logreturns = libbi_filtered_grp.variables['v'][:]

    real_states = []
    for state in state_cints['summary']['model_cints']:
        real_states.append(state[3])

    plt.plot(real_states, color='blue', label='pypfilt result')
    plt.plot(states, color='red', label='kalman result')
    plt.plot(libbi_logreturns.mean(axis=1),
             color='black',
             label='libbi results')
    plt.plot(data, color='green', label='real')
    plt.legend()
    plt.savefig('result.png')
    plt.show()
