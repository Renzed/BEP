import networkx
import numpy as np
import matplotlib.pyplot as plt
from model import Kuramoto, order_parameter, binder_cumulant, fluctuation, velocity
from scipy.optimize import curve_fit as cf
from tqdm import tqdm
import itertools as it
import parmap as pm


def matrix_coupling(n, mat=''):
    if isinstance(mat, str):
        raise ValueError('No matrix found')
    else:
        return mat


def noise_fun(n, noise_strength=0):
    return np.append(np.array([0]), np.random.normal(0, noise_strength, n - 1))


def multi_target(_star_size, _matrix, _noise_strength, _ssn):
    _initial_positions = np.random.uniform(-np.pi, np.pi, _star_size)
    _eigen_frequencies = np.append(np.array([1]), np.zeros(_star_size - 1))
    sim_params = {'timespan': 4 * 60, 'stepsize': 1 / 15}
    _model = Kuramoto(_eigen_frequencies, _initial_positions,
                      coupling_fun=matrix_coupling,
                      coupling_fun_kwargs=dict(mat=_matrix),
                      noise_fun=noise_fun,
                      noise_fun_kwargs=dict(noise_strength=_noise_strength))
    _result = _model.run(sim_params)
    _suborders = [order_parameter(state)[0] for state in _result]
    return np.average(_suborders[-_ssn:])


if __name__ == "__main__":
    starsize = 6
    G = networkx.Graph()
    G.add_edges_from([(0, i) for i in range(1, starsize)])
    matrix = networkx.to_numpy_array(G)
    resolution = 100
    noise_range = np.linspace(0, 3, resolution)
    averageing_number = 10000
    ssn = 120
    sim_params = {'timespan': 4 * 60, 'stepsize': 1 / 15}
    order_list = []
    for noise in noise_range:
        sub_order_list = []
        print(f'{noise/max(noise_range)*100:.1f}%')
        # for _ in range(averageing_number):
        #     initial_positions = np.random.uniform(-np.pi, np.pi, starsize)
        #     eigen_frequencies = np.append(np.array([1]), np.zeros(starsize - 1))
        #     model = Kuramoto(eigen_frequencies, initial_positions,
        #                      coupling_fun=matrix_coupling,
        #                      coupling_fun_kwargs=dict(mat=matrix),
        #                      noise_fun=noise_fun,
        #                      noise_fun_kwargs=dict(noise_strength=noise))
        #     result = model.run(sim_params)
        #     suborders = [order_parameter(state)[0] for state in result]
        #     sub_order_list.append(np.average(suborders[-ssn:]))
        # order_list.append(sum(sub_order_list) / len(sub_order_list))
        args = [(starsize, matrix, noise, ssn) for _ in range(averageing_number)]
        result = pm.starmap(multi_target, args, pm_pbar=True, pm_processes=min(4, averageing_number))
        order_list.append(np.mean(result))
    plt.plot(noise_range, order_list)
    plt.xlim([min(noise_range), max(noise_range)])
    plt.xlabel('Noise strength')
    plt.ylabel('Order parameter')
    plt.show()
