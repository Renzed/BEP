import networkx
import numpy as np
import matplotlib.pyplot as plt
from model import Kuramoto, order_parameter, binder_cumulant, fluctuation, velocity
from scipy.optimize import curve_fit as cf
from tqdm import tqdm
import itertools as it
import parmap as pm


def separation(mu1, mu2, sig1, sig2):
    return np.abs(mu1 - mu2) / (2 * (sig1 + sig2))


def bim_gaussian(x, mu1, mu2, sig):
    gauss1 = np.exp(-(x - mu1) ** 2 / (2 * sig ** 2)) / np.sqrt(8 * np.pi * sig ** 2)
    gauss2 = np.exp(-(x - mu2) ** 2 / (2 * sig ** 2)) / np.sqrt(8 * np.pi * sig ** 2)
    return gauss1 + gauss2


def multiprocessed_kuramoto(sigma, mat, mean_array, av_length, number_of_bins, histogram_range,
                            n_strength):  # we define a function so that we can multiprocess
    eigenfreqs = np.random.normal(mean_array, sigma)
    initpos = np.random.uniform(-np.pi, np.pi, len(mean_array))
    model = Kuramoto(eigenfreqs, initpos, matrix_coupling,
                     coupling_fun_kwargs=dict(mat=mat), noise_fun_kwargs=dict(D=n_strength))
    prms = {'timespan': 4 * 60, 'stepsize': 1 / 15}
    return np.histogram(np.mean(velocity(model.run(prms)[-av_length:], prms['stepsize']), axis=0), bins=number_of_bins,
                        range=histogram_range, weights=np.ones_like(initpos) / initpos.size, density=True)[0]


def matrix_coupling(n, mat=''):
    if isinstance(mat, str):
        raise ValueError('No matrix found')
    else:
        return mat


if __name__ == "__main__":
    halter_size = 10
    conn_length = 0
    ssn = 60
    nbins = 25
    n_res = 3
    n_cores = 5
    noise_range = np.linspace(0, 2.5, n_res)
    quenched_noise_range = [0]  # np.linspace(0, 2, 10)
    shifted_mean = 1
    means = [0 if i < halter_size + conn_length / 2 else shifted_mean for i in range(2 * halter_size + conn_length)]
    averaging_number = 1000
    matrix = networkx.to_numpy_array(networkx.barbell_graph(halter_size, conn_length))
    popt_arr = []
    for noise in noise_range:
        for _ in range(averaging_number):
            histrange = [-noise - 1, 1 + shifted_mean + int(np.ceil(noise / 1.5))]
            eigenfreqs = np.random.normal(means, 0)
            initpos = np.random.uniform(-np.pi, np.pi, 2 * halter_size + conn_length)
            model = Kuramoto(eigenfreqs, initpos, matrix_coupling, coupling_fun_kwargs=dict(mat=matrix),
                             noise_fun_kwargs=dict(noise))
            params = {'timespan': 4 * 60,
                      'stepsize': 1 / 15}
            result = model.run(params)
            sep_lst = []
            for time_state in result:
                heights, edges = np.histogram(velocity(time_state, params['stepsize']), bins=nbins, range=histrange,
                                              weights=np.ones_like(initpos) / initpos.size, density=True)
                centers = .5 * (edges[1:] + edges[:-1])
                popt, pcov = cf(bim_gaussian, centers, heights, p0=[0, 1, .2])
                sep_lst.append(separation(popt[0], popt[1], popt[2], popt[2]))
                try:
                    sep_array += np.array(sep_lst) / averaging_number
                except Exception:
                    sep_array = np.array(sep_lst) / averaging_number
