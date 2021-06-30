import networkx
import numpy as np
import matplotlib.pyplot as plt
from model import Kuramoto, order_parameter, binder_cumulant, fluctuation, velocity
from scipy.optimize import curve_fit as cf
from tqdm import tqdm
import itertools as it
import parmap as pm


def finder(a):
    c = np.concatenate([a, a])
    N = len(a)
    b = np.roll(c, 1)
    rmax, lmax = None, None
    temp = 0
    mu = np.repeat(np.mean(a), 2 * N)
    for r in range(1, 1 + N):
        for l in range(r, r + N):
            calc = np.abs(np.sum(mu[r:l + 1] - b[r:l + 1]))
            if calc > temp:
                temp = np.abs(np.sum(mu[r:l + 1] - b[r:l + 1]))
                rmax, lmax = r, l
                print(temp, r, l)
    print(f"rmax={rmax}, lmax={lmax}, Ks={temp}, n*={lmax + 1 - rmax}")
    new_arr = np.roll(c, 1-rmax)
    return rmax, lmax, lmax + 1 - rmax, temp, new_arr


def separation(mu1, mu2, sig1, sig2):
    return np.abs(mu1 - mu2) / (2 * (sig1 + sig2))


def bim_gaussian(x, mu1, mu2, sig):
    gauss1 = np.exp(-(x - mu1) ** 2 / (2 * sig ** 2)) / np.sqrt(8 * np.pi * sig ** 2)
    gauss2 = np.exp(-(x - mu2) ** 2 / (2 * sig ** 2)) / np.sqrt(8 * np.pi * sig ** 2)
    return gauss1 + gauss2


def multiprocessed_kuramoto(sigma, mat, mean_array, av_length, number_of_bins, histogram_range,
                            n_strength):  # we define a function so that we can multiprocess
    eigenfreqs = np.random.normal(mean_array, sigma)
    # initpos = np.random.uniform(-np.pi, np.pi, len(mean_array))
    d1 = np.repeat(np.random.uniform(-np.pi, np.pi, int(len(mean_array) / 2)))
    d2 = np.repeat(np.random.uniform(-np.pi, np.pi, int(len(mean_array) / 2)))
    initpos = np.append(d1, d2)
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
    ring_left = np.array([1.98725176, -2.06287988, 1.93896729, -0.14578174, 1.60407108,
                          -4.36029618, -13.67023732, -0.82889337, 2.78286337, 3.05041795])
    ring_right = np.array([9.5251671, -5.87193306, -2.50697559, -0.98473714, 1.3264844,
                           -9.50824015, -1.79208611, 0.28492023, -5.1938431, -1.38382884])
    ring_paper = np.array([6.9, 2.8, -0.4, -2.6, 1.3, -6.8, 0.8, -1.6, -9.5, -6.7])
    halter_size = 10
    conn_length = 0
    ssn = 60
    nbins = 11
    n_res = 3
    n_cores = 5
    noise_range = np.linspace(0, 2.5, n_res)
    quenched_noise_range = [0.2]  # np.linspace(0, 2, 10)
    shifted_mean = 1
    means = [0 if i < halter_size + conn_length / 2 else shifted_mean for i in range(2 * halter_size + conn_length)]
    averaging_number = 10000
    matrix = networkx.to_numpy_array(networkx.barbell_graph(halter_size, conn_length))
    popt_arr = []
    plot = False
    for noise in noise_range:
        for _ in tqdm(range(averaging_number)):
            histrange = [-noise - 1, 1 + shifted_mean + int(np.ceil(noise / 1.5))]
            eigenfreqs = np.random.normal(means, 0)
            initpos = np.random.uniform(-np.pi, np.pi, 2 * halter_size + conn_length)
            model = Kuramoto(eigenfreqs, initpos, matrix_coupling, coupling_fun_kwargs=dict(mat=matrix),
                             noise_fun_kwargs=dict(D=noise))
            params = {'timespan': 4 * 60,
                      'stepsize': 1 / 15}
            result = model.run(params)
            sep_lst = []
            ssn = 60
            div = int(600 / ssn)
            for i in range(int((len(result) - 1) / ssn)):
                heights, edges = np.histogram(np.mean(velocity(result[i:i + ssn], params['stepsize']), axis=0),
                                              bins=nbins, range=histrange,
                                              weights=np.ones_like(initpos) / initpos.size, density=True)
                centers = .5 * (edges[1:] + edges[:-1])
                try:
                    popt, pcov = cf(bim_gaussian, centers, heights, p0=[0, 1, .2])
                except RuntimeError:
                    popt = [0, 0, 1]
                except ValueError:
                    print(result[i:i + 2])
                    print(velocity(result[i:i + 2], params['stepsize'])[0])
                    print(heights, centers)
                if i % div == 0 and plot:
                    plt.figure()
                    plt.hist(edges[:-1], edges, weights=heights)
                    xax = np.linspace(min(edges), max(edges), 1000)
                    plt.plot(xax, bim_gaussian(xax, popt[0], popt[1], popt[2]))
                sep_lst.append(separation(popt[0], popt[1], popt[2], popt[2]))
            try:
                sep_array += np.array(sep_lst) / averaging_number
            except Exception:
                sep_array = np.array(sep_lst) / averaging_number
        xax = np.linspace(0, 4 * 60, len(sep_array))
        plt.plot(xax, sep_array, label=f'{noise}')
        plt.xlabel('Time (s)')
        plt.ylabel('Separation parameter')
        plt.legend()
        print(separation(0, 1, .1, .1))
        sep_array = np.zeros_like(sep_array)
    plt.show()
