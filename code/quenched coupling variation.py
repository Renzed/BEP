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


def gen_coupled_rings(ringsize, coupling, n_rings):
    nodes = []
    edges = []
    conn_edges = []
    for i in range(n_rings):
        nodes += [j + 1 + ringsize * i for j in range(ringsize)]
        edges += [(j + 1 + i * ringsize, (j + 1) % ringsize + 1 + i * ringsize) for j in range(ringsize)]
        conn_edges += [(1 + i * ringsize, ((i + 1) * ringsize + 1) % (n_rings * ringsize), coupling)]
    G = networkx.Graph()
    # print(nodes, edges, conn_edges)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G.add_weighted_edges_from(conn_edges)
    # networkx.draw(G)
    return networkx.to_numpy_array(G)


if __name__ == "__main__":
    halter_size = 10
    conn_length = 0
    ssn = 60
    nbins = 25
    n_res = 100
    n_cores = 5
    noise_range = np.linspace(0, 2, n_res)
    quenched_noise_range = [.2]  # np.linspace(0, 2, 10)
    shifted_mean = 1
    means = [0 if i < halter_size + conn_length / 2 else shifted_mean for i in range(2 * halter_size + conn_length)]
    averaging_number = 1000
    popt_arr = []
    for quenched_noise in quenched_noise_range:
        print(f"sigma={quenched_noise:.2f}")
        for noise_index in tqdm(range(len(noise_range))):
            histrange = [-noise_range[noise_index] - 1, 1 + shifted_mean + int(np.ceil(noise_range[noise_index] / 1.5))]
            matrix = gen_coupled_rings(halter_size, noise_range[noise_index], 2)
            sub_args = [(quenched_noise, matrix, means, ssn, nbins, histrange, noise_range[noise_index]) for i in
                        range(averaging_number)]
            temp_results = pm.starmap(multiprocessed_kuramoto, sub_args, pm_processes=min(n_cores, len(sub_args)))
            results = np.mean(temp_results, axis=0)
            edges = np.histogram([0], bins=nbins, range=histrange)[1]
            centers = (edges[1:] + edges[:-1]) / 2
            width = (edges[1] - edges[0]) * 0.9
            popt, pcov = cf(bim_gaussian, centers, results, p0=[0, 1, 0.2])
            plt.figure()
            plt.bar(centers, results, align='center', width=width)
            xax = np.linspace(min(edges), max(edges), 1000)
            popt_arr.append((noise_range[noise_index], popt))
            plt.plot(xax, bim_gaussian(xax, popt[0], popt[1], popt[2]), '-r')
            sep = (popt[1] - popt[0]) / (4 * popt[2])
            # plt.title(f'd={np.abs(popt[1] - popt[0]):.1e}, $\sigma$={popt[2]:.1e}, sep={sep:.2e}')
            plt.xlabel('Velocity (rad/s)')
            plt.ylabel('Probability')
            plt.savefig(
                f'figures/10plus10plus0ring/sigma{quenched_noise:.2f}K{noise_range[noise_index]:.2f}gap{shifted_mean}average{averaging_number}.png')
            plt.close('all')
        plt.figure()
        plt.xlabel('Noise strength $D$')
        plt.ylabel('Separation parameter')
        plt.plot(noise_range, [np.abs(separation(i[1][0], i[1][1], i[1][2], i[1][2])) for i in popt_arr])
        plt.savefig(f'figures/10plus10plus0ring/sigma{quenched_noise:.2f}separation.png')
        # with open(f'sigma{quenched_noise:.2f}average{averaging_number}.npy', 'wb') as f:
        #     np.save(f, np.array(popt_arr, dtype=object))
