import numpy as np
import matplotlib.pyplot as plt
import networkx
import itertools as it
from verification import multiprocessed_kuramoto
from model import Kuramoto, order_parameter, binder_cumulant, fluctuation
import parmap


def matrix_coupling(n, mat=''):
    if isinstance(mat, str):
        raise ValueError('No matrix found')
    else:
        return mat


def multiprocessed_kuramoto(ef, pos, n_strength, mat):
    sim_params = {
        'timespan': 2*60,
        'stepsize': 1 / 30
    }
    model = Kuramoto(ef, pos,
                     coupling_fun=matrix_coupling,
                     coupling_fun_kwargs=dict(mat=mat),
                     noise_fun_kwargs=dict(D=n_strength))
    subresult = model.run(sim_params)
    return dict(output=subresult, noise=n_strength)


if __name__ == '__main__':
    nosc = 100
    msims = 10
    edgerange = (np.linspace(1, np.sqrt(nosc*(nosc-1)/2), msims) ** 2).astype(np.int32)
    # edgerange = np.linspace(1, nosc-1, msims).astype(np.int32)
    eigenfreqs = np.zeros(nosc)
    initialpos = np.random.uniform(0, 2 * np.pi, nosc)
    noisestrength = 0
    edgecount = 1
    for edges in edgerange:
        network = networkx.gnm_random_graph(nosc, edges)
        # network = networkx.barabasi_albert_graph(nosc, edges)
        matrix = networkx.to_numpy_array(network)*4/max(2*edges/nosc, 1)
        nsims = 50
        noise_range = np.linspace(0, 3, nsims)
        print(f'Starting for network with {edgecount} edge(s)...')
        averageingnum = 25
        orders = np.zeros(nsims)
        full_order_result = {noise: [] for noise in noise_range}
        plt.figure()
        plt.subplot(131)
        networkx.draw(network)
        plt.subplot(132)
        for k in range(averageingnum):
            eigen_freqs = np.zeros(nosc)
            pos_ini = np.random.uniform(0, 2 * np.pi, nosc)
            args = list(it.product([eigen_freqs], [pos_ini], noise_range, [matrix]))
            # subargs = list(it.product(eigen_freqs, [pos_ini]))
            # args = [tuple(list(subargs[i])+[noise_range[i]]) for i in range(nsims)]
            print(f'Calculating run {edgecount}x{k + 1} out of {msims}x{averageingnum}')
            results = parmap.starmap(multiprocessed_kuramoto, args, pm_pbar=True, pm_processes=min(5, nsims))

            noises = []
            orderplot = []
            for i in results:
                noises.append(i['noise'])
                suborders = np.array([order_parameter(i['output'][j, :].astype(np.float64))
                                      for j in range(len(i['output'][:, 0]))])[:, 0]
                full_order_result[i["noise"]].append(suborders)
                orderplot.append(np.average(suborders[-600:]))

            orders += np.array(orderplot) / averageingnum
        edgecount += 1
        plt.xlabel('Noise-strength D')
        plt.ylabel('Order parameter')
        plt.ylim(0, 1.01)
        plt.xlim(min(noises), max(noises))
        plt.plot(noises, orders)
        plt.subplot(133)
        plt.xlabel('Noise-strength D')
        plt.ylabel('Binder cumulant')
        b10 = [binder_cumulant(full_order_result[i], 600) for i in full_order_result]
        plt.ylim(0.32, 0.67)
        plt.xlim(min(noises), max(noises))
        plt.plot(noises, b10)
    plt.show()
