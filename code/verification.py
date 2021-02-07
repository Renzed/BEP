from model import Kuramoto, nearest_neighbour, order_parameter
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import itertools as it
import parmap
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit


def square_nn_coupling(n, K=1):
    n = int(n ** (1 / 2))
    matrix = np.zeros(shape=(n ** 2, n ** 2))
    for i in range(n):
        for j in range(n):
            if j - i == -1 or j - i == 1:
                addition = np.eye(n, n, 0)
            elif j - i == 0:
                addition = np.eye(n, n, 1) + np.eye(n, n, -1) + np.eye(n, n, n-1) + np.eye(n, n, 1-n)
            else:
                addition = np.eye(n, n, n)
            matrix[n * i:n * i + n, n * j:n * j + n] = addition
    return K * matrix


def multiprocessed_kuramoto(ef, pos, n_strength):
    sim_params = {
        'timespan': 5,
        'stepsize': 1/30
    }
    model = Kuramoto(ef, pos,
                     coupling_fun=square_nn_coupling,
                     noise_fun_kwargs=dict(D=n_strength))
    subresult = model.run(sim_params)
    return dict(output=subresult, noise=n_strength)


if __name__ == "__main__":

    # for dictionary in results:
    #     result = dictionary['output']
    #     noise_strength = dictionary['noise']
    #     plt.figure()
    #     l = len(result.t)
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Order parameter')
    #     nbyn = int(np.sqrt(len(eigen_freqs)))
    #     plt.title(f'K=1, D={noise_strength}, coupling=square({nbyn}x{nbyn})')
    #     plt.ylim(0, 1)
    #     plt.plot(result.t, np.array([order_parameter(result.y[:, i]) for i in range(l)])[:, 0])

    # with open('backup.npy','wb') as f:
    #     np.save(f, results)

    nsims = 50
    noise_range = np.linspace(0, 3, nsims)

    eigen_freqs = np.zeros(100)
    pos_ini = np.random.uniform(0, 2 * np.pi, 100)
    args = list(it.product([eigen_freqs], [pos_ini], noise_range))

    averageingnum = 500
    orders = np.zeros(50)
    for k in range(averageingnum):
        print(f'Calculating run {k+1} out of {averageingnum}')
        results = parmap.starmap(multiprocessed_kuramoto, args, pm_pbar=True, pm_processes=min(5, nsims))

        noises = []
        suborders = []
        for i in results:
            noises.append(i['noise'])
            avterm = np.average(np.array([order_parameter(i['output'][-(j + 1), :].astype(np.float64))
                                          for j in range(60)])[:, 0])
            suborders.append(avterm)
        orders += np.array(suborders)/averageingnum
    # with open('20x20noisestrength.npy', 'wb') as f:
    #     np.save(f, orders)
    # plt.figure()
    plt.xlabel('Noise-strength D')
    plt.ylabel('Order parameter')
    plt.ylim(0, 1.01)
    plt.xlim(min(noises), max(noises))
    plt.plot(noises, orders)

    # def func(x, a, b, c):
    #     return c * x ** 6 + a * x ** 4 + b * x ** 2 + 1
    #
    #
    # popt, pcov = curve_fit(func, noises, orders)
    # xnew = np.linspace(min(noises), max(noises), 20)
    # spline = make_interp_spline(noises, orders, k=1)
    # smoothed = spline(xnew)
    # plt.plot(xnew, smoothed)
    # plt.plot(xnew, func(xnew, *popt))
