from model import Kuramoto, nearest_neighbour, order_parameter, binder_cumulant, max_eigenvalue, velocity
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import itertools as it
import parmap
import pickle
import time
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


def multiprocessed_kuramoto(ef, pos, n_strength):  # we define a function so that we can multiprocess
    sim_params = {
        'timespan': 4*60,
        'stepsize': 1/15
    }
    model = Kuramoto(ef, pos,
                     # coupling_fun=square_nn_coupling,
                     noise_fun_kwargs=dict(D=0))
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

    nsims = 100  # amount of noise points
    noise_range = np.linspace(0, 2.5, nsims)
    nosc = 10**2
    mat_coupl = square_nn_coupling(nosc)
    ssn = 60

    averageingnum = 25  # amount of runs
    orders = np.zeros(nsims)
    factors = np.zeros(nsims)
    full_order_result = {noise: [] for noise in noise_range}
    full_velocity_result = {noise: [] for noise in noise_range}
    for k in range(averageingnum):
        eigen_freqs = list(np.random.normal(0, noise_range**2, (nosc, nsims)).transpose())
        pos_ini = np.random.uniform(0, 2 * np.pi, nosc)
        # args = list(it.product(eigen_freqs, [pos_ini], [0]))
        subargs = list(it.product(eigen_freqs, [pos_ini]))
        args = [tuple(list(subargs[i])+[noise_range[i]]) for i in range(nsims)]
        print(f'Calculating run {k+1} out of {averageingnum}')
        results = parmap.starmap(multiprocessed_kuramoto, args, pm_pbar=True, pm_processes=min(5, nsims))  # multiprocessing

        noises = []
        orderplot = []
        for j, i in enumerate(results):
            noises.append(i['noise'])
            # tic = time.perf_counter()
            # stability = np.array([max_eigenvalue(i['output'][j, :].astype(np.float64), mat_coupl)
            #                      for j in range(len(i['output'][:, 0]))])
            # print(f"Eigenvalues for run {k+1} and noise {i['noise']} were calculated in {time.perf_counter()-tic:.2f}s")
            full_velocity_result[i['noise']].append(np.average(velocity(i['output'].astype(np.float64), 1/15)[-ssn:, :],
                                                               axis=0))
            factor = np.max(np.histogram(full_velocity_result[i['noise']][-1], bins=31, range=(-10, 10),
                                         weights=np.ones(nosc)/nosc)[0])
            factors[j] += factor/averageingnum
            suborders = np.array([order_parameter(i['output'][j, :].astype(np.float64))
                                  for j in range(len(i['output'][:, 0]))])[:, 0]
            full_order_result[i["noise"]].append(suborders)
            orderplot.append(np.average(suborders[-ssn:]))

        orders += np.array(orderplot)/averageingnum
    # with open('10x10 full result 240s quenched 1/15dt.pkl', 'wb') as f:  # save data
    #     pickle.dump(full_order_result, f)
    # with open('10x10 velocity result 240s quenched 1/15dt.pkl', 'wb') as f:  # save data
    #     pickle.dump(full_velocity_result, f)
    # plt.figure()
    # plt.hist()
    plt.xlabel('Noise-strength D')
    plt.ylabel('Order parameter')
    plt.ylim(0, 1.01)
    plt.xlim(min(noises), max(noises))
    plt.plot(noises, orders)
    plt.figure()

    def fractionapprox(r, sigma, K=1):
        return r+(sigma/(K*r))*(np.exp(-K*r**2/(2*sigma**2)))/(np.sqrt(2*np.pi))

    plt.plot(noises, factors)
    plt.xlabel('Noise strength (D)')
    plt.ylabel('Synchronized fraction $f$')

    plt.figure()
    plt.plot(orders, factors, label='Simulated')
    plt.plot(orders, fractionapprox(orders, noise_range), label='Approximated')
    plt.legend()
    plt.xlabel('Order parameter $r$')
    plt.ylabel('Synchronized fraction $f$')

    plt.figure()
    plt.plot(noise_range/orders, factors, label='Simulated')
    plt.plot(noise_range/orders, fractionapprox(orders, noise_range), label='Approximated')
    plt.legend()
    plt.xlabel('$\\sigma/Kr$')
    plt.ylabel('Synchronized fraction $f$')

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
    plt.show()
