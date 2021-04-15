from model import Kuramoto, velocity
import numpy as np
import matplotlib.pyplot as plt
import parmap as pm
from scipy.stats import chi2


def multiprocessed_kuramoto(ef, pos, n_strength):  # we define a function so that we can multiprocess
    sim_params = {
        'timespan': 4 * 60,
        'stepsize': 1 / 15
    }
    model = Kuramoto(ef, pos,
                     # coupling_fun=square_nn_coupling,
                     noise_fun_kwargs=dict(D=1*n_strength))
    subresult = model.run(sim_params)
    return dict(output=subresult, noise=n_strength)


if __name__ == "__main__":
    nsims = 100
    noise_range = np.linspace(0, 9, nsims)
    averagingnum = 800
    nosc = 3
    ssn = 300
    full_velocity_result = {noise: [] for noise in noise_range}
    factors_tot = np.zeros(nsims)
    factors_partial = np.zeros(nsims)
    for _ in range(averagingnum):
        print(f"{_}/{averagingnum}")
        eigen_freqs = np.random.normal(0, noise_range, (nosc, nsims)).T
        pos_ini = np.random.uniform(0, 2 * np.pi, (nosc, nsims)).T
        args = [tuple([eigen_freqs[i]] + [pos_ini[i]] + [noise_range[i]]) for i in range(nsims)]
        results = pm.starmap(multiprocessed_kuramoto, args, pm_pbar=True, pm_processes=min(2, nsims))
        for res in results:
            full_velocity_result[res['noise']].append(
                np.average(velocity(res['output'].astype(np.float64), 1 / 15)[-ssn:, :],
                           axis=0))

        for j, i in enumerate(results):
            factors_tot[j] += int(np.max(np.histogram(full_velocity_result[i['noise']][-1], bins=11, range=(-i['noise']*2,i['noise']*2), weights=np.ones(nosc)/nosc)[0]))/averagingnum
            factors_partial[j] += int(np.max(
                np.histogram(full_velocity_result[i['noise']][-1], bins=11, range=(-i['noise'] * 2, i['noise'] * 2),
                             weights=np.ones(nosc) / nosc)[0]) > 0.6) / averagingnum

    def totsync(x, y):
        return chi2.cdf(2/(x**2+y**2+1e-12), 1)**2

    def partsync(x, y):
        return 1-(1-chi2.cdf(2/(x**2+y**2+1e-12), 1))**2

    plt.plot(noise_range, factors_partial, label='Simulated')
    plt.xlabel(u'Noise strength $\sqrt{\sigma^2+D^2}$')
    plt.ylabel('Probability of partial synchronization')
    plt.plot(noise_range, partsync(np.sqrt(2)*noise_range, np.zeros(noise_range.shape)), label='Calculated')
    plt.legend()
    plt.figure()

    plt.plot(noise_range, factors_tot,label='Simulated')
    plt.xlabel(u'Noise strength $\sqrt{\sigma^2+D^2}$')
    plt.ylabel('Probability of total synchronization')
    plt.plot(noise_range, totsync(np.sqrt(2)*noise_range, np.zeros(noise_range.shape)), label='Calculated')
    plt.legend()
    plt.show()
