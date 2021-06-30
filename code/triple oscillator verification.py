from model import Kuramoto, velocity
import numpy as np
import matplotlib.pyplot as plt
import parmap as pm
from scipy.stats import chi2
from scipy.special import erf
import pickle


def multiprocessed_kuramoto(ef, pos, n_strength, scale):  # we define a function so that we can multiprocess
    sigma_internal = n_strength/np.sqrt(1+scale**2)
    D_noise = scale*sigma_internal
    sim_params = {
        'timespan': 4 * 60,
        'stepsize': 1 / 15
    }
    model = Kuramoto(ef, pos,
                     # coupling_fun=square_nn_coupling,
                     noise_fun_kwargs=dict(D=D_noise))
    subresult = model.run(sim_params)
    return dict(output=subresult, noise=n_strength)


if __name__ == "__main__":
    plt.figure()
    nsims = 100
    noise_range = np.linspace(0, 9, nsims)
    averagingnum = 100
    nosc = 3
    ssn = 300
    noise_scaling = [.1, 1, 10]
    for scale in noise_scaling:
        full_velocity_result = {noise: [] for noise in noise_range}
        factors_tot = np.zeros(nsims)
        factors_partial = np.zeros(nsims)
        sigma = noise_range/np.sqrt(1+scale**2)
        D = scale*sigma
        for _ in range(averagingnum):
            print(f"{noise_scaling.index(scale)+1}x{_}/{len(noise_scaling)}x{averagingnum}")
            eigen_freqs = np.random.normal(0, sigma, (nosc, nsims)).T
            pos_ini = np.random.uniform(0, 2 * np.pi, (nosc, nsims)).T
            args = [tuple([eigen_freqs[i]] + [pos_ini[i]] + [noise_range[i]] + [scale]) for i in range(nsims)]
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
        with open(f'scale{scale} trioscvertot.pkl', 'wb') as f:  # save data
            pickle.dump(factors_tot, f)
        with open(f'scale{scale} trioscverpart.pkl', 'wb') as f:  # save data
            pickle.dump(factors_partial, f)

    def totsync(x, y):
        return chi2.cdf(2/(x**2+y**2+1e-12), 1)**2

    def partsync(x, y):
        return 1-(1-chi2.cdf(2/(x**2+y**2+1e-12), 1))**2

    def strogatz(x):
        return np.sqrt(2)*erf(np.sqrt(2)/x)**3

    for scale in noise_scaling:
        plt.plot(noise_range, factors_partial[scale], label=f'Simulated, $D={scale}\\sigma$')
    plt.xlabel(u'Noise strength $\sqrt{\sigma^2+D^2}$')
    plt.ylabel('Probability of partial synchronization')
    plt.plot(noise_range, partsync(np.sqrt(2)*noise_range, np.zeros(noise_range.shape)), label='Calculated')
    plt.plot(noise_range, strogatz(noise_range), label='Strogatz & Mirollo')
    plt.xlim(min(noise_range), max(noise_range))
    plt.legend()
    plt.figure()

    for scale in noise_scaling:
        plt.plot(noise_range, factors_tot[scale], label=f'Simulated, $D={scale}\\sigma$')
    plt.xlabel(u'Noise strength $\sqrt{\sigma^2+D^2}$')
    plt.ylabel('Probability of total synchronization')
    plt.plot(noise_range, totsync(np.sqrt(2)*noise_range, np.zeros(noise_range.shape)), label='Calculated')
    plt.xlim(min(noise_range), max(noise_range))
    plt.legend()
    plt.show()
