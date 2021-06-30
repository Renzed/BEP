from model import XY, order_parameter_local
import numpy as np

vortex_seeds_30 = [0,456]

if __name__ == "__main__":
    seed = 0
    entries = dict()
    quenched = 0.05
    while 1:
        print(quenched, seed)
        np.random.seed(seed)
        single_axis_size = 50
        nosc = single_axis_size**2
        # eigen_freqs = np.zeros(nosc)
        eigen_freqs = np.random.normal(0, quenched, nosc)
        initials = np.random.uniform(-np.pi, np.pi, nosc)
        model = XY(eigen_freqs, initials, noise_fun_kwargs=dict(D=0), static_coupling=True)
        sim_params = {"timespan": 4*60, "stepsize": 1/15}
        result = model.run(sim_params)
        check = order_parameter_local(result[-1], (single_axis_size, single_axis_size))
        if np.any(check < .8):
            seed = 0
            quenched += .05
            print('yay')
            model.draw(result[-1], (single_axis_size, single_axis_size))
            entries[quenched] = seed
        else:
            seed += 1
        if quenched > 1:
            break
