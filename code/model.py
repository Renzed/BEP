from cmath import phase
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import itertools as it

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy
from matplotlib import animation


def default_coupling(n, k=1):
    return k/n * np.ones((n, n))


def nearest_neighbour(n, k=1):
    return k * (np.eye(n, n, 1) + np.eye(n, n, -1))


def default_noise(n, D=1):
    return D * np.random.normal(0, 1, n)


def order_parameter(state):
    number = sum(np.exp(state*1j))/len(state)
    return abs(number), phase(number)

class Kuramoto:

    def __init__(self, osc_freqs: np.ndarray, initials: np.ndarray,
                 coupling_fun: type(default_coupling) = default_coupling,
                 coupling_fun_kwargs: dict = dict(),
                 noise_fun: type(default_coupling) = default_noise,
                 noise_fun_kwargs: dict = dict()):
        self.oscillator_states_ini = initials
        self.oscillator_feqs = osc_freqs
        self.oscillator_count = len(initials)
        self.coupling_fun = coupling_fun
        self.coupling_fun_kwargs = coupling_fun_kwargs
        self.noise_fun = noise_fun
        self.noise_fun_kwargs = noise_fun_kwargs
        self.solution = None

    @staticmethod
    def strided_method(ar):
        ar = ar[::-1]
        a = np.concatenate((ar, ar[:-1]))
        L = len(ar)
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a[L - 1:], (L, L), (-n, n))

    def theta_diff(self, t, states, freqs):
        n = len(states)
        ownstates = np.tile(states, (n, 1)).transpose()
        rolledstates = self.strided_method(states)
        calc = np.sum(self.coupling_fun(n, **self.coupling_fun_kwargs) * ((rolledstates - ownstates)),
                      axis=1)
        noise = self.noise_fun(n, **self.noise_fun_kwargs)
        out = freqs + calc + noise
        return freqs + calc + noise

    def run(self, params: dict):
        solution = solve_ivp(self.theta_diff,
                             t_span=params['timespan'],
                             y0=self.oscillator_states_ini,
                             method=params['method'],
                             args=(self.oscillator_feqs,),
                             rtol=params['rtol'],
                             atol=params['atol'],
                             max_step=params['max step'])
        return solution

    @staticmethod
    def draw(solution, timespan, filename=None):
        fig, ax = plt.subplots()
        size = len(solution[0])
        x, y = np.cos(solution[:, 0]), np.sin(solution[:, 0])

        mat, = ax.plot(x, y, 'o')

        def animate_matplotlib(j):
            i = int(j * size / (60 * timespan))
            x, y = np.cos(solution[:, i]), np.sin(solution[:, i])
            mat.set_data(x, y)
            ax.set_title(f't={i * timespan / size:.2f} s')
            return mat,

        # def animate_moviepy(j):
        #     i = int(j*size/timespan)
        #     x, y = np.cos(solution[:, i]), np.sin(solution[:, i])
        #     mat.set_data(x, y)
        #     ax.set_title('t=' + str(i * timespan / size))
        #     return mplfig_to_npimage(fig)

        ax.axis([-1.5, 1.5, -1.5, 1.5])
        anim = animation.FuncAnimation(fig, animate_matplotlib, frames=int(60 * timespan), interval=10 ** 3 / 60,
                                       blit=False)
        return anim


if __name__ == '__main__':
    # eigen_freqs = np.random.normal(1, 2, 100)
    # pos_ini = np.random.uniform(0, 2 * np.pi, 100)

    # model = Kuramoto(eigen_freqs, pos_ini)


    sim_params = {
        'timespan': [0, 20],
        'method': 'RK45',
        'rtol': 1e90,
        'atol': 1e90,
        'max step': 20 / 2400
    }

    eigen_freqs = np.zeros(100)
    pos_ini = np.random.uniform(-1e-3, 1e-3, 100)

    ntc_range = np.array([0.01,0.1,0.5,2.5])
    ntc = 0.1
    appendices = np.array([])
    for i in tqdm(range(1000)):
        model = Kuramoto(eigen_freqs, pos_ini,
                         coupling_fun=nearest_neighbour,
                         noise_fun_kwargs=dict(D=ntc))
        result = model.run(sim_params)
        appendices = np.append(appendices, result.y[:, -1])

    plt.hist(appendices, range=(-0.01, 0.01), bins=100, weights=np.ones_like(appendices)/len(appendices))

    # movie = model.draw(result.y, sim_params['timespan'][1])
    #
    # plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\guita\PycharmProjects\BEP\code\bin\ffmpeg.exe"
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=60, bitrate=8192, codec='libx265')
    # movie.save('100osck0.3r0.1.mp4', writer=writer)
