from cmath import phase
from collections import deque
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Qt5Agg')
from copy import deepcopy
import matplotlib.pyplot as plt
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy
from matplotlib import animation
from numba import jit, njit

def default_coupling(n, k=1):
    return k / n * np.ones((n, n))


def nearest_neighbour(n, k=1):
    return k * (np.eye(n, n, 1) + np.eye(n, n, -1))


def default_noise(n, D=1):
    return D * np.random.normal(0, 1, n)


def order_parameter(state):
    number = sum(np.exp(state * 1j)) / len(state)
    return abs(number), phase(number)


def binder_cumulant(inlist, stationaryno):
    matrix = np.array(inlist)
    rho4 = np.mean(matrix[:, -stationaryno:]**4, 1)
    rho2 = np.mean(matrix[:, -stationaryno:]**2, 1)
    fraction = rho4/(3*(rho2**2))
    return 1-np.mean(fraction)


def fluctuation(inlist, L, stationaryno):
    matrix = np.array(inlist)
    rho2 = np.mean(matrix[:, -stationaryno:]**2, 1)
    rho = np.mean(matrix[:, -stationaryno:], 1)
    avterm = np.mean(rho2-rho**2)
    return avterm*(L**2)

class Kuramoto:

    def __init__(self, osc_freqs: np.ndarray, initials: np.ndarray,
                 coupling_fun: type(default_coupling) = default_coupling,  # defines the function which generates the coupling matrix
                 coupling_fun_kwargs: dict = dict(),  # give the arguments for said function
                 noise_fun: type(default_coupling) = default_noise,  # defines the noise generating function
                 noise_fun_kwargs: dict = dict(),
                 static_coupling: bool = True):  # plus its arguments
        self.oscillator_states_ini = initials
        self.n = len(initials)
        self.oscillator_feqs = osc_freqs
        self.oscillator_count = len(initials)
        self.coupling_fun = coupling_fun
        self.coupling_fun_kwargs = coupling_fun_kwargs
        self.noise_fun = noise_fun
        self.noise_fun_kwargs = noise_fun_kwargs
        self.solution = None  # Initialization
        if static_coupling:  # switches for faster execution
            self.coupling_matrix = self.coupling_fun(self.n, **self.coupling_fun_kwargs)
            if self.noise_fun == default_noise:
                self.differential_fun = self.theta_diff_static_gaussian
            else:
                self.differential_fun = self.theta_diff_static

    @staticmethod
    def strided_method(ar):  # This function is a fast method for calculating a circulant matrix
        ar = ar[::-1]
        a = np.concatenate((ar, ar[:-1]))
        L = len(ar)
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a[L - 1:], (L, L), (-n, n))

    # def theta_diff_2(self, t, states, freqs):  # differential
    #     n = len(states)
    #     ownstates = np.tile(states, (n, 1)).transpose()
    #     rolledstates = self.strided_method(states)
    #     calc = np.sum(self.coupling_fun(n, **self.coupling_fun_kwargs) * (np.sin(rolledstates - ownstates)),
    #                   axis=1)
    #     noise = self.noise_fun(n, **self.noise_fun_kwargs)
    #     return freqs + calc + noise
    #
    # def run2(self, params: dict):  # scipy method for solving ivps
    #     solution = solve_ivp(self.theta_diff_2,
    #                          t_span=params['timespan'],
    #                          y0=self.oscillator_states_ini,
    #                          method=params['method'],
    #                          args=(self.oscillator_feqs,),
    #                          rtol=params['rtol'],
    #                          atol=params['atol'],
    #                          max_step=params['max step'])
    #     return solution

    def theta_diff(self, dt, states, freqs):  # calculates the differential vector
        ownstates = np.tile(states, (self.n, 1)).transpose()  # This creates a sq matrix with each state duplicated on the row
        rolledstates = self.strided_method(states)  # This circulates the states
        calc = np.sum(self.coupling_fun(self.n, **self.coupling_fun_kwargs) * (np.sin(rolledstates - ownstates)),
                      axis=1)
        noise = self.noise_fun(self.n, **self.noise_fun_kwargs)
        return dt * freqs + dt * calc + np.sqrt(dt) * noise

    def theta_diff_static(self, dt, states, freqs):  # calculates the differential vector
        ownstates = np.tile(states, (self.n, 1))  # This creates a sq matrix with each state duplicated on the column
        calc = np.sum(self.coupling_matrix * (np.sin(ownstates-ownstates.T)), axis=1)
        noise = self.noise_fun(self.n, **self.noise_fun_kwargs)
        return dt * freqs + dt * calc + np.sqrt(dt) * noise

    def theta_diff_static_gaussian(self, dt, states, freqs):  # calculates the differential vector
        ownstates = np.tile(states, (self.n, 1))  # This creates a sq matrix with each state duplicated on the column
        calc = np.sum(self.coupling_matrix * (np.sin(ownstates-ownstates.T)), axis=1)
        noise = np.random.normal(0, self.noise_fun_kwargs['D'], self.n)
        return dt * freqs + dt * calc + np.sqrt(dt) * noise

    def theta_diff_linear(self, dt, states, freqs):
        n = len(states)
        ownstates = np.tile(states, (n, 1)).transpose()
        rolledstates = self.strided_method(states)
        calc = np.sum(self.coupling_fun(n, **self.coupling_fun_kwargs) * ((rolledstates - ownstates)),
                      axis=1)
        noise = self.noise_fun(n, **self.noise_fun_kwargs)
        return dt * freqs + dt * calc + np.sqrt(dt) * noise

    def run(self, params: dict):
        t = 0
        timestep = params['stepsize']
        solution = deque([self.oscillator_states_ini])
        while t < params['timespan']:
            dtheta = self.differential_fun(timestep, solution[-1], self.oscillator_feqs)
            solution.append(solution[-1]+dtheta)
            t += timestep
        return np.array(solution)

    def run_linear(self, params: dict):
        t = 0
        timestep = params['stepsize']
        states = deepcopy(self.oscillator_states_ini)
        solution = [states]
        while t < params['timespan']:
            dtheta = self.theta_diff_linear(timestep, states, self.oscillator_feqs)
            states += dtheta
            # states = states % (2*np.pi)
            solution.append(states)
            t += timestep
        return np.array(solution, dtype=object)

    @staticmethod
    def draw(solution, timespan):
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
    syssize = 100
    eigen_freqs = np.zeros(syssize)
    pos_ini = np.random.uniform(0,2*np.pi,syssize)

    sim_params = {
        'timespan': 5,
        'stepsize': 1/60
    }

    model = Kuramoto(eigen_freqs, pos_ini, noise_fun_kwargs=dict(D=0))
    result = model.run(sim_params)
    model.draw(result, 5)

    # plt.hist(appendices, range=(-0.01, 0.01), bins=100, weights=np.ones_like(appendices) / len(appendices))

    # movie = model.draw(result.y, sim_params['timespan'][1])
    #
    # plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\guita\PycharmProjects\BEP\code\bin\ffmpeg.exe"
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=60, bitrate=8192, codec='libx265')
    # movie.save('100osck0.3r0.1.mp4', writer=writer)
