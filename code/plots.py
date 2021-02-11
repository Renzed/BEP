from model import binder_cumulant, fluctuation, Kuramoto
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle

with open('10x10 full result 20s quenched.pkl', 'rb') as f:
    d10 = pickle.load(f)

with open('20x20 full result 20s quenched.pkl', 'rb') as f:
    d20 = pickle.load(f)

with open('30x30 full result 20s quenched.pkl', 'rb') as f:
    d30 = pickle.load(f)

ssn = 60

plt.figure()
plt.xlabel('Noise strength D')
plt.ylabel('Binder cumulant')
b10 = [binder_cumulant(d10[i], ssn) for i in d10]
b20 = [binder_cumulant(d20[i], ssn) for i in d20]
b30 = [binder_cumulant(d30[i], ssn) for i in d30]
x = [i for i in d10]
plt.xlim(min(x), max(x))
plt.ylim(0.32, 0.67)
plt.plot(x, b10, 'o-', label='10x10')
plt.plot(x, b20, 'x-', label='20x20')
plt.plot(x, b30, 'x-', label='30x30')
plt.legend()

plt.figure()
plt.xlabel('Noise strength D')
plt.ylabel('Fluctuation')
f10 = [fluctuation(d10[i], 10, ssn) for i in d10]
f20 = [fluctuation(d20[i], 20, ssn) for i in d20]
f30 = [fluctuation(d30[i], 30, ssn) for i in d30]
plt.xlim(min(x), max(x))
plt.plot(x, f10, 'o-', label='10x10')
plt.plot(x, f20, 'x-', label='20x20')
plt.plot(x, f30, 'x-', label='30x30')
plt.legend()
plt.show()
