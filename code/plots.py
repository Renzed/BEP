from model import binder_cumulant, fluctuation, Kuramoto
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle

with open('4x4 full result 120s quenched.pkl','rb') as f:
    d4 = pickle.load(f)

with open('7x7 full result 120s quenched.pkl','rb') as f:
    d7 = pickle.load(f)

with open('10x10 full result annealed 240s quenched 1div15dt frsk.pkl', 'rb') as f:
    d10 = pickle.load(f)

# with open('20x20 full result annealed 240s quenched 1div15dt frsk.pkl', 'rb') as f:
#     d20 = pickle.load(f)
#
# with open('30x30 full result annealed 240s quenched 1div15dt frsk.pkl', 'rb') as f:
#     d30 = pickle.load(f)

ssn = 900

plt.figure()
plt.xlabel('Noise strength D')
plt.ylabel('Binder cumulant')
b4 = [binder_cumulant(d4[i], ssn) for i in d4]
b7 = [binder_cumulant(d7[i], ssn) for i in d7]
b10 = [binder_cumulant(d10[i], ssn) for i in d10]
# b20 = [binder_cumulant(d20[i], ssn) for i in d20]
# b30 = [binder_cumulant(d30[i], ssn) for i in d30]
x = [i for i in d10]
plt.xlim(min(x), max(x))
plt.ylim(0.32, 0.67)
plt.plot(x, b4, '>-', label='4x4')
plt.plot(x, b7, '<-', label='7x7')
plt.plot(x, b10, 'o-', label='10x10')
# plt.plot(x, b20, 'x-', label='20x20')
# plt.plot(x, b30, 'x-', label='30x30')
plt.legend()

# plt.figure()
# plt.xlim(min(x), max(x))
# plt.xlabel('Noise strength D')
# plt.ylabel('Fluctuation')
# f10 = [fluctuation(d10[i], 10, ssn) for i in d10]
# f20 = [fluctuation(d20[i], 20, ssn) for i in d20]
# f30 = [fluctuation(d30[i], 30, ssn) for i in d30]
# plt.plot(x, f10, 'o-', label='10x10')
# # plt.plot(x, f20, 'x-', label='20x20')
# # plt.plot(x, f30, 'x-', label='30x30')
# plt.legend()

plt.figure()
plt.xlabel('Noise strength D')
plt.ylabel('Order parameter')
o4 = [np.average(np.array(d4[i])[:, -ssn:]) for i in d4]
o7 = [np.average(np.array(d7[i])[:, -ssn:]) for i in d7]
o10 = [np.average(np.array(d10[i])[:, -ssn:]) for i in d10]
# o20 = [np.average(np.array(d20[i])[:, -ssn:]) for i in d20]
# o30 = [np.average(np.array(d30[i])[:, -ssn:]) for i in d30]
plt.xlim(min(x),max(x))
plt.plot(x, o4, '>-', label='4x4')
plt.plot(x, o7, '<-', label='7x7')
plt.plot(x, o10, 'o-', label='10x10')
plt.xlim(min(x), max(x))
# plt.plot(x, o20, '.-', label='20x20')
# plt.xlim(min(x), max(x))
# plt.plot(x, o30, '^-', label='30x30')
# plt.legend()

plt.show()

plt.figure()