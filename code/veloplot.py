import matplotlib.pyplot as plt
import numpy as np
import pickle

Lsize = 10
ssn = 900

nosc = Lsize**2

with open(f"{Lsize}x{Lsize} velocity result 120s quenched.pkl", 'rb') as f:
    v10 = pickle.load(f)

with open(f'{Lsize}x{Lsize} full result 120s quenched.pkl', 'rb') as f:
    d10 = pickle.load(f)


o10 = np.array([np.average(np.array(d10[i])[:, -ssn:]) for i in d10])
noise = np.array([i for i in v10])
fs = []
# velocities = [np.mean(np.array(v10[i])[0, -ssn:-1, :], axis=0) for i in v10]
# print(np.array(velocities).shape)
velocities = np.zeros((len(v10), nosc))
for i in v10:
    temp = []
    print(np.array(v10[i]).shape)
    for j in range(np.array(v10[i]).shape[0]):
        temp.append(np.mean(np.array(v10[i])[j, -ssn:-1, :], axis=0))
    cumulative = np.array(temp).flatten()
    n = nosc * np.array(v10[i]).shape[0]
    f = max(np.histogram(cumulative, bins=30, weights=np.ones(n) / n,
                         range=(min(-10, np.min(cumulative)), max(10, np.max(cumulative))))[0])
    fs.append(f)

# while 1:
#     inp = input(f"Index out of {len(noise) - 1}: ")
#     try:
#         num = int(inp)
#         weights, edges, patches = plt.hist(velocities[num], label='1', bins=30, weights=np.ones(nosc) / nosc,
#                                            range=(min(-10, min(velocities[num])), max(10, max(velocities[num]))))
#         plt.xlabel('Velocity')
#         plt.title(f"r={o10[num]:.2f}, f={max(weights):.2f}")
#         plt.show()
#     except ValueError:
#         if inp == 'plot':
#             # fractions = [max(plt.hist(velocities[i], label='1', bins=30, weights=np.ones(nosc) / nosc,
#             #                           range=(min(-10, min(velocities[i])), max(10, max(velocities[i]))))[0])
#             #              for i in range(len(noise))]
#             plt.figure()
#             plt.xlabel('Order parameter')
#             plt.ylabel('Synchronized fraction')
#             plt.plot(o10, fs, 'o-')
#             plt.figure()
#             plt.ylabel('Order parameter')
#             plt.xlabel('Synchronized fraction')
#             plt.plot(fs, o10, 'o-')
#             plt.show()
#         else:
#             break


def fractionapprox(r, sigma, K=1):
    return r+(sigma/(K*r))*(np.exp(-K*r**2/(2*sigma**2)))/(np.sqrt(2*np.pi))

plt.figure()
plt.plot(o10, fs)
plt.show()