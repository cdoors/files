import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

import numpy as np

# np.random.seed(1000)

# y = np.random.standard_normal(20)
# x = np.arange(len(y))
# plt.plot(x, y)
# plt.plot(y)

# plt.plot(y.cumsum())
# plt.grid(False)
# plt.axis('equal')
# plt.xlim(-1,20)
# plt.ylim(np.min(y.cumsum()) -1, np.max(y.cumsum()) + 1)

# plt.figure(figsize = (10,6))
# plt.plot(y.cumsum(), 'bh', lw=1.5)
# plt.plot(y.cumsum(), 'kh')
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('A Simple Plot')
# np.random.seed(1000)
# y = np.random.standard_normal((20,2)).cumsum(axis=0)

# y[:,0] = y[:,0] * 100


# fig, ax1 = plt.subplots()
# plt.figure(figsize=(10,6))
# plt.subplot(211)
# plt.plot(y[:,0], 'b', lw=1.5, label='1st')
# plt.plot(y[:,0], 'ro')
# plt.legend(loc=0)


# plt.figure(figsize=(10,6))
# plt.plot(y, lw=1.5)
# plt.plot(y[:,0], lw=1.5, label='1st')
# plt.plot(y[:,1], lw=1.5, label='2nd')
# plt.plot(y, 'ro')
# plt.legend(loc=0)
# plt.xlabel('index')
# plt.ylabel('# value')
# plt.title('A Simple Plot')

# ax2 = ax1.twinx()
# plt.subplot(212)
# plt.plot(y[:,1],'g', lw=1.5, label='2nd')
# plt.plot(y[:,1],'ro')
# plt.legend(loc=0)
# plt.xlabel('index')
# plt.ylabel('value')

# plt.figure(figsize=(10,6))
# plt.subplot(121)
# plt.plot(y[:,0], lw=1.5, label='1st')
# plt.plot(y[:,0],'ro')
# plt.legend(loc=0)
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('1st data set')
# plt.subplot(122)
# plt.bar(np.arange(len(y)), y[:,1], width = 0.5, color='g', label = '2nd')
# plt.legend(loc = 0)
# plt.xlabel('index')
# plt.title('2nd Data Set')


# y = np.random.standard_normal((1000,2))
# c = np.random.randint(0,10,len(y))
# plt.figure(figsize=(10,6))
# plt.scatter(y[:,0], y[:,1],c=c, cmap='coolwarm',marker='o')
# plt.colorbar()
# plt.plot(y[:,0], y[:,1],'ro')

# plt.hist(y,label=['1st','2nd'], bins=25)

# plt.hist(y, label=['1st','2nd'], color=['b','g'],stacked=True, bins=20, alpha=0.5)
# fig, ax = plt.subplots(figsize=(10,6))
# plt.boxplot(y)
# plt.setp(ax, xticklabels=['1st','2nd'])
# plt.xlabel('data set')
# plt.ylabel('value')
# plt.title('Boxplot')

# plt.legend(loc=0)




# plt.xlabel('value')
# plt.ylabel('freq')
# plt.title('Histogram')

# plt.show()



# plt.show()


# def func(x):
#     return 0.5 * np.exp(x) + 1
# a,b = 0.5, 1.5
# x = np.linspace(0,2)
# y = func(x)
# Ix = np.linspace(a,b)
# Iy = func(Ix)
# verts = [(a,0)] + list(zip(Ix, Iy)) + [(b,0)]

# from matplotlib.patches import Polygon
# fig, ax = plt.subplots(figsize=(10,6))
# plt.plot(x,y,'b', linewidth=2)
# plt.ylim(bottom=0)
# poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
# ax.add_patch(poly)
# plt.text(0.5*(a+b), 1, r'$\int_a^b f(x)\mathrm{d}x$', horizontalalignment='center', fontsize=20)
# plt.figtext(0.9, 0.075, '$x$')
# plt.figtext(0.075,0.9,'$f(x)$')
# ax.set_xticks((a,b))
# ax.set_xticklabels(('$a$','$b$'))
# ax.set_yticks([func(a), func(b)])
# ax.set_yticklabels(('$f(a)$','$f(b)$'))

# plt.show()
import pandas as pd
strike = np.linspace(50,150,24)
# strike = pd.DataFrame(strike)K
# print(strike.info())

ttm = np.linspace(0.5,2.5,24)
strike,ttm = np.meshgrid(strike,ttm)

strike[:2].round(1)
iv = (strike-100)**2 / (100*strike) / ttm
iv[:5,:3]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5,antialiased=True)
ax.set_xlabel('strike', labelpad=10)
ax.set_ylabel('time-to-maturity', labelpad=10)
ax.set_zlabel('implied volatility', labelpad=10)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
