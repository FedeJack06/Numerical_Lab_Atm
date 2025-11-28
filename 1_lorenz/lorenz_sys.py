import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20              # dimensione testo di default (titoli, etichette, legende)
plt.rcParams['axes.titlesize'] = 20         # dimensione del titolo degli assi
plt.rcParams['axes.labelsize'] = 20         # dimensione delle etichette degli assi
plt.rcParams['xtick.labelsize'] = 20        # dimensione dei tick sull’asse x
plt.rcParams['ytick.labelsize'] = 20        # dimensione dei tick sull’asse y
plt.rcParams['legend.fontsize'] = 20        # dimensione del testo nella legenda
plt.rcParams['figure.titlesize'] = 20       # dimensione del titolo della figura (fig.suptitle)

#derivative
def xdot(sigma, x, y):
    return sigma*(y-x)

def ydot(r, x, y, z):
    return r*x -x*z -y

def zdot(b, x, y, z):
    return(x*y -b*z)

#parameters
sigma = 10
b = 8./3
cond = 'B'
if cond == 'A':
    r = 28
elif cond == 'B':
    r = 9
epsilon = 1e-7

tf = 60
dt = 0.005
x0 = 9
y0 = 10
z0 = 18

#containers
N = int(tf/dt +1)
t = np.linspace(0, tf, N)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
#set initial cond
x[0] = x0
y[0] = y0
z[0] = z0

for i in range(N-1):
    x[i+1] = x[i] + xdot(sigma, x[i], y[i])*dt
    y[i+1] = y[i] + ydot(r, x[i], y[i], z[i])*dt
    z[i+1] = z[i] + zdot(b, x[i], y[i], z[i])*dt

###################### PERTURBATION
x2 = np.zeros(N)
y2 = np.zeros(N)
z2 = np.zeros(N)
x2[0] = x0 + epsilon
y2[0] = y0
z2[0] = z0

for i in range(N-1):
    x2[i+1] = x2[i] + xdot(sigma, x2[i], y2[i])*dt
    y2[i+1] = y2[i] + ydot(r, x2[i], y2[i], z2[i])*dt
    z2[i+1] = z2[i] + zdot(b, x2[i], y2[i], z2[i])*dt

##################### RMSE
rmse = np.sqrt((x-x2)**2)

########PLOTS
fig,ax=plt.subplots(1,2,figsize=(20,7))
#true
ax[0].set_title('Lorenz ($\sigma,b,r$)=('+str(sigma)+',8/3,'+str(r)+')')
ax[0].plot(x,z,label='L(9,10,18)')
ax[0].legend()
ax[0].set_xlabel('x(t)')
ax[0].set_ylabel('z(t)')
#perturbation
ax[1].set_title('Lorenz ($\sigma,b,r$)=('+str(sigma)+',8/3,'+str(r)+'), $\epsilon=1\cdot10^{-7}$')
ax[1].plot(x2,z2,'r',label='L(9+$\epsilon$,10,18)')
ax[1].legend()
ax[1].set_xlabel('x(t)')
ax[1].set_ylabel('z(t)')

fig2, ax2 = plt.subplots(figsize=(10,7))
ax2.set_title('Lorenz ($\sigma,b,r$)=('+str(sigma)+',8/3,'+str(r)+'))')
ax2.plot(t,x,label='L(9,10,18)')
ax2.plot(t,x2,'r',label='L(9+$\epsilon$,10,18)')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('x(t)')
fig2.savefig('pert_x(t)_'+cond+'.jpg',bbox_inches='tight', dpi=300)

fig3, ax3 = plt.subplots(1,2,figsize=(20,7))
ax3[0].set_title('root-mean-square error')
ax3[0].plot(t, rmse)
ax3[0].set_ylabel('$\sqrt{(x_{true} - x_\epsilon)^2}$')
ax3[0].set_xlabel('t')
ax3[1].set_title('semilog root-mean-square error')
ax3[1].semilogy(t, rmse, 'r')
ax3[1].set_ylabel('$\sqrt{(x_{true} - x_\epsilon)^2}$')
ax3[1].set_xlabel('t')
fig3.savefig('rmse_'+cond+'.jpg',bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

fig1,ax1=plt.subplots(figsize=(10,7))
ax1.set_title('Lorenz ($\sigma,b,r$)=('+str(sigma)+',8/3,'+str(r)+')')
ax1.plot(x,z,label='L(9,10,18)')#, color='r')
ax1.legend()
ax1.set_xlabel('x(t)')
ax1.set_ylabel('z(t)')
fig1.savefig('xz_'+cond+'.jpg',bbox_inches='tight', dpi=300)
