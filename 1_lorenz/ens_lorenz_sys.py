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
cond = 'A'
if cond == 'A':
    r = 28
elif cond == 'B':
    r = 9

tf = 15
dt = 0.005
x0 = 9
y0 = 10
z0 = 18

#random perturbation epsilon between:
a = -0.75
b = 0.75
M = 100 #number of ensemble

#containers
N = int(tf/dt +1) #grid point number
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

###################################
###################### ENSEMBLE RUN
#np.random.seed(1000)

x2 = np.zeros(N)
y2 = np.zeros(N)
z2 = np.zeros(N)
y2[0] = y0
z2[0] = z0

mse = np.ndarray((M,N))
ensemble = np.ndarray((M,N))
for j in range(M):
    epsilon = np.random.random()*(b-a) + a #gen between a and b
    x2[0] = x0 + epsilon

    #print(epsilon)
    #print(str(x2[0]) + " " + str(y2[0]) + " " + str(z2[0]))

    for i in range(N-1):
        x2[i+1] = x2[i] + xdot(sigma, x2[i], y2[i])*dt
        y2[i+1] = y2[i] + ydot(r, x2[i], y2[i], z2[i])*dt
        z2[i+1] = z2[i] + zdot(b, x2[i], y2[i], z2[i])*dt

    mse[j] = (x-x2)**2 # x_true - x_k
    ensemble[j] = x2   # save ensemble run

#ensemble average
avg = np.zeros(N) # <x>
for i in range(N): # over time
    for j in range(M): # over members
        avg[i] += ensemble[j][i] / M
#mse ens_avg and true
mse_true_avg = (x - avg)**2

#average mean square error
avg_mse = np.zeros(N)
for i in range(N):
    for j in range(M):
        avg_mse[i] += mse[j][i] / M


fig, ax = plt.subplots(1,2,figsize=(20,8))
#plot all ensemble, true and avg
for j in range(M):
    ax[0].plot(t, ensemble[j], alpha=0.4)
ax[0].plot(t, x, color='black', linewidth=3, label='true')
ax[0].plot(t, avg, color='red', linewidth=2, label='ens avg')

#plot all mse and avg
f = int(4/dt)
for j in range(M):
    ax[1].semilogy(t[:f], mse[j][:f], alpha=0.4)
ax[1].semilogy(t[:f], avg_mse[:f], color='red', linewidth=3, label='avg mse')
ax[1].semilogy(t[:f], mse_true_avg[:f], color = 'black', linewidth=3, label='mse true-avg')

#graph
ax[0].set_title('Ensemble run & ensemble avg & true run')
ax[0].legend()
ax[0].set_xlabel('t')
ax[0].set_ylabel('x(t)')
ax[1].set_title('Mean square error')
ax[1].legend()
ax[1].set_xlabel('t')
ax[1].set_ylabel('$(x_{true} - x_k)^2$')
#save fig
fig.savefig('ens_'+cond+'.jpg',bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
    