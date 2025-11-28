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

def ydot(r, x, y, z, f):
    return r*x -x*z -y +f

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

tf = 200
dt = 0.005

#random perturbation epsilon between:
a_x = -20; b_x = 20
a_y = -20; b_y = 20
a_z = 0; b_z = 30
M = 50 #number of ensemble

#containers
N = int(tf/dt +1) #grid point number
t = np.linspace(0, tf, N)

#forzante
f = np.zeros(N)
#f = np.sqrt(t)

###################################
###################### ENSEMBLE RUN
np.random.seed(1000)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

ensemble = np.ndarray((M,N))
for j in range(M):
    x[0] = np.random.random()*(b_x - a_x) + a_x
    y[0] = np.random.random()*(b_y - a_y) + a_y
    z[0] = np.random.random()*(b_z - a_z) + a_z
    #print(str(x[0]) + " " + str(y[0]) + " " + str(z[0]))

    for i in range(N-1):
        x[i+1] = x[i] + xdot(sigma, x[i], y[i])*dt
        y[i+1] = y[i] + ydot(r, x[i], y[i], z[i], f[i])*dt
        z[i+1] = z[i] + zdot(b, x[i], y[i], z[i])*dt
    ensemble[j] = x   # save ensemble run

#ensemble average
n_sub_interval = int(tf / (5000*dt))
n_neg = np.zeros((M,n_sub_interval))
for j in range(M):                          #for every ensemble
    for tau in range(n_sub_interval):    #for number of window
        i_low = 5000*tau                      #lower limit
        i_high = 5000*(tau + 1)               #higher limit
        for i in range(i_low, i_high, 1):   #in 5000 window step
            if ensemble[j][i] < 0:
                n_neg[j][tau] += 1
p_k = n_neg/5000

#avg probability over tau for each ensemble member
p_k_avg = np.zeros(M)
for j in range(M):
    for tau in range(n_sub_interval):
        p_k_avg[j] += p_k[j][tau] / n_sub_interval

#avg over ensemble member
p_k_avg_tau = np.zeros(n_sub_interval)
for tau in range(n_sub_interval):
    for j in range(M):
        p_k_avg_tau[tau] += p_k[j][tau] / M

fig, ax = plt.subplots(1,2,figsize=(20,8))
x_axes = np.linspace(0,n_sub_interval-1,n_sub_interval)
for j in range(M):
    ax[0].plot(x_axes, p_k[j], alpha=0.4)
    ax[0].plot([0,7],[p_k_avg[j],p_k_avg[j]], color='tab:blue')
ax[0].plot(x_axes, p_k_avg_tau, linewidth = 5, color='black', label="avg(tau)")
ax[0].set_ylabel("prob (x<0)")
ax[0].set_xlabel("tau (5000 time step)")
ax[0].set_title("Lorenz sys with no forcing")
ax[0].legend()

######################################################################
#forzante
#f = np.zeros(N)
f = np.sqrt(t)

###################################
###################### ENSEMBLE RUN
np.random.seed(1000)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

ensemble = np.ndarray((M,N))
for j in range(M):
    x[0] = np.random.random()*(b_x - a_x) + a_x
    y[0] = np.random.random()*(b_y - a_y) + a_y
    z[0] = np.random.random()*(b_z - a_z) + a_z
    #print(str(x[0]) + " " + str(y[0]) + " " + str(z[0]))

    for i in range(N-1):
        x[i+1] = x[i] + xdot(sigma, x[i], y[i])*dt
        y[i+1] = y[i] + ydot(r, x[i], y[i], z[i], f[i])*dt
        z[i+1] = z[i] + zdot(b, x[i], y[i], z[i])*dt
    ensemble[j] = x   # save ensemble run

#ensemble average
n_sub_interval = int(tf / (5000*dt))
n_neg = np.zeros((M,n_sub_interval))
for j in range(M):                          #for every ensemble
    for tau in range(n_sub_interval):    #for number of window
        i_low = 5000*tau                      #lower limit
        i_high = 5000*(tau + 1)               #higher limit
        for i in range(i_low, i_high, 1):   #in 5000 window step
            if ensemble[j][i] < 0:
                n_neg[j][tau] += 1
p_k = n_neg/5000

#avg probability over tau for each ensemble member
p_k_avg = np.zeros(M)
for j in range(M):
    for tau in range(n_sub_interval):
        p_k_avg[j] += p_k[j][tau] / n_sub_interval

#avg over ensemble member
p_k_avg_tau = np.zeros(n_sub_interval)
for tau in range(n_sub_interval):
    for j in range(M):
        p_k_avg_tau[tau] += p_k[j][tau] / M

x_axes = np.linspace(0,n_sub_interval-1,n_sub_interval)
for j in range(M):
    ax[1].plot(x_axes, p_k[j], alpha=0.4)
    ax[1].plot([0,7],[p_k_avg[j],p_k_avg[j]], color='tab:blue')
ax[1].plot(x_axes, p_k_avg_tau, linewidth = 5, color='black', label="avg(tau)")
ax[1].set_ylabel("prob (x<0)")
ax[1].set_xlabel("tau (5000 time step)")
ax[1].set_title("Lorenz sys with forcing on y dot $+\sqrt{t}$")
ax[1].legend()
fig.savefig('ens_forcing_'+cond+'.jpg',bbox_inches='tight', dpi=300)

fig2 , ax2 = plt.subplots(figsize=(10,5))
ax2.set_ylabel("$\sqrt{t}$")
ax2.set_xlabel("t")
ax2.set_title("Forcing on y dot")
ax2.plot(t,f)
fig2.savefig('forcing_'+cond+'.jpg',bbox_inches='tight', dpi=300)

plt.show()
    