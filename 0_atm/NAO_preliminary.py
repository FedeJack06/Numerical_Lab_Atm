#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:44:16 2021

@author: Paolo
"""

###################################################
##############IMPORT LIBRARIES#####################
################################################### 
import numpy as np
import matplotlib.pyplot as plt
###################################################
##############HELLO WORLD##########################
################################################### 
print('Hello world')

###################################################
##############DEFINE FUNCTIONS#####################
################################################### 
def friction(x,c):
    xdot = -c*(x)
    return xdot

###################################################
##############PARAMETERS###########################
################################################### 
num_steps=50#number of time steps 
c=3.0#friction parameter 
x0=50.0#initial condition 
dt=0.05#time step  (units of c^-1)

###################################################
##############INITIALISE ARRAYS####################
###################################################
xEU=np.zeros(num_steps+1)#numerical solution with Euler forward method
xLF=np.zeros(num_steps+1)#numerical solution with Leapfrog method
xLF[0]=x0#Assign initial condition to first element of the array
xEU[0]=x0
t=np.linspace(0,num_steps,num_steps+1)*dt#time variable 
steps=np.linspace(0,num_steps,num_steps+1)*dt#time variable 
xAN=x0*np.exp(-t*c)#analytical solution computed at each finite time increment 
eps=0.5*x0*t*dt*c*c
###################################################
##############EULER FORWARD SOLUTION###############
###################################################
for i in range(num_steps):
    x_dot = friction(xEU[i],c)#compute derivative at time i 
    xEU[i + 1] = xEU[i] + (x_dot * dt)#compute solution at time i + 1
      
###################################################
##############LEAPFROG SOLUTION####################
###################################################    
for i in range(num_steps):
    x_dot = friction(xLF[i],c)#compute derivative at time i 
    if i ==0:
        xLF[i + 1] = xLF[i] + (x_dot * dt)#first step is Euler forward       
    else:
        xLF[i + 1] = xLF[i - 1] + (x_dot * dt*2)#compute solution at time i + 1


fig,(ax,bx,cx)=plt.subplots(3,1,figsize=(6,6))
ax.plot(t,xEU,'k',marker='+')
ax.plot(t,xLF,'r',linewidth=1)
ax.plot(t[0::1],xAN[0::1],color='g',marker='.',linewidth=0,label='Analytical (plotted every 3rd time step')
ax.set_ylim([-70,70])
ax.legend(loc=3)
# costA=-dt*c+np.sqrt(1+dt*c*dt*c)
# costB=-dt*c-np.sqrt(1+dt*c*dt*c)
# ax.plot(t[0::1],x0*(costB)**np.linspace(0,num_steps,num_steps+1))

bx.plot(np.sqrt((xEU-xAN)**2),'k',label='Euler forward')
bx.plot(np.sqrt((xLF-xAN)**2),'r',label='Leapfrog')
bx.set_ylim([0,1.5])
bx.legend()
cx.plot(np.log10(np.sqrt((xEU-xAN)**2)[1:]),color='k')
cx.plot(np.log10(np.sqrt((xLF-xAN)**2)[1:]),color='r')
ax.set_title('a) Solutions of friction equation')
ax.set_ylabel('x(t)')
ax.set_xlabel('t')

bx.set_title('b) Root mean square error')
bx.set_ylabel('$\sqrt{(x(t)-x_{exact}(t))^2}$')
bx.set_xlabel('time steps')

cx.set_title('c) Logarithm of root mean square error')
cx.set_ylabel('$\log_{10}{\sqrt{(x(t)-x_{exact}(t))^2}}$')
cx.set_xlabel('time steps')

ax.grid()
bx.grid()
cx.grid()
plt.tight_layout()
fig.savefig('Time_stepping.jpg',bbox_inches='tight')
plt.tight_layout()
plt.show()


#Truncation error
num_steps=2
xEU=np.zeros(num_steps+1)#numerical solution with Euler forward method
xLF=np.zeros(num_steps+1)#numerical solution with Leapfrog method
xLF[0]=x0#Assign initial condition to first element of the array
xEU[0]=x0
m=0
xoutEU=np.zeros(30)
xoutLF=np.zeros(30)
dts=np.linspace(0,0.1,30)
for dt in dts:
    t=np.linspace(0,num_steps,num_steps+1)*dt#time variable 
    xAN=x0*np.exp(-t*c)
    for i in range(num_steps):
        x_dot = friction(xEU[i],c)#compute derivative at time i 
        xEU[i + 1] = xEU[i] + (x_dot * dt)#compute solution at time i + 1
    if i ==0:
        xLF[i + 1] = xLF[i] + (x_dot * dt)#first step is Euler forward       
    else:
        xLF[i + 1] = xLF[i - 1] + (x_dot * dt*2)#compute solution at time i + 1
    xoutEU[m]=abs(xEU[i + 1]-xAN[i + 1])
    xoutLF[m]=abs(xLF[i + 1]-xAN[i + 1])
    m=m+1
fig,ax=plt.subplots()
ax.plot(dts,xoutEU,color='black',label='Euler forward')
ax.plot(dts,xoutLF,color='red',label='Leapfrog')    
ax.legend()
ax.set_xlabel('$\Delta$ t')
ax.set_ylabel('RMSE')
ax.grid()
ax.set_title('Truncation error with n=2 ')
fig.savefig('Truncation_error.png',dpi=400)
