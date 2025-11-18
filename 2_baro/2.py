#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:50:18 2023

@author: Paolo
"""
###################################################
##############PARAMETERS###########################
################################################### 
DAYLEN  = 1   #  Forecast length in days.
DtHours = .5   #  Timestep in hours.
###################################################
##############IMPORT LIBRARIES#####################
################################################### 
import numpy as np
import matplotlib.pyplot as plt
###################################################
##############DEFINE FUNCTIONS#####################
###################################################     


def make_Laplacian(Z, const):
    #  Compute the Laplacian 
    #of the geopotential height
    # within the boundary
    #(boundary excluded)
    M       = Z.shape[0]
    N       = Z.shape[1]
    Zxx  = np.zeros([M,N])     #  second x derivative of Z
    Zyy  = np.zeros([M,N])     #  second y derivative of Z
    L0in     = np.zeros([M,N])     #  Laplacian of Z
    # Compute within the domain (no boundaries) 
    # Second x-derivative of Z
    Zxx[1:M-1,:] = (Z[2:M,:]+Z[0:M-2,:]-2*Z[1:M-1,:])/(736e+3**2)
    # Second y-derivative of Z
    Zyy[:,1:N-1] = (Z[:,2:N]+Z[:,0:N-2]-2*Z[:,1:N-1])/(736e+3**2)    
    ##  Laplacian of height (or vorticity)
    L0in[1:M-1,1:N-1] = Zxx[1:M-1,1:N-1]+Zyy[1:M-1,1:N-1]

    if not const:
      for i in range(1,M-1):
        L0in[i,0] = 2*L0in[i,1]-L0in[i,2]
        L0in[i,N-1] = 2*L0in[i,N-2]-L0in[i,N-3]
      for j in range(N):
        L0in[0,j] = 2*L0in[1,j]-L0in[2,j]
        L0in[M-1,j] = 2*L0in[M-2,j]-L0in[M-3,j]
    
    return L0in

def make_Jacobian(Z,ABS_VOR):
    M       = Z.shape[0]
    N       = Z.shape[1]
    Zx    = np.zeros([M,N])     #  x derivative of Z
    Zy    = np.zeros([M,N])     #  y derivative of Z
    ABS_VORx  = np.zeros([M,N])     #  x derivative of ABS_VOR
    ABS_VORy  = np.zeros([M,N])     #  y derivative of ABS_VOR
    # Compute within 
    #the domain (boundary excluded) 
    # x-derivative of Z
    Zx[1:M-1,:] = (Z[2:M,:]-Z[0:M-2,:])/(2*736e+3)
    # y-derivative of Z
    Zy[:,1:N-1] = (Z[:,2:N]-Z[:,0:N-2])/(2*736e+3)
    # x-derivative of the absolute vorticity 
    ABS_VORx[1:M-1,:] = (ABS_VOR[2:M,:]-ABS_VOR[0:M-2,:])/(2*736e+3)
    # y-derivative of the absolute vorticity 
    ABS_VORy[:,1:N-1] = (ABS_VOR[:,2:N]-ABS_VOR[:,0:N-2])/(2*736e+3)
    ##  Compute the Jacobian J(ABS_VOR,Z)
    Jacobi = ABS_VORx * Zy - ABS_VORy * Zx
    return Jacobi

def Poisson_solver(Jacobi, const):
    M       = Jacobi.shape[0]
    N       = Jacobi.shape[1]
    SM=np.zeros([M-2,M-2])
    SN=np.zeros([N-2,N-2])
    EIGEN=np.zeros([M-2,N-2])
    Zdot = np.zeros([M,N])
    ##  Coefficients for x-transformation
    for m1 in range(0,M-2):
     for m2 in range(0,M-2):
      SM[m1,m2] = np.sin(np.pi*(m1+1)*(m2+1)/(M-1))       
    ##  Coefficients for y-transformation
    for n1 in range(0,N-2):
     for n2 in range(0,N-2):
      SN[n1,n2] = np.sin(np.pi*(n1+1)*(n2+1)/(N-1))        
    ##  Eigenvalues of Laplacian operator
    for mm in range(0,M-2):
     for nn in range(0,N-2):
      eigen = (np.sin(np.pi*(mm+1)/(2*(M-1))))**2 +(np.sin(np.pi*(nn+1)/(2*(N-1))))**2
      EIGEN[mm,nn] = (-4/736e+3**2) * eigen
    #  Tendency values in interior.
    Ldot = Jacobi[1:M-1,1:N-1]
    #  Compute the transform of the solution
    LDOT = np.dot(SM,np.dot(Ldot,SN))
    #  Convert transform of d(xi)/dt to transform of d(Z)/dt
    ZDOT = LDOT / EIGEN 
    #  Compute inverse transform to get the height tendency.
    Zdot[1:M-1,1:N-1] = (4/((M-1)*(N-1))) *np.dot(SM,np.dot(ZDOT,SN))

    if not const:
      for i in range(1,M-1):
        Zdot[i,0] = 2*Zdot[i,1]-Zdot[i,2]
        Zdot[i,N-1] = 2*Zdot[i,N-2]-Zdot[i,N-3]
      for j in range(N):
        Zdot[0,j] = 2*Zdot[1,j]-Zdot[2,j]
        Zdot[M-1,j] = 2*Zdot[M-2,j]-Zdot[M-3,j]
    return Zdot
      
 
def make_f_and_h(N,M,Xp,Yp):
    FCOR=np.zeros([M,N])
    h=np.zeros([M,N])    
    a = (4*10**7)/(2*np.pi)      #  Radius of the Earth
    grav = 9.80665           #  Gravitational acceleration
    Omega = 2*np.pi/(24*60*60)  #  Angular velocity of Earth.
    ##  Compute Coriolis Parameter and Map Factor
    ##  and parameter h = g*m**2/f used in the BVE
    for ny in range(0,N):
     for nx in range(0,M):
      xx = (nx-Xp)*736e+3
      yy = (ny-Yp)*736e+3
      rr = np.sqrt(xx**2+yy**2)
      phi = 2*((np.pi/4)-np.arctan(rr/(2*a)))
      mapPS = 2 / (1+np.sin(phi))
      f = 2*Omega*np.sin(phi)
      FCOR[nx,ny] = f
      h[nx,ny] = grav * mapPS**2 / f
    return FCOR,h

###################################################
##############FIXED PARAMETERS###########################
###################################################   
M  = 16 # Points in y direction
N  = 19 # Points in x direction
Xp = 8 # Coord. of North Pole
Yp = 12 # Coord. of North Pole

###################################################
###############COORDINATES AND TIME################
###################################################
daylen = DAYLEN                   #  Integration time (in days)
seclen = int(daylen*24*60*60)     #  Integration time (in secon736e+3)
Dt = DtHours*60*60                #  Timestep in secon736e+3
nt = int(seclen//Dt )             #  Total number of time-steps.
# Define the (X,Y) grid (for plotting)
X, Y  = np.meshgrid(np.linspace(1,M,M),np.linspace(1,N,N))
X = np.transpose(X)
Y = np.transpose(Y)
#Coriolis and map factor
FCOR,h=make_f_and_h(N,M,Xp,Yp)

###################################################
###############DEFINE WORKING ARRAYS###############
###################################################
Zout=np.zeros([nt+1,M,N])
L=np.zeros([nt+1,M,N])
###################################################
###############READ INPUT DATA##########################
###################################################
###   Read and plot the initial and verification height data
## Input files
File1 = 'case1_0503' #The initial value 
File2 = 'case1_0603' #The final value 
Z0  = np.genfromtxt(File1)
Z24 = np.genfromtxt(File2)
Z24 = np.transpose(Z24)
Z0 = np.transpose(Z0)

Zout[0,:,:]  = Z0      #  Copy initial height field
L[0] = make_Laplacian(Zout[0], 0)

nome = "ilbello"
for s in range(Zout.shape[0]-1):
  if s == 0:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot = Poisson_solver(Ldot, s)
    L[s+1] = Ldot*Dt + L[s]
    Zout[s+1] = Zdot*Dt + Zout[s]
  else:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot = Poisson_solver(Ldot, s)
    L[s+1] = Ldot*Dt*2 + L[s-1]
    Zout[s+1] = Zdot*Dt*2 + Zout[s-1]

'''nome = "barbarossa"
for s in range(Zout.shape[0]-1):
  if s == 0:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot = Poisson_solver(Ldot, s)
    Zout[s+1] = Zdot*Dt + Zout[s]
    L[s+1] = make_Laplacian(Zout[s+1], s)
  else:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot = Poisson_solver(Ldot, s)
    Zout[s+1] = Zdot*Dt*2 + Zout[s-1]
    L[s+1] = make_Laplacian(Zout[s+1], s)'''

'''nome = "ficcanaso" #NO integrazione + derivata Sparisce il continuo
for s in range(Zout.shape[0]-1):
  if s == 0:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    L[s+1] = Ldot*Dt + L[s]
    Zout[s+1] = Poisson_solver(L[s+1], s)
  else:
    Ldot = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    L[s+1] = Ldot*Dt*2 + L[s-1]
    Zout[s+1] = Poisson_solver(L[s+1], s)'''

maxZ = np.max(Zout)
minZ = np.min(Zout)

############################################ ANOMALY
anomaly=np.zeros([nt+1,M,N])
anomaly[:] = np.subtract(Zout[:],Zout[0])
maxD = np.max(anomaly)
minD = np.min(anomaly)

############################################ PLOTS
fig, ax = plt.subplots()
contour = ax.contourf(Zout[0], levels=np.linspace(minZ,maxZ,15))
cb = fig.colorbar(contour, ax = ax)

fig2, ax2 = plt.subplots()
tend = ax2.contourf(np.subtract(Zout[0],Zout[0]), levels=np.linspace(minD,maxD,15))
cb2 = fig2.colorbar(tend, ax = ax2)

plt.ion()  # modalità interattiva
plt.show()

# Loop di aggiornamento
for tt in range(Zout.shape[0]):
  contour.remove()
  tend.remove()

  # Disegna il nuovo contour
  contour = ax.contourf(Zout[tt], levels=np.linspace(minZ,maxZ,15))
  cb.update_normal(contour)

  tend = ax2.contourf(np.subtract(Zout[tt],Zout[0]), levels=np.linspace(minD,maxD,15))
  cb2.update_normal(tend)
  # Aggiorna il grafico
  plt.draw()

  fig.savefig(nome + '/Zout_'+str(tt)+'.png',bbox_inches='tight', dpi=300)
  fig2.savefig(nome + '/tend_'+str(tt)+'.png',bbox_inches='tight', dpi=300)
  print(tt)
  plt.pause(0.1)  # attende 1 secondo

ax.contour(Z24, colors="red")
fig.savefig(nome + '/Z24_'+str(tt)+'.png',bbox_inches='tight', dpi=300)
plt.ioff()
plt.show()