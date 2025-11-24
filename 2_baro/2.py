"""
@author: Fede, Tommi & Filo
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import os

#Parameters
DAYLEN  = 1  # Forecast length in days
DtHours = 0.5 # Timestep in hours
DX = 736e3   # Spatial grid spacing in meters

M  = 16 # Points in y direction
N  = 19 # Points in x direction
Xp = 8  # Coord. of North Pole
Yp = 12 # Coord. of North Pole

#COORDINATES AND TIME
daylen = DAYLEN                   #  Integration time (in days)
seclen = int(daylen*24*60*60)     #  Integration time (in seconds)
Dt = DtHours*60*60                #  Timestep in seconds
nt = int(seclen//Dt )             #  Total number of time-steps.

#Output directory
img_folder = "output_images"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

#Define functions
def make_Laplacian(Z, timestep):
    #Compute the Laplacian of the geopotential height
    #within the boundary (boundary excluded)
    M       = Z.shape[0]
    N       = Z.shape[1]
    Zxx  = np.zeros([M,N])     #  second x derivative of Z
    Zyy  = np.zeros([M,N])     #  second y derivative of Z
    L0in     = np.zeros([M,N]) #  Laplacian of Z
    #Compute within the domain (no boundaries) 
    #Second x-derivative of Z
    Zxx[1:M-1,:] = (Z[2:M,:]+Z[0:M-2,:]-2*Z[1:M-1,:])/(DX**2)
    #Second y-derivative of Z
    Zyy[:,1:N-1] = (Z[:,2:N]+Z[:,0:N-2]-2*Z[:,1:N-1])/(DX**2)    
    #Laplacian of height (or vorticity)
    L0in[1:M-1,1:N-1] = Zxx[1:M-1,1:N-1]+Zyy[1:M-1,1:N-1]

    if not timestep: #interpolate only for first timestep (=0)
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
    #Compute within the domain (boundary excluded) 
    # x-derivative of Z
    Zx[1:M-1,:] = (Z[2:M,:]-Z[0:M-2,:])/(2*DX)
    # y-derivative of Z
    Zy[:,1:N-1] = (Z[:,2:N]-Z[:,0:N-2])/(2*DX)
    # x-derivative of the absolute vorticity 
    ABS_VORx[1:M-1,:] = (ABS_VOR[2:M,:]-ABS_VOR[0:M-2,:])/(2*DX)
    # y-derivative of the absolute vorticity 
    ABS_VORy[:,1:N-1] = (ABS_VOR[:,2:N]-ABS_VOR[:,0:N-2])/(2*DX)
    ##  Compute the Jacobian J(ABS_VOR,Z)
    Jacobi = ABS_VORx * Zy - ABS_VORy * Zx
    return Jacobi

def Poisson_solver(Jacobi, timestep):
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
      EIGEN[mm,nn] = (-4/DX**2) * eigen
    #  Tendency values in interior.
    Ldot = Jacobi[1:M-1,1:N-1]
    #  Compute the transform of the solution
    LDOT = np.dot(SM,np.dot(Ldot,SN))
    #  Convert transform of d(xi)/dt to transform of d(Z)/dt
    ZDOT = LDOT / EIGEN 
    #  Compute inverse transform to get the height tendency.
    Zdot[1:M-1,1:N-1] = (4/((M-1)*(N-1))) *np.dot(SM,np.dot(ZDOT,SN))

    if not timestep: #interpolate only for first timestep (=0)
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
      xx = (nx-Xp)*DX
      yy = (ny-Yp)*DX
      rr = np.sqrt(xx**2+yy**2)
      phi = 2*((np.pi/4)-np.arctan(rr/(2*a)))
      mapPS = 2 / (1+np.sin(phi))
      f = 2*Omega*np.sin(phi)
      FCOR[nx,ny] = f
      h[nx,ny] = grav * mapPS**2 / f
    return FCOR,h

# Define the (X,Y) grid (for plotting)
X, Y  = np.meshgrid(np.linspace(1,M,M),np.linspace(1,N,N))
X = np.transpose(X)
Y = np.transpose(Y)
#Coriolis and map factor
FCOR,h=make_f_and_h(N,M,Xp,Yp)

#Read input data
#Read and plot the initial and verification height data
File1 = 'case1_0503' #The initial value 
File2 = 'case1_0603' #The final value 
Z0  = np.genfromtxt(File1)
Z0 = np.transpose(Z0)
Z24 = np.genfromtxt(File2)
Z24 = np.transpose(Z24)

#Initial Laplacian
L0 = make_Laplacian(Z0, timestep=0) #with boundary interpolation

def run_model_ilbello(Z0, L0, Dt, nt, method="leapfrog"):
  Zout = np.zeros([nt+1,M,N])
  Zdot = np.zeros([nt+1,M,N])
  L    = np.zeros([nt+1,M,N])
  Ldot = np.zeros([nt+1,M,N])

  #Copy initial height field
  Zout[0]  = Z0
  L[0] = L0

  for s in range(Zout.shape[0]-1):
    Ldot[s] = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot[s] = Poisson_solver(Ldot[s], s)

    if method == "leapfrog":
      if s == 0:
        L[s+1] = Ldot[s]*Dt + L[s]
        Zout[s+1] = Zdot[s]*Dt + Zout[s]
      else:
        L[s+1] = Ldot[s]*Dt*2 + L[s-1]
        Zout[s+1] = Zdot[s]*Dt*2 + Zout[s-1]

    elif method == "AB4":
      if s == 0:
        L[s+1] = Ldot[s]*Dt + L[s]
        Zout[s+1] = Zdot[s]*Dt + Zout[s]
      elif s==1:
        L[s+1] = L[s] + Dt*(3/2*Ldot[s] -0.5*Ldot[s-1])
        Zout[s+1] = Zout[s] + Dt*(3/2*Zdot[s] -0.5*Zdot[s-1])
      elif s==2:
        L[s+1] = L[s] + Dt*(23/12*Ldot[s] -4/3*Ldot[s-1] +5/12*Ldot[s-2])
        Zout[s+1] = Zout[s] + Dt*(23/12*Zdot[s] -4/3*Zdot[s-1] +5/12*Zdot[s-2])
      else:
        L[s+1] = L[s] + Dt*(55/24*Ldot[s] -59/24*Ldot[s-1] +37/24*Ldot[s-2] -9/24*Ldot[s-3])
        Zout[s+1] = Zout[s] + Dt*(55/24*Zdot[s] -59/24*Zdot[s-1] +37/24*Zdot[s-2] -9/24*Zdot[s-3])
    #costant boundary null
    #L[s+1,:,0] = 0 #prima colonna
    #L[s+1,0,:] = 0 #prima riga
    #L[s+1,M-1,:] = 0 #ultima riga
    #L[s+1,:,N-1] = 0 #ultima colonna
  return Zout, L, "ilbello"

def run_model_ilbarbarossa(Z0, L0, Dt, nt, method="leapfrog"):
  Zout = np.zeros([nt+1,M,N])
  Zdot = np.zeros([nt+1,M,N])
  L    = np.zeros([nt+1,M,N])
  Ldot = np.zeros([nt+1,M,N])

  #Copy initial height field
  Zout[0]  = Z0
  L[0] = L0

  for s in range(Zout.shape[0]-1):
    Ldot[s] = make_Jacobian(Zout[s],np.multiply(h,L[s])+FCOR)
    Zdot[s] = Poisson_solver(Ldot[s], s)

    if method == "leapfrog":
      if s == 0:
        Zout[s+1] = Zdot[s]*Dt + Zout[s]
      else:
        Zout[s+1] = Zdot[s]*Dt*2 + Zout[s-1]

    elif method == "AB4":
      if s == 0:
        Zout[s+1] = Zdot[s]*Dt + Zout[s]
      elif s==1:
        Zout[s+1] = Zout[s] + Dt*(3/2*Zdot[s] -0.5*Zdot[s-1])
      elif s==2:
        Zout[s+1] = Zout[s] + Dt*(23/12*Zdot[s] -4/3*Zdot[s-1] +5/12*Zdot[s-2])
      else:
        Zout[s+1] = Zout[s] + Dt*(55/24*Zdot[s] -59/24*Zdot[s-1] +37/24*Zdot[s-2] -9/24*Zdot[s-3])
    L[s+1] = make_Laplacian(Zout[s+1], timestep=s)
    #L boundary costant
    L[s+1,:,0] = L[s,:,0] #prima colonna
    L[s+1,0,:] = L[s,0,:] #prima riga
    L[s+1,M-1,:] = L[s,M-1,:] #ultima riga
    L[s+1,:,N-1] = L[s,:,N-1] #ultima colonna
  return Zout, L, "ilbarbarossa"

#plots functions
def plot_contour(ax, Z, levels, cmap, cb=None, title=""):
  img = ax.contourf(Z, levels=levels, cmap=cmap)
  if cb is not None:
    cb.update_normal(img)
    cb.set_label('[m]')
  ax.plot(Xp, Yp, marker='*', markersize=15, color='orange')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title(title)
  return img

def salva_frame(fig, name, folder, tt):
  fig.savefig(f"{folder}/{name}_{tt}.png", bbox_inches='tight', dpi=300)

def rmse(Z1, Z2):
  return np.sqrt(np.mean((Z1 - Z2)**2))

def mean_error(Z1, Z2):
  return np.mean(Z1 - Z2)

def plot_pointwise_error(Z1, Z2, folder, filename, title="Pointwise Absolute Error"):
  M, N = Z1.shape
  
  error = np.abs(Z1 - Z2)

  fig, ax = plt.subplots(figsize=(8, 6))
  im = ax.imshow(error, origin='lower', interpolation='none', cmap='Reds', extent=[0, N, 0, M], vmin=0)
  
  ax.set_xticks(np.arange(0, N, 1))
  ax.set_yticks(np.arange(0, M, 1))
  ax.set_xticklabels(np.arange(0, N, 1))
  ax.set_yticklabels(np.arange(0, M, 1))

  ax.set_xticks(np.arange(0, N+1, 1), minor=True)
  ax.set_yticks(np.arange(0, M+1, 1), minor=True)
  ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
  ax.tick_params(which='minor', bottom=False, left=False)

  ax.plot(Xp + 0.5, Yp + 0.5, marker='*', markersize=15, color='orange',
          markeredgecolor='k', markeredgewidth=0.5)

  ax.set_aspect('equal')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title(title)

  cb = fig.colorbar(im, ax=ax)
  cb.set_label('Absolute Error [m]')
  
  plt.savefig(f"{folder}/absolute_error_{filename}.png", bbox_inches='tight', dpi=300)
  plt.close(fig)

##############################################  MAIN      ############################################################
#######################################################################################################################
#Run the model
Zout, L, model_type = run_model_ilbarbarossa(Z0, L0, Dt, nt, method="leapfrog")
#Zout, L, model_type = run_model_ilbello(Z0, L0, Dt, nt, method="leapfrog")

#plots Zout[-1]
maxZ = np.max(Zout)
minZ = np.min(Zout)
fig, ax = plt.subplots(figsize=(8, 6))
levelsZ = np.linspace(minZ, maxZ, 15)
contourZ = ax.contourf(Zout[0], levels=levelsZ)
cb = fig.colorbar(contourZ, ax=ax)
plot_contour(ax, Zout[-1], levelsZ, cmap='viridis', cb=cb, title="Geopotential Height (500hPa) at $t=t_0+24h$")
salva_frame(fig, "Zout_"+model_type, img_folder, tt="")
#plots Zout[0]
fig6, ax6 = plt.subplots(figsize=(8, 6))
contourZ0 = ax6.contourf(Zout[0], levels=levelsZ)
cb6 = fig6.colorbar(contourZ0, ax=ax6)
plot_contour(ax6, Zout[0], levelsZ, cmap='viridis', cb=cb6, title="Geopotential Height (500hPa) at $t=t_0$")
salva_frame(fig6, "Z0_"+model_type, img_folder, tt="")

# Add final Z24 contours
z24_contour = ax.contour(Z24, colors="k", levels=levelsZ)
ax.clabel(z24_contour, fontsize=10)
salva_frame(fig, name="Zout_"+model_type, folder=img_folder, tt="Z24")

#Tendency Z[fin] - Z0
tendModel = np.subtract(Zout[:], Z0) #at any time
fig2, ax2 = plt.subplots(figsize=(8, 6))
levelsTmodel = np.linspace(-np.max(np.abs(tendModel)), np.max(np.abs(tendModel)), 15)
contourTmodel = ax2.contourf(tendModel[0], levels=levelsTmodel)
cb2 = fig2.colorbar(contourTmodel, ax=ax2)
plot_contour(ax2, tendModel[-1], levelsTmodel, cmap='seismic', cb=cb2, title="Model tendency $Z(t_0+24h) - Z_0$")
salva_frame(fig2, "tend_model_"+model_type, img_folder, tt="final")

#expected tendency Z24 - Z0
z24_z0 = np.subtract(Z24, Z0)
fig3, ax3 = plt.subplots(figsize=(8, 6))
levelsDiff = np.linspace(-np.max(np.abs(z24_z0)), np.max(np.abs(z24_z0)), 15)
contourDiff = ax3.contourf(z24_z0, levels=levelsDiff)
cb3 = fig3.colorbar(contourDiff, ax=ax3)
plot_contour(ax3, z24_z0, levelsDiff, cmap='seismic', cb=cb3, title='Expected tendency $Z_{24}-Z_0$')
salva_frame(fig3, "expected_tend_Z24-Z0", img_folder, tt="")

#Zout[-1] - Z24, spatial error of the model
zout_z24 = np.subtract(Zout[-1],Z24)
fig4, ax4 = plt.subplots(figsize=(8, 6))
levelsDiff = np.linspace(-np.max(np.abs(zout_z24)), np.max(np.abs(zout_z24)), 15)
contourDiff = ax4.contourf(zout_z24, levels=levelsDiff)
cb4 = fig4.colorbar(contourDiff, ax=ax4)
contourDiff = plot_contour(ax4, zout_z24, levelsDiff, cmap='seismic', cb=cb4, title='$Z(t+24h) - Z_{24}$')
salva_frame(fig4, "spatial_error_Zout-Z24", img_folder, tt="")

print("done "+model_type)

#RMSE
rmse_model = np.zeros([nt+1])
for i in range(nt+1):
  rmse_model[i] = rmse(Zout[i], Z24)
print("RMSE model (Zout[-1] vs Z24):", rmse_model[-1])

rmse_pers = rmse(Z0, Z24)
print("RMSE persistency (Z0 vs Z24):", rmse_pers)

#Mean Error
mean_error_model = mean_error(Zout[-1], Z24)
print("ME model (Zout[-1] vs Z24):", mean_error_model) 

mean_error_pers = mean_error(Z0, Z24)
print("ME persistency (Z0 vs Z24):", mean_error_pers) 

#plot_pointwise_error(Zout[-1], Z24, img_folder, "Zf_Z24_"+model_type, title="Pointwise Error\n$Z(t+24h)$ vs $Z_{24}$")
#plot_pointwise_error(Z0, Z24, img_folder, "Z0_Z24_"+model_type, title="Pointwise Error\n$Z_0$ vs $Z_{24}$")
#plot_pointwise_error(Zout[-1], Z0, img_folder, "Zf_Z0_"+model_type, title="Pointwise Error\n$Z(t+24h)$ vs $Z_{0}$")

# RMSE vs TIME
fig5, ax5 = plt.subplots(figsize=(8, 5))
timestep = np.linspace(0, nt*Dt/3600, nt+1)
ax5.plot(timestep, rmse_model, linewidth=3)
ax5.set_title("Evolution of model RMSE")
ax5.set_xlabel("t [Hr]")
ax5.set_ylabel("RMSE $Z(t) - Z_{24}$")
fig5.savefig(img_folder+"/rmse_time_"+model_type+".png", dpi=300, bbox_inches='tight')

plt.show()