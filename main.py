# This program will find a numerical solution of the Camassa-Holm equation on the circle.

# u_t - u_{txx} + 3uu_x - uu_{xxx} - 2u_x u_{xx} = 0
# or
# u_t + uu_x = -(1-\partial x^2)^{-1} \partial x (u^2 + u_x^2 / 2)

################################################################################

# Imported packages

import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import math 
import scipy as sp
import scipy.fftpack as spf
from scipy.integrate import odeint

#Other imported files
import functions as fcn

#import flow_solverRK4 as flows
from scipy.interpolate import interp2d

import time

starttime = time.time()

################################################################################

# Global Variables to be set by user:

# initial time
tinit = 0.0 #make sure this is a float
# final time
tfin = 5. #make sure this is a float
# time partition size
Mtime = 100
# spatial partition size
N = 100
# Partition of [0,1]
# 0, 1/N, 2/N, ..., N/N
# total (N+1) points

tstep = (tfin-tinit)/float(Mtime)
xstep = 1/float(N)

################################################################################
# Global variables to be computed:

rho = np.zeros((N,Mtime+1))
phi = np.zeros((N,Mtime+1))
eta = np.zeros((N,Mtime+1))
varphi = np.zeros((N,Mtime+1))

G = np.zeros((N,Mtime+1))
F = np.zeros((N,Mtime+1))
H = np.zeros((N,Mtime+1))

psi = np.zeros((Mtime+1,N,N))

umean = np.zeros(Mtime+1)
l2norm = np.zeros(Mtime+1)
uenergy = np.zeros(Mtime+1)
uh1dotenergy = np.zeros(Mtime+1)

etaaux = np.zeros((Mtime+1))

# tinit=0, 1*tstep, 2*tstep, ..., (Mtime-1)*tstep, (Mtime)*tstep=tfin
# total (Mtime + 1) points

################################################################################
# Set up the initial profile
# Here, we need to set up the initial conditions for
# rho0=rho[,0] and phi0(x)=.5*u0'(x)

# 1. rho(x)=1, a constant function
for n in range(N):
    rho[n,0]=1

for n in range(N):
    phi[n,0] = np.cos(2 * np.pi * (n+1) * xstep) - 2 * np.sin(4 * np.pi * (n+1) * xstep)

# 2.

################################################################################
# Conserved quantity
# The mean of the velocity field is a conserved quantity for the CH equation.

mu = 0

################################################################################
# Here, we define necessary auxiliary functions to integrate periodic Camassa-Holm equation numerically.

################################################################################

import numpy as np

def RHS(N,f,g,h):
    v = np.zeros(N)
    for n in range(N):
        v[n] = .5 * f[n] * (g[n]**2 - h[n])
    return v
    
def Functionvarphi(N,f,g,b):
    y = np.zeros(N)
    y[0] = 2 * b * f[0] * g[0]
    for n in range(N-1):
        y[n+1] = y[n] + 2 * b * f[n+1] * g[n+1]    
    return y    
    
def FunctionG(N,mu,f,g,b):
    nu = 0
    for n in range(N):
        nu = nu + b * f[n] * g[n]**2
    z = f + mu - nu
    return z

def Functionpsi(N,f,b):
    w = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            if n>=m:
                for l in range(m,n+1):
                    w[n,m] = w[n,m] + b * f[l]**2
            else:
                for l in range(n+1,m+1):
                    w[n,m] = w[n,m] + b * f[l]**2
    return w
    
def FunctionF(N, f, g, h, p, b):
    t = np.zeros(N)
    for n in range(N):
        for k in range(N):
            t[n] = t[n]+(1/(2*np.sinh(.5))) * b * (np.cosh(p[n,k]-.5))*(f[k]**2 * g[k]**2 + 2 * h[k]**2)
    return t

def FunctionH(N, f, g, h, p, b):
    q = np.zeros(N)
    for n in range(N):
        for k in range(1,N):
            q[n] = q[n]+(1/(2*np.sinh(.5))) * b * (np.sinh(p[n,k]-.5))*(f[k]**2 * g[k]**2 + 2 * h[k]**2)
    return q

def L2normrho(N, f, xstep):
    v=0
    for n in range(N):
        v = v + xstep * f[n]**2
    return v

def H1energy(N, f, g, h, b):
    v=0
    for n in range(N):
        v = v + b * (f[n]**2 * g[n]**2 + 4 * h[n]**2)
    return v

def H1dotenergy(N, h, b):
    v=0
    for n in range(N):
        v = v + b * 4 * h[n]**2
    return v
    
def CHRK2(M, N, mu, xstep, tstep, rho, phi, RHS, F, G):
    xi1 = np.zeros(N)
    xi2 = np.zeros(N)
    
    for m in range(M):
        xi1 = rho[:,m] + tstep * .5 * phi[:,m]
        xi2 = phi[:,m] + tstep * .5 * RHS(N, rho[:,m], G[:,m], F[:,m])
        rho[:,m+1] = rho[:,m] + tstep * xi2
        phi[:,m+1] = phi[:,m] + tstep * RHS(N, xi1, G[:,m], F[:,m])
    
        varphi[:,m+1] = Functionvarphi(N, rho[:,m+1], phi[:,m+1], xstep)
        G[:,m+1] = FunctionG(N, mu, varphi[:,m+1], rho[:,m+1], xstep)
        psi[m+1,:,:] = Functionpsi(N, rho[:,m+1], xstep)
        H[:,m+1] = FunctionH(N, rho[:,m+1], G[:,m+1], phi[:,m+1], psi[m+1,:,:], xstep)
        F[:,m+1] = FunctionF(N, rho[:,m+1], G[:,m+1], phi[:,m+1], psi[m+1,:,:], xstep)
        uenergy[m+1] = H1energy(N, rho[:,m+1], G[:,m+1], phi[:,m+1], xstep)
        uh1dotenergy[m+1] = H1dotenergy(N, phi[:,m+1], xstep)
        l2norm[m+1] = L2normrho(N, rho[:,m+1], xstep)        
    return rho, phi, varphi, G, psi, H, F, uenergy, uh1dotenergy, l2norm


################################################################################
# Set up the initial function values

varphi[:,0] = Functionvarphi(N, rho[:,0], phi[:,0], xstep)
G[:,0] = FunctionG(N, mu, varphi[:,0], rho[:,0], xstep)
psi[0,:,:] = Functionpsi(N, rho[:,0], xstep)
F[:,0] = FunctionF(N, rho[:,0], G[:,0], phi[:,0], psi[0,:,:], xstep)
H[:,0] = FunctionH(N, rho[:,0], G[:,0], phi[:,0], psi[0,:,:], xstep)
uenergy[0] = H1energy(N, rho[:,0], G[:,0], phi[:,0], xstep)
uh1dotenergy[0] = H1dotenergy(N, phi[:,0], xstep)
l2norm[0] = L2normrho(N, rho[:,0], xstep)

################################################################################
# Implement the numerical integration.

rho, phi, varphi, G, psi, H, F, uenergy, uh1dotenergy, l2norm = CHRK2(Mtime, N, mu, xstep, tstep, rho, phi, RHS, F, G)

################################################################################