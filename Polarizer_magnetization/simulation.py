#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simulate the field of a magnetized cylinder (longitudinal, transveral
or both). Used to simulate the surrounding field of the polarizer foils and
compare the theory with real life measurements.
Theory: Programmed by Jeffery Martin
        Mathematical theory: A. Caciagli et al, "Exact expression for the 
        magnetic field of a finite cylinder with arbitrary uniform 
        magnetization", Journal of Magnetism and Magnetic Materials, 2018
Implementation of misalignments and real dara: Fabian Piermaier

Transformation of the theoretical (x_th,y_th,z_th) and our measurement
coordinate system (x_m,y_m,z_m):
    Parallel     |   Transverse
    x_m = y_th   |   x_m = x_th
    y_m = z_th   |   y_m = z_th
    z_m = x_th   |   z_m = y_th
Signs may wary, depending on the direction of alignment during measurement
"""

from mpmath import *
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy.stats
plt.rcParams.update({'font.size': 20})

# %%
#---DEFINE FUNCTIONS---#

mu0 = 1.25663706212*10**(-6)

def squiggle_p(z):
    return z+l

def squiggle_m(z):
    return z-l

def alpha_p(rho,z):
    return 1./sqrt(squiggle_p(z)**2+(rho+r)**2)

def alpha_m(rho,z):
    return 1./sqrt(squiggle_m(z)**2+(rho+r)**2)

def beta_p(rho,z):
    return squiggle_p(z)*alpha_p(rho,z)

def beta_m(rho,z):
    return squiggle_m(z)*alpha_m(rho,z)

def gamma(rho):
    return (rho-r)/(rho+r)

def k_p_2(rho,z):
    return (squiggle_p(z)**2+(rho-r)**2)/(squiggle_p(z)**2+(rho+r)**2)

def k_m_2(rho,z):
    return (squiggle_m(z)**2+(rho-r)**2)/(squiggle_m(z)**2+(rho+r)**2)

def k_p(rho,z):
    return sqrt(k_p_2(rho,z))

def k_m(rho,z):
    return sqrt(k_m_2(rho,z))

def curlyk(k):
    #return ellipk(sqrt(1-k**2))
    #See difference in definition between Eq. (6) of paper and sympy doc.
    #https://docs.sympy.org/0.7.1/modules/mpmath/functions/elliptic.html#ellipk
    return ellipk(1-k**2)

def curlye(k):
    #return ellipe(sqrt(1-k**2))
    return ellipe(1-k**2)

def curlyp(gamma,k):
    #return ellippi(1-gamma**2,sqrt(1-k**2))
    return ellippi(1-gamma**2,1-k**2)

def p1(k):
    return curlyk(k)-2*(curlyk(k)-curlye(k))/(1-k**2)

def p2(k,rho):
    return -(gamma(rho)/(1-gamma(rho)**2))*(curlyp(gamma(rho),k)-curlyk(k))-(1/(1-gamma(rho)**2))*(gamma(rho)**2*curlyp(gamma(rho),k)-curlyk(k))

def p3(k,rho):
    return (curlyk(k)-curlye(k))/(1-k**2)-(gamma(rho)**2/(1-gamma(rho)**2))*(curlyp(gamma(rho),k)-curlyk(k))

def p4(k,rho):
    return (gamma(rho)/(1-gamma(rho)**2))*(curlyp(gamma(rho),k)-curlyk(k))+(gamma(rho)/(1-gamma(rho)**2))*(gamma(rho)**2*curlyp(gamma(rho),k)-curlyk(k))-p1(k)


# Transversal magnetization
def phi_t(rho,phi,z):
    return (m*r*cos(phi)/pi)*(beta_p(rho,z)*p3(k_p(rho,z),rho)-beta_m(rho,z)*p3(k_m(rho,z),rho))

def hrho_t(rho,phi,z):
    return (r*cos(phi)/(2*pi*rho))*(beta_p(rho,z)*p4(k_p(rho,z),rho)-beta_m(rho,z)*p4(k_m(rho,z),rho))

def hphi_t(rho,phi,z):
    return (r*sin(phi)/(pi*rho))*(beta_p(rho,z)*p3(k_p(rho,z),rho)-beta_m(rho,z)*p3(k_m(rho,z),rho))

def hz_t(rho,phi,z):
    return (r*cos(phi)/(pi))*(alpha_p(rho,z)*p1(k_p(rho,z))-alpha_m(rho,z)*p1(k_m(rho,z)))

def hx_cart_t(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*(hrho_t(rho,phi,z)*cos(phi)-hphi_t(rho,phi,z)*sin(phi))

def hy_cart_t(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*(hrho_t(rho,phi,z)*sin(phi)+hphi_t(rho,phi,z)*cos(phi))

def hz_cart_t(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*hz_t(rho,phi,z)

# Longitudinal magnetization
def hrho_l(rho,phi,z):
    return (r/pi)*(alpha_p(rho,z)*p1(k_p(rho,z))-alpha_m(rho,z)*p1(k_m(rho,z)))

def hz_l(rho,phi,z):
    return (r/(pi*(rho+r)))*(beta_p(rho,z)*p2(k_p(rho,z),rho)-beta_m(rho,z)*p2(k_m(rho,z),rho))

def hx_cart_l(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*hrho_l(rho,phi,z)*sin(phi)

def hy_cart_l(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*hrho_l(rho,phi,z)*cos(phi)

def hz_cart_l(x,y,z,m):
    rho=sqrt(x**2+y**2)
    phi=atan2(y,x)
    return m*hz_l(rho,phi,z)

def conf(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    conf = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return conf


# %%
#---THEORY---#

# Define cylinder
r=.15/2      # radius of cylinder (m)
l=400.e-9/2     # half-height of cylinder (m)
h=.03         # height of scan above plane (m)
Deltax=.2413    # range of scan (m)
dx=.001         # step size for scan (m)

f = 0           # Scaling factor between long. and trans. magnetization
g = 1-f

#---PARALLEL SCAN---#

x = np.arange(-Deltax,Deltax,dx)
xnonzero = np.array([xp for xp in x if abs(xp)>1.e-10])
theta = 0*(np.pi/180)   # Angle between magnetization axis and traverse axis
tau = 0*(np.pi/180)     # Angle between cylinder surface and traverse axis
y = 0+xnonzero*np.arctan(theta)
z = h+xnonzero*np.arctan(tau)
m = 1

# Transversal magnetization
hxes_pt = np.array([])
hyes_pt = np.array([])
hzes_pt = np.array([])
for i in range(len(xnonzero)):
    yp = y[i]
    xp = xnonzero[i]
    zp = z[i]
    hzes_pt = np.append(hzes_pt, float(g*hx_cart_t(xp,yp,zp,m)))
    hxes_pt = np.append(hxes_pt, float(g*hy_cart_t(xp,yp,zp,m)))
    hyes_pt = np.append(hyes_pt, float(g*hz_cart_t(xp,yp,zp,m)))

# Longitudinal magnetization
hxes_pl = np.array([])
hyes_pl = np.array([])
hzes_pl = np.array([])
for i in range(len(xnonzero)):
    yp = y[i]
    xp = xnonzero[i]
    zp = z[i]
    hzes_pl = np.append(hzes_pl, float(f*hx_cart_l(xp,yp,zp,m)))
    hxes_pl = np.append(hxes_pl, float(f*hy_cart_l(xp,yp,zp,m)))
    hyes_pl = np.append(hyes_pl, float(f*hz_cart_l(xp,yp,zp,m)))

# Combined magnetization
xcom_p = hxes_pt+hxes_pl
ycom_p = hyes_pt+hyes_pl
zcom_p = hzes_pt+hzes_pl

#---TRANSVERSAL SCAN---#

y = np.arange(-Deltax,Deltax,dx)
ynonzero = np.array([yp for yp in y if abs(xp)>1.e-10])
theta = 0*(np.pi/180)
tau = 0*(np.pi/180)
x = 0+ynonzero*np.arctan(theta)
z = h+ynonzero*np.arctan(tau)
m = 1

# Transveral magnetization
hxes_tt = np.array([])
hyes_tt = np.array([])
hzes_tt = np.array([])
for i in range(len(ynonzero)):
    yp = ynonzero[i]
    xp = x[i]
    zp = z[i]
    hxes_tt = np.append(hxes_tt, float(g*hx_cart_t(xp,yp,zp,m)))
    hzes_tt = np.append(hzes_tt, float(g*hy_cart_t(xp,yp,zp,m)))
    hyes_tt = np.append(hyes_tt, float(g*hz_cart_t(xp,yp,zp,m)))

# Longitudinal magnetization
hxes_tl = np.array([])
hyes_tl = np.array([])
hzes_tl = np.array([])
for i in range(len(ynonzero)):
    yp = ynonzero[i]
    xp = x[i]
    zp = z[i]
    hxes_tl = np.append(hxes_tl, float(f*hx_cart_l(xp,yp,zp,m)))
    hzes_tl = np.append(hzes_tl, float(f*hy_cart_l(xp,yp,zp,m)))
    hyes_tl = np.append(hyes_tl, float(f*hz_cart_l(xp,yp,zp,m)))
    i += 1

# Combined magnetization
xcom_t = hxes_tt+hxes_tl
ycom_t = hyes_tt+hyes_tl
zcom_t = hzes_tt+hzes_tl

fig, ax0 = plt.subplots(1,2, sharey=True)
ax0[0].plot(xnonzero,xcom_p*10**(6),'b',label=r'$B_{th,x}$')
ax0[0].plot(xnonzero,ycom_p*10**(6),'r',label=r'$B_{th,y}$')
ax0[0].plot(xnonzero,zcom_p*10**(6),'g',label=r'$B_{th,z}$')
ax0[0].set_xlabel('$z\;in\;m$')
ax0[0].set_ylabel('$B_{th}\;in\;\mu T$')
ax0[0].set_ylim([-2,2.5])
ax0[0].set_title('Parallel')
ax0[0].axvline(x=-r, color='k', linewidth=1)
ax0[0].axvline(x=r, color='k', linewidth=1)
ax0[1].plot(ynonzero,xcom_t*10**(6),'b',label=r'$B_{th,x}$')
ax0[1].plot(ynonzero,ycom_t*10**(6),'r',label=r'$B_{th,y}$')
ax0[1].plot(ynonzero,zcom_t*10**(6),'g',label=r'$B_{th,z}$')
ax0[1].axvline(x=-r, color='k', linewidth=1)
ax0[1].axvline(x=r, color='k', linewidth=1)
ax0[1].set_xlabel('$z\;in\;m$')
ax0[1].set_title('Transverse')
fig.legend([r'$B_x$',r'$B_y$',r'$B_z$'], 
           labels=[r'$B_x$',r'$B_y$',r'$B_z$'],
           loc="upper center",
           borderaxespad=0, ncol=3)

# %% THORSTEN
#---IMPORT DATA---#

# Molvatec FOIL4
df4_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1141_FOIL4_after_170A_background_avg.csv')
df4_p = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1101_FOIL4_after_170A_parallel_avg_corr.csv')
df4_t = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1126_FOIL4_after_170A_transverse_avg.csv')
B4_p = (df4_p[['Bx','By','Bz']] - df4_bg[['Bx','By','Bz']])*10
B4_t = (df4_t[['Bx','By','Bz']] - df4_bg[['Bx','By','Bz']])*10

#---DEFINE CYLINDER---#

r=.14115/2      # radius of cylinder (m)
l=200.e-9/2     # half-height of cylinder (m)
h=.0279         # height of scan above plane (m)
Deltax=.2413    # range of scan (m)
dx=.005         # step size for scan (m)

rh = .085/2      # Inner radius of the foilholder

# %%
#---COMPARISON---#

m_th = 0.5

#---PARALLEL---#  

zx_off_p = -0.02
off_p = -0.05
theta_p = 0*(np.pi/180)+np.arctan(1/740)
tau_p = 0*(np.pi/180)+np.arctan(.15/141.15)+np.arctan(12/740)   # Measured
zx_p = (np.array(df4_bg.z+14.5)*0.0254+zx_off_p)*np.cos(tau_p)
f = 0.00
g = 1-f

x = off_p+np.array(-zx_p)*np.arctan(theta_p)
y = h+np.array(-zx_p)*np.arctan(tau_p)

# Transverse magnetization
hxes_pt = np.array([])
hyes_pt = np.array([])
hzes_pt = np.array([])
for i in range(len(zx_p)):
    yp = y[i]
    xp = x[i]
    zp = -zx_p[i]
    hzes_pt = np.append(hzes_pt,float(g*hx_cart_t(zp,xp,yp,m_th)))
    hxes_pt = np.append(hxes_pt,float(g*hy_cart_t(zp,xp,yp,m_th)))
    hyes_pt = np.append(hyes_pt,float(g*hz_cart_t(zp,xp,yp,m_th)))

# Longitudinal magnetization
hxes_pl = np.array([])
hyes_pl = np.array([])
hzes_pl = np.array([])
for i in range(len(zx_p)):
    yp = y[i]
    xp = x[i]
    zp = -zx_p[i]
    hzes_pl = np.append(hzes_pl, float(f*hx_cart_l(zp,xp,yp,m_th)))
    hxes_pl = np.append(hxes_pl, float(f*hy_cart_l(zp,xp,yp,m_th)))
    hyes_pl = np.append(hyes_pl, float(f*hz_cart_l(zp,xp,yp,m_th)))

# Combined magnetization
xcom_p = hxes_pt+hxes_pl
ycom_p = hyes_pt+hyes_pl
zcom_p = hzes_pt+hzes_pl

hx_conf_p = conf(np.sqrt((B4_p['Bx']-xcom_p*10e6)**2), 0.10)
hy_conf_p = conf(np.sqrt((B4_p['By']-ycom_p*10e6)**2), 0.10)
hz_conf_p = conf(np.sqrt((B4_p['Bz']-zcom_p*10e6)**2), 0.10)

fig, ax0 = plt.subplots(1,2)
ax0[0].plot(zx_p,xcom_p,'b',linestyle='dashed',label=r'$H_{x,th}\;in\;T$')
ax0[0].plot(zx_p,ycom_p,'r',linestyle='dashed',label=r'$H_{y,th}\;in\;T$')
ax0[0].plot(zx_p,zcom_p,'g',linestyle='dashed',label=r'$H_{z,th}\;in\;T$')
ax0[0].axvline(x=-r, color='k', linewidth=0.5)
ax0[0].axvline(x=r, color='k', linewidth=0.5)
ax0[0].axvline(x=-rh, color='k', linewidth=0.5)
ax0[0].axvline(x=rh, color='k', linewidth=0.5)
ax1 = ax0[0].twinx()
ax1.plot(zx_p,B4_p.Bx-hx_conf_p,'b', alpha=0.4)
ax1.plot(zx_p,B4_p.By-hy_conf_p,'r', alpha=0.4)
ax1.plot(zx_p,B4_p.Bz-hz_conf_p,'g', alpha=0.4)
ax1.plot(zx_p,B4_p.Bx+hx_conf_p,'b', alpha=0.4)
ax1.plot(zx_p,B4_p.By+hy_conf_p,'r', alpha=0.4)
ax1.plot(zx_p,B4_p.Bz+hz_conf_p,'g', alpha=0.4)
ax1.plot(zx_p, B4_p.Bx, 'b.', label=r'$B_{x}$')
ax1.plot(zx_p, B4_p.By, 'r^', label=r'$B_{y}$')
ax1.plot(zx_p, B4_p.Bz, 'gs', label=r'$B_{z}$')
ax0[0].legend(loc='upper left')
ax0[0].set_xlabel('$z$ (m)')
ax0[0].set_ylabel('$H_{th}\;in\;T$')
ax1.set_ylabel(r'$B_{measured}\;in\;\mu T$')
plt.legend()
ax0[0].ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
ax1.set_ylim([-2, 2])
ax0[0].set_ylim([-2e-6, 2e-6])

#---TRANSVERSE---#

zx_off_t = -0.02
off_t = -0.001
theta_t = np.arctan(1/740)+10*(np.pi/180)
tau_t = np.arctan(.30/141.15)+np.arctan(12/740)+1*(np.pi/180)  # Measured

zx_t = (np.array(df4_bg.z+14.5)*0.0254+zx_off_t)
x = off_t+np.array(-zx_t)*np.arctan(theta_t)
y = h+np.array(-zx_t)*np.arctan(tau_t)+0.00

# Transverse magnetization
hxes_tt = np.array([])
hyes_tt = np.array([])
hzes_tt = np.array([])
for i in range(len(zx_t)):
    yp = y[i]
    xp = x[i]
    zp = -zx_t[i]
    hxes_tt = np.append(hxes_tt, float(g*hx_cart_t(xp,zp,yp,m_th)))
    hzes_tt = np.append(hzes_tt, float(g*hy_cart_t(xp,zp,yp,m_th)))
    hyes_tt = np.append(hyes_tt, float(g*hz_cart_t(xp,zp,yp,m_th)))

# Longitudinal magnetization
hxes_tl = np.array([])
hyes_tl = np.array([])
hzes_tl = np.array([])
for i in range(len(zx_t)):
    yp = y[i]
    xp = x[i]
    zp = -zx_t[i]
    hxes_tl = np.append(hxes_tl, float(f*hx_cart_l(xp,zp,yp,m_th)))
    hzes_tl = np.append(hzes_tl, float(f*hy_cart_l(xp,zp,yp,m_th)))
    hyes_tl = np.append(hyes_tl, float(f*hz_cart_l(xp,zp,yp,m_th)))

# Combined magnetization
xcom_t = hxes_tt+hxes_tl
ycom_t = hyes_tt+hyes_tl
zcom_t = hzes_tt+hzes_tl

hx_conf_t = conf(B4_t['Bx']-xcom_t, 0.90)
hy_conf_t = conf(B4_t['By']-ycom_t, 0.90)
hz_conf_t = conf(B4_t['Bz']-zcom_t, 0.90)

ax0[1].plot(zx_t,xcom_t,'b',label=r'$Hx\;in\;T$')
ax0[1].plot(zx_t,ycom_t,'r',label=r'$Hy\;in\;T$')
ax0[1].plot(zx_t,zcom_t,'g',label=r'$Hz\;in\;T$')
ax0[1].plot(zx_t,xcom_t-hx_conf_t,'b', alpha=0.4)
ax0[1].plot(zx_t,ycom_t-hy_conf_t,'r', alpha=0.4)
ax0[1].plot(zx_t,zcom_t-hz_conf_t,'g', alpha=0.4)
ax0[1].plot(zx_t,xcom_t+hx_conf_t,'b', alpha=0.4)
ax0[1].plot(zx_t,ycom_t+hy_conf_t,'r', alpha=0.4)
ax0[1].plot(zx_t,zcom_t+hz_conf_t,'g', alpha=0.4)
ax0[1].axvline(x=-r, color='k', linewidth=0.5)
ax0[1].axvline(x=r, color='k', linewidth=0.5)
ax0[1].axvline(x=-rh, color='k', linewidth=0.5)
ax0[1].axvline(x=rh, color='k', linewidth=0.5)
ax1 = ax0[1].twinx()
ax1.plot(zx_t, B4_t.Bx, 'b.', label='Bx')
ax1.plot(zx_t, B4_t.By, 'r.', label='By')
ax1.plot(zx_t, B4_t.Bz, 'g.', label='Bz')
ax0[1].set_xlabel('$z$ (m)')
ax0[1].set_ylabel('$H_{th}\;in\;T$')
ax1.set_ylabel(r'$B_{measured}\;in\;\mu T$')
ax0[1].ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
ax1.set_ylim([-1.5, 0.5])
ax0[1].set_ylim([-1.5e-6, 0.5e-6])
plt.show()


# %% LEBOW
#---IMPORT DATA---#

# LeBow FOIL2
df2_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1356_POL2_FOIL2magnetizatibackground_avg_corr.csv')
df2_p = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1525_POL2_FOIL2_100A_mZ_avg.csv')
df2_t = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1534_POL2_FOIL2_100A_pX_ttransverse_avg.csv')
B2_p = (df2_p[['Bx','By','Bz']] - df2_bg[['Bx','By','Bz']])*10
B2_t = (df2_t[['Bx','By','Bz']] - df2_bg[['Bx','By','Bz']])*10
B2_bg = df2_bg[['Bx','By','Bz']]

#---DEFINE CYLINDER---#

r=.14115/2      # radius of cylinder (m)
l=400.e-9/2     # half-height of cylinder (m)
h=.0282         # height of scan above plane (m)
Deltax=.2413    # range of scan (m)
dx=.005         # step size for scan (m)

# %%
#---COMPARISON---#

m_th = 0.974

#---PARALLEL---#  

zx_off_p = -0.02
zx_p = np.array(df2_bg.z+14.5)*0.0254+zx_off_p
off_p = -0.0
theta_p = 0*(np.pi/180)+np.arctan(1/740)
tau_p = 0*(np.pi/180)+np.arctan(.15/141.15)+np.arctan(12/740)   # Measured

f = 0.00
g = 1-f

x = np.arange(-Deltax,Deltax,dx)*np.cos(tau_p)
xnonzero=[xp for xp in x if abs(xp)>1.e-10]
y = off_p+np.array(-zx_p)*np.arctan(theta_p)
z = h+np.array(-zx_p)*np.arctan(tau_p)

# Transverse magnetization
hxes_pt = np.array([])
hyes_pt = np.array([])
hzes_pt = np.array([])
for i in range(len(zx_p)):
    yp = y[i]
    xp = -zx_p[i]
    zp = z[i]
    hzes_pt = np.append(hzes_pt,float(g*hx_cart_t(xp,yp,zp,m_th)))
    hxes_pt = np.append(hxes_pt,float(g*hy_cart_t(xp,yp,zp,m_th)))
    hyes_pt = np.append(hyes_pt,float(g*hz_cart_t(xp,yp,zp,m_th)))

# Longitudinal magnetization
hxes_pl = np.array([])
hyes_pl = np.array([])
hzes_pl = np.array([])
for i in range(len(zx_p)):
    yp = y[i]
    xp = -zx_p[i]
    zp = z[i]
    hzes_pl = np.append(hzes_pl, float(f*hx_cart_l(xp,yp,zp,m_th)))
    hxes_pl = np.append(hxes_pl, float(f*hy_cart_l(xp,yp,zp,m_th)))
    hyes_pl = np.append(hyes_pl, float(f*hz_cart_l(xp,yp,zp,m_th)))

# Combined magnetization
xcom_p = hxes_pt+hxes_pl
ycom_p = hyes_pt+hyes_pl
zcom_p = hzes_pt+hzes_pl

hx_conf_p = conf(xcom_p, 0.90)
hy_conf_p = conf(ycom_p, 0.90)
hz_conf_p = conf(zcom_p, 0.90)

fig, ax0 = plt.subplots(1,2)
ax0[0].plot(zx_p,xcom_p,'b',label=r'$Hx\;in\;T$')
ax0[0].plot(zx_p,ycom_p,'r',label=r'$Hy\;in\;T$')
ax0[0].plot(zx_p,zcom_p,'g',label=r'$Hz\;in\;T$')
#ax0[0].plot(zx_p,xcom_p-hx_conf_p,'b', alpha=0.4)
#ax0[0].plot(zx_p,ycom_p-hy_conf_p,'r', alpha=0.4)
#ax0[0].plot(zx_p,zcom_p-hz_conf_p,'g', alpha=0.4)
#ax0[0].plot(zx_p,xcom_p+hx_conf_p,'b', alpha=0.4)
#ax0[0].plot(zx_p,ycom_p+hy_conf_p,'r', alpha=0.4)
#ax0[0].plot(zx_p,zcom_p+hz_conf_p,'g', alpha=0.4)
ax0[0].axvline(x=-r, color='k', linewidth=0.5)
ax0[0].axvline(x=r, color='k', linewidth=0.5)
ax0[0].axvline(x=-rh, color='k', linewidth=0.5)
ax0[0].axvline(x=rh, color='k', linewidth=0.5)
ax1 = ax0[0].twinx()
ax1.plot(zx_p, B2_p.Bx, 'b.', label='Bx')
ax1.plot(zx_p, B2_p.By, 'r.', label='By')
ax1.plot(zx_p, B2_p.Bz, 'g.', label='Bz')
ax0[0].legend(loc='upper left')
ax0[0].set_xlabel('$x$ (m)')
ax0[0].set_ylabel('$H_{th}\;in\;T$')
ax1.set_ylabel(r'$B_{measured}\;in\;\mu T$')
plt.legend()
ax0[0].ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
ax1.set_ylim([-3, 3])
ax0[0].set_ylim([-3e-6, 3e-6])

#---TRANSVERSE---#

zx_off_t = -0.02
zx_t = np.array(df4_bg.z+14.5)*0.0254+zx_off_t
off_t = -0.00
theta_t = np.arctan(1/740)+10*(np.pi/180)
tau_t = np.arctan(.30/141.15)+np.arctan(12/740)+1*(np.pi/180)  # Measured

x = off_t+np.array(-zx_t)*np.arctan(theta_t)
z = h+np.array(-zx_t)*np.arctan(tau_t)+0.00

# Transverse magnetization
hxes_tt = np.array([])
hyes_tt = np.array([])
hzes_tt = np.array([])
for i in range(len(zx_t)):
    yp = -zx_t[i]
    xp = x[i]
    zp = z[i]
    hxes_tt = np.append(hxes_tt, float(g*hx_cart_t(xp,yp,zp,m_th)))
    hzes_tt = np.append(hzes_tt, float(g*hy_cart_t(xp,yp,zp,m_th)))
    hyes_tt = np.append(hyes_tt, float(g*hz_cart_t(xp,yp,zp,m_th)))

# Longitudinal magnetization
hxes_tl = np.array([])
hyes_tl = np.array([])
hzes_tl = np.array([])
for i in range(len(zx_t)):
    yp = -zx_p[i]
    xp = x[i]
    zp = z[i]
    hxes_tl = np.append(hxes_tl, float(f*hx_cart_l(xp,yp,zp,m_th)))
    hzes_tl = np.append(hzes_tl, float(f*hy_cart_l(xp,yp,zp,m_th)))
    hyes_tl = np.append(hyes_tl, float(f*hz_cart_l(xp,yp,zp,m_th)))

# Combined magnetization
xcom_t = hxes_tt+hxes_tl
ycom_t = hyes_tt+hyes_tl
zcom_t = hzes_tt+hzes_tl

hx_conf_t = conf(xcom_t, 0.90)
hy_conf_t = conf(ycom_t, 0.90)
hz_conf_t = conf(zcom_t, 0.90)

ax0[1].plot(zx_t,xcom_t,'b',label=r'$Hx\;in\;T$')
ax0[1].plot(zx_t,ycom_t,'r',label=r'$Hy\;in\;T$')
ax0[1].plot(zx_t,zcom_t,'g',label=r'$Hz\;in\;T$')
#ax0[1].plot(zx_t,xcom_t-hx_conf_t,'b', alpha=0.4)
#ax0[1].plot(zx_t,ycom_t-hy_conf_t,'r', alpha=0.4)
#ax0[1].plot(zx_t,zcom_t-hz_conf_t,'g', alpha=0.4)
#ax0[1].plot(zx_t,xcom_t+hx_conf_t,'b', alpha=0.4)
#ax0[1].plot(zx_t,ycom_t+hy_conf_t,'r', alpha=0.4)
#ax0[1].plot(zx_t,zcom_t+hz_conf_t,'g', alpha=0.4)
ax0[1].axvline(x=-r, color='k', linewidth=0.5)
ax0[1].axvline(x=r, color='k', linewidth=0.5)
ax0[1].axvline(x=-rh, color='k', linewidth=0.5)
ax0[1].axvline(x=rh, color='k', linewidth=0.5)
ax1 = ax0[1].twinx()
ax1.plot(zx_t, B2_t.Bx, 'b.', label='Bx')
ax1.plot(zx_t, B2_t.By, 'r.', label='By')
ax1.plot(zx_t, B2_t.Bz, 'g.', label='Bz')
ax0[1].set_xlabel('$x$ (m)')
ax0[1].set_ylabel('$H_{th}\;in\;T$')
ax1.set_ylabel(r'$B_{measured}\;in\;\mu T$')
ax0[1].ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
ax1.set_ylim([-2, 1])
ax0[1].set_ylim([-2e-6, 1e-6])
plt.show()


# %%
'''
#---FITTING SIMULATION AND MEASUREMENT---#
# Measurements might have a misalignment 
# Searching for the best fit between simulation and measurement

#---DEFINING THE PARAMETERSPACE---#
# Construct a parameterspace of theta, tau and offset

mag = np.linspace(0.6,0.9,5)
theta_f = (np.linspace(-5,-5,10))*(np.pi/180)
tau_f = (np.linspace(-1,1,10))*(np.pi/180)
off_f = (np.linspace(-0.002,-0.001,5))

# %%
#---FITTING MAGNETIZATION---#

dx_m = zx[0] - zx[1]
x_m = np.arange(zx[-1],zx[0],dx_m)
xnonzero_m=[xp for xp in x_m if abs(xp)>1.e-10]
ym = 0
zm = h

for m in mag:
    hxes=[hx_cart_t(xp,ym,zm) for xp in -zx]
    hyes=[hy_cart_t(xp,ym,zm) for xp in -zx]
    hzes=[hz_cart_t(xp,ym,zm) for xp in -zx]
    fig, ax0 = plt.subplots()
    ax0.plot(zx,hxes,'m',label='Hz/Mz')
    ax0.plot(zx,hyes,'y',label='Hx/Mz')
    ax0.plot(zx,hzes,'k',label='Hy/Mz')
    ax1 = ax0.twinx()
    ax1.plot(zx, B_p.Bx, 'y.', label='Bx')
    ax1.plot(zx, B_p.By, 'k.', label='By')
    ax1.plot(zx, B_p.Bz, 'm.', label='Bz')
    ax0.legend(loc='upper left')
    ax0.set_xlabel('$x$ (m)')
    ax0.set_ylabel('$H_{measured}\;in\;T$')
    ax1.set_ylabel(r'Measured B-field in $\mu T$')
    plt.legend()
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
    ax1.set_ylim([-2, 2])
    ax0.set_ylim([-2e-6, 2e-6])
    plt.show()


# %%
#---FITTING ALGORITHM---#
# Iterate over the the parameters
# Determine the LMS
# Refine the parameterspace around the LMS parameters
# Repeat

# First test with FOIL6: thb = 0, ofb = 0, tab = 0.03490658503988659
# First fine test with FOIL6: thb = -0.0029088820866572163, ofb = -0.0005555555555555557, tab = 0.009335881099545066

dx = zx[0] - zx[1]
x = np.arange(-Deltax,Deltax,dx)
xnonzero=[xp for xp in x if abs(xp)>1.e-10]
x_array = np.array(xnonzero)

# Parallel
ms_array = np.array([])
para_list = []
for th in theta_f:
    for of in off_f:
        y = of + x_array*np.arctan(th)
        for ta in tau_f:
            z = h+x_array*np.arctan(ta)

            ms = 0
            for i in range(len(y)):
                yp = y[i]
                xp = xnonzero[i]
                zp = z[i]
                msi = float(abs((abs(hx_cart_t(xp,yp,zp))*10**6)-abs(B_p.Bz[i]))+
                      (abs(abs(hy_cart_t(xp,yp,zp))*10**6)-abs(B_p.Bx[i]))+
                      (abs(abs(hz_cart_t(xp,yp,zp))*10**6)-abs(B_p.By[i])))
                ms += msi
            print(ms, th, of, ta)
            ms_array = np.append(ms_array, ms)
            para_list.append((th, of, ta))

best_idx = ms_array.argmin()
thb = para_list[best_idx][0]
ofb = para_list[best_idx][1]
tab = para_list[best_idx][2]
yb = ofb + x_array*np.arctan(thb)
zb = h+x_array*np.arctan(tab)

hxes = []
hyes = []
hzes = []
for i in range(len(xnonzero)):
    yp = yb[i]
    xp = xnonzero[i]
    zp = zb[i]
    hxes.append(hx_cart_t(xp,yp,zp))
    hyes.append(hy_cart_t(xp,yp,zp))
    hzes.append(hz_cart_t(xp,yp,zp))

fig, ax0 = plt.subplots()
ax0.plot(xnonzero,hxes,'m',label='Hz/Mz')
ax0.plot(xnonzero,hyes,'y',label='Hx/Mz')
ax0.plot(xnonzero,hzes,'k',label='Hy/Mz')
ax1 = ax0.twinx()
ax1.plot(z_m, -B_p.Bx, 'y.', label='Bx')
ax1.plot(-z_m, B_p.By, 'k.', label='By')
ax1.plot(-z_m, B_p.Bz, 'm.', label='Bz')
ax0.legend(loc='upper left')
ax0.set_xlabel('$x$ (m)')
ax0.set_ylabel('$H_{measured}/M_{disk}$ (dimensionless)')
ax1.set_ylabel(r'Measured B-field in $\mu T$')
plt.legend()
ax0.ticklabel_format(style='sci', axis='y', scilimits=(0,-5))
ax1.set_ylim([-2, 2])
ax0.set_ylim([-2e-6, 2e-6])
plt.show()
'''