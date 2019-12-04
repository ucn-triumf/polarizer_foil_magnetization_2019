#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:55:37 2019

@author: fpiermaier
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
plt.rcParams.update({'font.size': 20})

# %%

d = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/Characterizing_field_of_coil_20191114/I-B-curve.csv')

it = np.linspace(0,200,201)
t = 0.6932*it-0.3241

lv20 = mlines.Line2D([20,20], [0,t[20]], linewidth=.5, color='k')
lh20 = mlines.Line2D([0,20], [t[20],t[20]], linewidth=.5, color='k')
lv100 = mlines.Line2D([100,100], [0,t[100]], linewidth=.5, color='k')
lh100 = mlines.Line2D([0,100], [t[100],t[100]], linewidth=.5, color='k')
lv173 = mlines.Line2D([173,173], [0,t[173]], linewidth=.5, color='k')
lh173 = mlines.Line2D([0,173], [t[173],t[173]], linewidth=.5, color='k')

extraticks = [0, t[20],t[100],t[173], t[200]]

fig,ax = plt.subplots(1,2)
ax[0].plot(it, t, 'r--')
ax[0].plot(d.I, d.B, 'b.', markersize=8)
ax[0].set_xlabel(r'$I\;in\;A$')
ax[0].set_ylabel(r'$\mathbf{B}\;in\;mT$')
ax[0].add_line(lv20)
ax[0].add_line(lh20)
ax[0].add_line(lv100)
ax[0].add_line(lh100)
ax[0].add_line(lv173)
ax[0].add_line(lh173)
ax[0].set_yticks(list(ax[0].get_yticks()) + extraticks)
ax[0].set_ylim([0,t.max()])
ax[0].set_xlim([0,it.max()])
ax[0].text(90,128,'f(x)=0.6932x-0.3241')

# %%

d2 = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/Characterizing the coil 20191118/20191118_1149_center of top plate is origin_avg.csv')
d2.Bz = d2.Bz*100

# %%
zt = np.linspace(0,29,146)
x0 = -0.6222759*zt**2+21.27345*zt+511.8209
xm45 = -0.65438854*zt**2+21.8441596*zt+520.1338
xm85 = -0.77373387*zt**2+24.5569847*zt+526.9394
xp45 = -0.62748750*zt**2+21.3179445*zt+514.6913
xp85 = -0.71883515*zt**2+23.4905201*zt+518.4414
trends = [xm85, xm45, x0, xp45, xp85]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
xes = np.array(d2.x.unique())
xes = np.sort(xes)
stat = np.array(trends)
stat = stat[:,50:125]
sigma = np.std(stat)
mean = np.mean(stat)
l_stat = mlines.Line2D([10,25], [mean/10,mean/10], linewidth=.5, color='r')

for x in range(len(xes)):
    ax[1].plot(d2[d2.x==xes[x]].z, d2[d2.x==xes[x]].Bz, colors[x],marker='s', linewidth=0,label='Bz, R='+str(xes[x])+' cm')
    ax[1].plot(zt, trends[x]/10,colors[x])    
ax[1].set_ylabel(r'$B_{z}\;in\;mT$')
ax[1].set_xlabel(r'$z\;in\;cm$')
ax[1].axvline(x=25, color='k', linewidth=0.5)
ax[1].axvline(x=10, color='k', linewidth=0.5)
ax[1].set_ylim(0,75)
ax[1].legend()
