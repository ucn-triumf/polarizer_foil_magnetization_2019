#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current state of the 6 foils, as they are on 22th Nov 2019

@author: fabian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [8.0*2, 6.0*2]
plt.rcParams['legend.fontsize'] = 'large'

# %%
#---IMPORT DATA---#

# FOIL1: 100 A (LeBow)
df1_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1356_POL2_FOIL2magnetizatibackground_avg_corr.csv')
df1_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL1_100A/20191120_1616_POL1_FOIL1_100A_mZ_para_avg.csv')
df1_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL1_100A/20191120_1623_POL1_FOIL1_100A_pX_transv_avg.csv')
B1_p = (df1_par[['Bx','By','Bz']] - df1_bg[['Bx','By','Bz']])*10
B1_t = (df1_tran[['Bx','By','Bz']] - df1_bg[['Bx','By','Bz']])*10
B1_bg = df1_bg[['Bx','By','Bz']]

# FOIL2: 100 A (LeBow)
df2_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1356_POL2_FOIL2magnetizatibackground_avg_corr.csv')
df2_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1525_POL2_FOIL2_100A_mZ_avg.csv')
df2_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191120_2018_FOIL2_pol2/20191120_1534_POL2_FOIL2_100A_pX_ttransverse_avg.csv')
B2_p = (df2_par[['Bx','By','Bz']] - df2_bg[['Bx','By','Bz']])*10
B2_t = (df2_tran[['Bx','By','Bz']] - df2_bg[['Bx','By','Bz']])*10
B2_bg = df2_bg[['Bx','By','Bz']]

# FOIL3: 100 A (Molvatec)
df3_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191118_FOIL3_100Amp/20191118_1712_background_avg.csv')
df3_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191118_FOIL3_100Amp/20191118_1638_antiparallel_avg_corrected.csv')
df3_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191118_FOIL3_100Amp/20191118_1700_transverse_avg.csv')
B3_p = (df3_par[['Bx','By','Bz']] - df3_bg[['Bx','By','Bz']])*10
B3_t = (df3_tran[['Bx','By','Bz']] - df3_bg[['Bx','By','Bz']])*10
B3_bg = df3_bg[['Bx','By','Bz']]

# FOIL4: 173 A (Molvatec)
df4_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1141_FOIL4_after_170A_background_avg.csv')
df4_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1101_FOIL4_after_170A_parallel_avg_corr.csv')
df4_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191122_FOIL4_Remagnetization_173A/20191122_1126_FOIL4_after_170A_transverse_avg.csv')
B4_p = (df4_par[['Bx','By','Bz']] - df4_bg[['Bx','By','Bz']])*10
B4_t = (df4_tran[['Bx','By','Bz']] - df4_bg[['Bx','By','Bz']])*10
B4_bg = df4_bg[['Bx','By','Bz']]

# FOIL5: 20 A, 30h Masonhall (LeBow)
df5_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191115_FOIL5_after_mesonhall/20191115_1142_background_avg.csv')
df5_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191115_FOIL5_after_mesonhall/20191115_1124_antiparallel_avg.csv')
df5_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191115_FOIL5_after_mesonhall/20191115_1132_transverse_avg.csv')
B5_p = (df5_par[['Bx','By','Bz']] - df5_bg[['Bx','By','Bz']])*10
B5_t = (df5_tran[['Bx','By','Bz']] - df5_bg[['Bx','By','Bz']])*10
B5_bg = df5_bg[['Bx','By','Bz']]

# FOIL6: 100 A (LeBow)
df6_bg = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191119_FOIL6/20191119_1140_background_avg.csv')
df6_par = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191119_FOIL6/20191119_1351_100A_aligned_FOIL6_avg.csv')
df6_tran = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191119_FOIL6/20191119_1402_100A_transverse_FOIL6_avg.csv')
B6_p = (df6_par[['Bx','By','Bz']] - df6_bg[['Bx','By','Bz']])*10
B6_t = (df6_tran[['Bx','By','Bz']] - df6_bg[['Bx','By','Bz']])*10
B6_bg = df6_bg[['Bx','By','Bz']]

zx = np.array(df1_bg.z+14.5)*0.0254
zx5 = np.array(df5_bg.z+14.5)*0.0254

# %%
#---PLOT---#

# 1-3
fig13, ax0 = plt.subplots(3, 2)
# FOIL1
ax0[0,0].plot(zx, B1_p['Bx'], 'b', marker='.', markersize=5, label='Bx')
ax0[0,0].plot(zx, B1_p['By'], 'r', marker='^', markersize=5, label='By')
ax0[0,0].plot(zx, B1_p['Bz'], 'g', marker='s', markersize=5, label='Bz')
#ax0[0,0].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[0,0].set_title('FOIL1:Parallel')
ax0[0,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[0,0].set_xticklabels('')
ax0[0,0].set_ylim([-2.6,2.6])

ax0[0,1].plot(zx, B1_t['Bx'], 'b', marker='.', markersize=5, label='Bx')
ax0[0,1].plot(zx, B1_t['By'], 'r', marker='^', markersize=5, label='By')
ax0[0,1].plot(zx, B1_t['Bz'], 'g', marker='s', markersize=5, label='Bz')
#ax0[0,1].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[0,1].set_title('FOIL1: Transverse')
ax0[0,1].set_xticklabels('')
ax0[0,1].set_ylim([-1.5,1.5])

# FOIL2
ax0[1,0].plot(zx, B2_p['Bx'], 'b', marker='.', markersize=5)
ax0[1,0].plot(zx, B2_p['By'], 'r', marker='^', markersize=5)
ax0[1,0].plot(zx, B2_p['Bz'], 'g', marker='s', markersize=5)
#ax0[1,0].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[1,0].set_title('FOIL2: Parallel')
ax0[1,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[1,0].set_xticklabels('')
ax0[1,0].set_ylim([-2.6,2.6])

ax0[1,1].plot(zx, B2_t['Bx'], 'b', marker='.', markersize=5)
ax0[1,1].plot(zx, B2_t['By'], 'r', marker='^', markersize=5)
ax0[1,1].plot(zx, B2_t['Bz'], 'g', marker='s', markersize=5)
#ax0[1,1].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[1,1].set_title('FOIL2: transverse')
ax0[1,1].set_xticklabels('')
ax0[1,1].set_ylim([-1.5,1.5])

# FOIL3
ax0[2,0].plot(zx, B3_p['Bx'], 'b', marker='.', markersize=5)
ax0[2,0].plot(zx, B3_p['By'], 'r', marker='^', markersize=5)
ax0[2,0].plot(zx, B3_p['Bz'], 'g', marker='s', markersize=5)
#ax0[2,0].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[2,0].set_title('FOIL3: Parallel')
ax0[2,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[2,0].set_xlabel('z in cm')
ax0[2,0].set_ylim([-1,1])

ax0[2,1].plot(zx, B3_t['Bx'], 'b', marker='.', markersize=5)
ax0[2,1].plot(zx, B3_t['By'], 'r', marker='^', markersize=5)
ax0[2,1].plot(zx, B3_t['Bz'], 'g', marker='s', markersize=5)
#ax0[2,1].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[2,1].set_title('FOIL3: Transverse')
ax0[2,1].set_ylabel(r'$B \; in \; \mu T$')
ax0[2,1].set_xlabel('z in cm')
ax0[2,1].set_ylim([-0.5,0.5])

fig13.legend([r'$B_x$',r'$B_y$',r'$B_z$'], 
                labels=[r'$B_x$',r'$B_y$',r'$B_z$'],
                loc="upper center",
                borderaxespad=0, ncol=3)

# 4-6
fig46, ax0 = plt.subplots(3, 2)

# FOIL4
ax0[0,0].plot(zx, B4_p['Bx'], 'b', marker='.', markersize=5)
ax0[0,0].plot(zx, B4_p['By'], 'r', marker='^', markersize=5)
ax0[0,0].plot(zx, B4_p['Bz'], 'g', marker='s', markersize=5)
#ax0[0,0].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[0,0].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[0,0].set_title('FOIL4: Parallel')
ax0[0,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[0,0].set_xticklabels('')
ax0[0,0].set_ylim([-1,1])

ax0[0,1].plot(zx, B4_t['Bx'], 'b', marker='.', markersize=5)
ax0[0,1].plot(zx, B4_t['By'], 'r', marker='^', markersize=5)
ax0[0,1].plot(zx, B4_t['Bz'], 'g', marker='s', markersize=5)
#ax0[0,1].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[0,1].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[0,1].set_title('FOIL4: Transverse')
ax0[0,1].set_xticklabels('')
ax0[0,1].set_ylim([-0.5,0.5])

# FOIL5
ax0[1,0].plot(zx5, B5_p['Bx'], 'b', marker='.', markersize=5)
ax0[1,0].plot(zx5, B5_p['By'], 'r', marker='^', markersize=5)
ax0[1,0].plot(zx5, B5_p['Bz'], 'g', marker='s', markersize=5)
#ax0[1,0].axvline(x=-0.081, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=0.081, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=-0.071, color='k', linewidth=0.5)
#ax0[1,0].axvline(x=0.071, color='k', linewidth=0.5)
ax0[1,0].set_title('FOIL5: Parallel')
ax0[1,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[1,0].set_xticklabels('')
ax0[1,0].set_ylim([-2.6,2.6])

ax0[1,1].plot(zx5, B5_t['Bx'], 'b', marker='.', markersize=5)
ax0[1,1].plot(zx5, B5_t['By'], 'r', marker='^', markersize=5)
ax0[1,1].plot(zx5, B5_t['Bz'], 'g', marker='s', markersize=5)
#ax0[1,1].axvline(x=-0.081, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=0.081, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=-0.071, color='k', linewidth=0.5)
#ax0[1,1].axvline(x=0.071, color='k', linewidth=0.5)
ax0[1,1].set_title('FOIL5: Transverse')
ax0[1,1].set_xticklabels('')
ax0[1,1].set_ylim([-1.5,1.5])

# FOIL6
ax0[2,0].plot(zx, B6_p['Bx'], 'b', marker='.', markersize=5)
ax0[2,0].plot(zx, B6_p['By'], 'r', marker='^', markersize=5)
ax0[2,0].plot(zx, B6_p['Bz'], 'g', marker='s', markersize=5)
#ax0[2,0].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[2,0].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[2,0].set_title('FOIL6: parallel')
ax0[2,0].set_ylabel(r'$B \; in \; \mu T$')
ax0[2,0].set_xlabel('z in cm')
ax0[2,0].set_ylim([-2.6,2.6])

ax0[2,1].plot(zx, B6_t['Bx'], 'b', marker='.', markersize=5)
ax0[2,1].plot(zx, B6_t['By'], 'r', marker='^', markersize=5)
ax0[2,1].plot(zx, B6_t['Bz'], 'g', marker='s', markersize=5)
#ax0[2,1].axvline(x=-0.070575, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=0.070575, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=-0.0425, color='k', linewidth=0.5)
#ax0[2,1].axvline(x=0.0425, color='k', linewidth=0.5)
ax0[2,1].set_title('FOIL6: Transverse')
ax0[2,1].set_xlabel('z in cm')
ax0[2,1].set_ylim([-1.5,1.5])

fig46.legend([r'$B_x$',r'$B_y$',r'$B_z$'], 
                labels=[r'$B_x$',r'$B_y$',r'$B_z$'],
                loc="upper center",
                borderaxespad=0, ncol=3)

