#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:25:42 2019

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
plt.rcParams['axes.grid'] = True


# %%
#---IMPORT DATA---#

# Data from the first run: only x-offset
df_bg_1 = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191113_FOIL5_magnetization_re-measurement/calibration_antiparallel_and_x_offset.csv')
df_1 = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191113_FOIL5_magnetization_re-measurement/antiparallel_and_x_offset.csv')

# Data from the second run: y and x offset
df_bg_2 = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191113_FOIL5_magnetization_re-measurement/calibration_antiparallel_and_y_offset.csv')
df_2 = pd.read_csv('/home/fpiermaier/Work Fabian/Polarizers/Polarizer_magnetization/20191113_FOIL5_magnetization_re-measurement/antiparallel_and_y_offset.csv')

# %%
#---CALIBRATE---#

z = df_bg_1.z.unique()
B_1_cal = pd.DataFrame()
B_2_cal = pd.DataFrame()

# Run 1
for row in z:
    B_1_slice = df_1[df_1.z==row]
    B_1_cal = B_1_cal.append(B_1_slice)
B_1_cal = B_1_cal.sort_values(['x','z'],ascending=False).reset_index(drop=True)
df_bg_1 = df_bg_1.sort_values(['x','z'],ascending=False).reset_index(drop=True)

B_1_cal = B_1_cal[['x','y','z','Bx','By','Bz']]
B_1_cal[['Bx','By','Bz']] = (B_1_cal[['Bx','By','Bz']] - df_bg_1[['Bx','By','Bz']])*100
B_1_cal['y'] = B_1_cal['y'] - 1.5
B_1_cal['z'] = (-B_1_cal['z']*2.54)-35.65

# Run 2
for row in z:
    B_2_slice = df_2[df_2.z==row]
    B_2_cal = B_2_cal.append(B_2_slice)
B_2_cal = B_2_cal.sort_values(['x','z'],ascending=False).reset_index(drop=True)
df_bg_2 = df_bg_2.sort_values(['x','z'],ascending=False).reset_index(drop=True)

B_2_cal = B_2_cal[['x','y','z','Bx','By','Bz']]
B_2_cal[['Bx','By','Bz']] = (B_2_cal[['Bx','By','Bz']] - df_bg_2[['Bx','By','Bz']])*100
B_2_cal['y'] = B_2_cal['y'] - 1.5
B_2_cal['z'] = (-B_2_cal['z']*2.54)-35.65

# %%
#---SELECT DATA AROUND THE FOIL---#

z_cut = 15

B_1_cut = B_1_cal[B_1_cal.z > -z_cut]
B_1_cut = B_1_cut[B_1_cut.z < z_cut] 

B_2_cut = B_2_cal[B_2_cal.z > -z_cut]
B_2_cut = B_2_cut[B_2_cut.z < z_cut] 

# %%
#---QUIVER---#

quiv_plot = plt.figure()
axn = quiv_plot.add_subplot(111, projection='3d')
# Color by length of vector
cn_v = np.sqrt(B_1_cut['Bx']**2+B_1_cut['By']**2+B_1_cut['Bz']**2)
# Flatten and normalize
cn = (cn_v.ravel() - cn_v.min()) / np.ptp(cn_v)
# Repeat for each body line and two head lines
cn = np.concatenate((cn, np.repeat(cn, 2)))
# Colormap
cn = plt.cm.plasma(cn)
run1 = axn.quiver(B_1_cut['x'], B_1_cut['y'], B_1_cut['z'], B_1_cut['Bx'], B_1_cut['By'], B_1_cut['Bz'], colors=cn, length=.5)
run2 = axn.quiver(B_2_cut['x'], B_2_cut['y'], B_2_cut['z'], B_2_cut['Bx'], B_2_cut['By'], B_2_cut['Bz'], colors=cn, length=.5)
axn.set_xlabel('X axis')
axn.set_ylabel('Y axis')
axn.set_zlabel('Z axis')
axn.set_xlim(-20, 20)
axn.set_ylim(-20, 20)
axn.set_zlim(-15, 15)
axn.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
axn.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
axn.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
#run1.set_array(np.linspace(0,cn_v.max(),100))
#run1.set_edgecolor(cn)
#run1.set_facecolor(cn)
#cb = plt.colorbar(run1)
#cb.set_label(r'$B \; in \; \mu T$')
quiv_plot.tight_layout(pad=3,rect=[0, 0, 1, 0.99])
disk = Circle((0,0),7.65, color='k', fill=False)
axn.add_patch(disk)
art3d.pathpatch_2d_to_3d(disk, z=0, zdir="y")


# %%
#---PLOT---#


par_fig, ax0 = plt.subplots(2, 2)
ax0[0,0].plot(z, B_par_cal['Bx'], 'b.', markersize=8)
ax0[0,0].plot(z, B_tran_cal['Bx'], 'r.', markersize=8)
ax0[0,0].plot(z, B_bg['Bx'], 'g.', markersize=8)
ax0[0,0].plot(z, B_bg['Bx'], 'g', markersize=8)
ax0[0,0].plot(z, B_par_cal['Bx'], 'b')
ax0[0,0].plot(z, B_tran_cal['Bx'], 'r')
ax0[0,0].set_ylabel(r'$B_x \; in \; \mu T$')
ax0[0,0].set_xticklabels('')
ax0[1,0].plot(z, B_par_cal['By'], 'b.', markersize=8)
ax0[1,0].plot(z, B_par_cal['By'], 'b')
ax0[1,0].plot(z, B_bg['By'], 'g.', markersize=8)
ax0[1,0].plot(z, B_bg['By'], 'g', markersize=8)
ax0[1,0].plot(z, B_tran_cal['By'], 'r.', markersize=8)
ax0[1,0].plot(z, B_tran_cal['By'], 'r')
ax0[1,0].set_xlabel('z in cm')
ax0[1,0].set_ylabel(r'$B_y \; in \; \mu T$')
ax0[0,1].plot(z, B_par_cal['Bz'], 'b.', markersize=8)
ax0[0,1].plot(z, B_par_cal['Bz'], 'b')
ax0[0,1].plot(z, B_bg['Bz'], 'g.', markersize=8)
ax0[0,1].plot(z, B_bg['Bz'], 'g', markersize=8)
ax0[0,1].plot(z, B_tran_cal['Bz'], 'r.', markersize=8)
ax0[0,1].plot(z, B_tran_cal['Bz'], 'r')
ax0[0,1].set_ylabel(r'$B_z \; in \; \mu T$')
ax0[0,1].set_xlabel('z in cm')
ax0[-1,-1].axis('off')

par_fig.legend(['parallel','orthogonal','background'], labels=['parallel','orthogonal','background'], loc="lower center",
           bbox_to_anchor=(0.45, 0.2, 0.5, 0.5), title="Orientation")

