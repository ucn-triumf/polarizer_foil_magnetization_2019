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
df_bg_1 = pd.read_csv('/home/fpiermaier/Work Fabian//Polarizers/Polarizer_magnetization/20191113_after_magnetization_Lebow_re-measurement/calibration_antiparallel_and_x_offset.csv')
df_1 = pd.read_csv('/home/fpiermaier/Work Fabian//Polarizers/Polarizer_magnetization/20191113_after_magnetization_Lebow_re-measurement/antiparallel_and_x_offset.csv')

# %%
#---CALIBRATE---#

z = df_bg_1.z.unique()
B_1_cal = pd.DataFrame()

# Run 1
for row in z:
    B_1_slice = df_1[df_1.z==row]
    B_1_cal = B_1_cal.append(B_1_slice)
B_1_cal = B_1_cal.sort_values(['x','z'],ascending=False).reset_index(drop=True)
df_bg_1 = df_bg_1.sort_values(['x','z'],ascending=False).reset_index(drop=True)

B_1_cal = B_1_cal[['x','y','z','Bx','By','Bz']]
B_1_cal[['Bx','By','Bz']] = (B_1_cal[['Bx','By','Bz']] - df_bg_1[['Bx','By','Bz']])*100
B_1_cal['y'] = B_1_cal['y'] - 1.5
B_1_cal['z'] = -B_1_cal['z']
B_1_cal['B_mag'] = B_1_cal['x']**2 + B_1_cal['y']**2 + B_1_cal['z']**2

# %%
#---STREAMPLOT---#

B_1_mid = B_1_cal[B_1_cal.x == 0].reset_index(drop=True)

By = np.array(B_1_mid['By'])
By_pad = np.zeros((20,20))
By_pad[:,10] = By

Bz = np.array(B_1_mid['Bz'])
Bz_pad = np.zeros((20,20))
Bz_pad[:,10] = Bz

Y = Z = np.linspace(1,19,20)

fig = plt.figure()
plt.streamplot(Y, Z, By_pad, Bz_pad)




