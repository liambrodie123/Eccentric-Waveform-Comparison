#Copyright (C) 2019 Liam Brodie
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
'''
Putting the .dat files that were output from the HinderEIMR mathematica code into an hdf5 file
'''
#
#
import numpy as np
import h5py

ecc_array = np.linspace(0.000, 0.472, num = 472, endpoint = False)

for i in ecc_array:
    data = np.loadtxt('/path/to/local/hinderEIMR_' + str("%.3f" % round(i, 3)) + '_1_1_0_0_rh_.dat')
    with h5py.File('HinderEIMR_waveform_generation.hdf5', 'a') as f:
        f.create_dataset('hinderEIMR_' + str("%.3f" % round(i, 3)) + '_1_1_0_0_rh_.dat', data = data)

with h5py.File('HinderEIMR_waveform_generation.hdf5', 'r') as read_f:
    for groups in read_f:
        print(groups)











































'''
#mathematica_data = "C:/Users/liamb/research_projects/eccentricity_project/hinder_mathematica_code/EccentricIMR/hinderEIMR__0_01__1_1_0_0_rh_.dat" # Use if running from spyder IDE
#mathematica_data = "mnt/c/users/liamb/research_projects/eccentricity_project/hinder_mathematica_code/EccentricIMR/hinderEIMR__0_01__1_1_0_0_rh_.dat" # Use if running from terminal
hinder__0_01 = np.loadtxt("hinderEIMR__0_01__1_1_0_0_rh_.dat") # data for e = 0.01 HinderEIMR waveform
hinder__0_02 = np.loadtxt("hinderEIMR__0_02__1_1_0_0_rh_.dat")
hinder__0_03 = np.loadtxt("hinderEIMR__0_03__1_1_0_0_rh_.dat")
hinder__0_04 = np.loadtxt("hinderEIMR__0_04__1_1_0_0_rh_.dat")
hinder__0_05 = np.loadtxt("hinderEIMR__0_05__1_1_0_0_rh_.dat")
hinder__0_06 = np.loadtxt("hinderEIMR__0_06__1_1_0_0_rh_.dat")
hinder__0_10 = np.loadtxt("hinderEIMR__0_10__1_1_0_0_rh_.dat")
hinder__0_15 = np.loadtxt("hinderEIMR__0_15__1_1_0_0_rh_.dat")
hinder__0_20 = np.loadtxt("hinderEIMR__0_20__1_1_0_0_rh_.dat")
hinder__0_25 = np.loadtxt("hinderEIMR__0_25__1_1_0_0_rh_.dat")
hinder__0_30 = np.loadtxt("hinderEIMR__0_30__1_1_0_0_rh_.dat")
hinder__0_35 = np.loadtxt("hinderEIMR__0_35__1_1_0_0_rh_.dat")
hinder__0_40 = np.loadtxt("hinderEIMR__0_40__1_1_0_0_rh_.dat")


#with h5py.File('HinderEIMR_mathematica_waveform_generation.hdf5', 'a') as f: # creating an hdf5 file
#    f.create_dataset("hinderEIMR__0_01__1_1_0_0_rh_.dat", data = hinder__0_01) # making a dataset
#    f.create_dataset("hinderEIMR__0_02__1_1_0_0_rh_.dat", data = hinder__0_02)
#    f.create_dataset("hinderEIMR__0_03__1_1_0_0_rh_.dat", data = hinder__0_03)
#    f.create_dataset("hinderEIMR__0_04__1_1_0_0_rh_.dat", data = hinder__0_04)
#    f.create_dataset("hinderEIMR__0_05__1_1_0_0_rh_.dat", data = hinder__0_05)
#    f.create_dataset("hinderEIMR__0_06__1_1_0_0_rh_.dat", data = hinder__0_06)
#    f.create_dataset("hinderEIMR__0_10__1_1_0_0_rh_.dat", data = hinder__0_10)
#    f.create_dataset("hinderEIMR__0_15__1_1_0_0_rh_.dat", data = hinder__0_15)
#    f.create_dataset("hinderEIMR__0_20__1_1_0_0_rh_.dat", data = hinder__0_20)
#    f.create_dataset("hinderEIMR__0_25__1_1_0_0_rh_.dat", data = hinder__0_25)
#    f.create_dataset("hinderEIMR__0_30__1_1_0_0_rh_.dat", data = hinder__0_30)
#    f.create_dataset("hinderEIMR__0_35__1_1_0_0_rh_.dat", data = hinder__0_35)
#    f.create_dataset("hinderEIMR__0_40__1_1_0_0_rh_.dat", data = hinder__0_40)


with h5py.File('HinderEIMR_mathematica_waveform_generation.hdf5', 'r') as read_f:
    for groups in read_f:
        print(groups)
    
'''
    
    
    
    
    
    
    
    