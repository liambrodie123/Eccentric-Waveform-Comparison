#Copyright (C) 2019 Liam Brodie
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#
#
'''
Generating an hdf5 file that the data from the waveform comparison code will be stored in
'''
#
#
import numpy as np
import h5py

'''
Parameters of the system to be analyzed
'''
m1 = 30 # Mass of the first black hole [solar masses]
m2 = 30 # Mass of the second black hole [solar masses]
distance = 410 # Luminosity distance to the binary [megaparsecs]
s1z = 0 # z-component spin of first black hole [dimensionless]
s2z = 0 # z-component spin of second black hole [dimensionless]

w1_name = "HinderEIMR" # Name of the first waveform
w2_name = "SEOBNRv4" # Name of the second waveform

'''
Initially generating the arrays that the analysis data will be stored in
'''
eccentricity = []
op_snr = []
m_snr = []
mismatch = []
fitting_factor = []

comparison_data = np.vstack((eccentricity, op_snr, m_snr, mismatch, fitting_factor)).T # Creating a (5,1) array then transposing to create a (1,5) array
print('Shape of waveform comparison data array: ', comparison_data.shape)
# Naming the key (or dataset) that will be stored inside the hdf5 file
outfile = w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)

'''
Storing the array in an hdf5 file
'''
with h5py.File('HinderEIMR_SEOBNRv4_comparison.hdf5', 'w') as f: # creating an hdf5 file
    outfile = f.create_dataset(outfile, data = comparison_data) # making a dataset inside the hdf5 file
    for groups in f:
        print ('groups: ', groups)
        
'''
Reading the newly created hdf5 file
'''
with h5py.File('HinderEIMR_SEOBNRv4_comparison.hdf5', 'r') as read_analysis_file: # reading an hdf5 file
        print(read_analysis_file.keys()) # Printing the keys inside the hdf5 file
        idx_analysis = read_analysis_file[w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)] # Choosing the desired key inside the hdf5 file
        ecc_array = idx_analysis[:,0] # Eccentricity array
        op_snr_array = idx_analysis[:,1] # Optimal SNR array
        m_snr_array = idx_analysis[:,2] # Measured SNR array
        mismatch_array = idx_analysis[:,3] # Mismatch array
        ff_array = idx_analysis[:,4] # Fitting factor array
        print('Final Eccentricity Array: ', ecc_array)
        print('Final Optimal SNR Array: ', op_snr_array)
        print('Final Measured SNR Array: ', m_snr_array)
        print('Final Mismatch Array: ', mismatch_array)
        print('Final Fitting Factor Array: ', ff_array)
        
###