#Copyright (C) 2019 Liam Brodie
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#
#
'''
Plotting the data generated from the waveform comparison code
'''
#
#
import h5py
import matplotlib.pyplot as plt

w1_name = "HinderEIMR" # Name of the first approximant
w2_name = "SEOBNRv4" # Name of the second approximant
m1 = 30 # Mass of the first black hole [Solar Masses]
m2 = 30 # Mass of the second black hole [Solar Masses]
distance = 410 # Luminosity distance to the binary [Megaparsecs]
s1z = 0 # z-component of the spin of the first black hole [Dimensionaless]
s2z = 0 # z-component of the spin of the second black hole [Dimensionless]

with h5py.File('/local/path/to/HinderEIMR_SEOBNRE_comparison.hdf5', 'r') as read_analysis_file: # reading an hdf5 file
        print(read_analysis_file.keys()) 
        idx_analysis = read_analysis_file[w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)] # reading a specific key inside the hdf5 file
        ecc_array = idx_analysis[:,0] # eccentricity array
        op_snr_array = idx_analysis[:,1] # optimal SNR array
        m_snr_array = idx_analysis[:,2] # measured SNR array
        mismatch_array = idx_analysis[:,3] # mismatch array
        ff_array = idx_analysis[:,4] # fitting factor array
        print('Eccentricity Array: ', ecc_array)
        print('Optimal SNR Array: 'op_snr_array)
        print('Measured SNR Array: 'm_snr_array)
        print('Mismatch Array: 'mismatch_array)
        print('Fitting Factor Array: 'ff_array)
        plt.plot(ecc_array, ff_array, label = 'Fitting Factor')
        plt.title("HinderEIMR vs. SEOBNRv4")
        plt.legend(loc = 'best')
        plt.xlabel('Eccentricity')
        plt.ylabel('Fitting Factor')
        plt.savefig('HinderEIMR_SEOBNRv4_ff.png')
        
        
        
        