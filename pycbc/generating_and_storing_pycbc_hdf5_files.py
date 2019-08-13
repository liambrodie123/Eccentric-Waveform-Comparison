#Copyright (C) 2019 Liam Brodie
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
'''
A way of generating and storing PyCBC waveforms
'''
#
#
import matplotlib.pyplot as plt
import h5py
from pycbc.waveform import get_td_waveform
import numpy as np

'''
Listing the waveform parameters
'''
approximant = 'SEOBNRv4' # Name of the PyCBC waveform to generate
m1 = 30 # Mass of the first black hole [Solar masses]
m2 = 30 # Mass of the second black hole [Solar masses]
s1z = 0 # z-component of the spin of the first black hole [dimensionless]
s2z = 0 # z-component of the spin of the second black hole [dimensionless]
dt = 1.0/4096 # Inverse of the sampling frequency in seconds
fl = 10 # [Hz]
distance = 410 # [Megaparsecs]

'''
Generating the waveform
'''
hp, hc = get_td_waveform(approximant = approximant, 
                             mass1 = m1,
                             mass2 = m2,
                             spin1z = s1z,
                             spin2z = s2z,
                             delta_t = dt,
                             f_lower = fl,
                             distance = distance,)

# Plotting the generated waveform
#plt.plot(hp.sample_times, hp, label = approximant)
#plt.ylabel('Strain')
#plt.xlabel('Time (s)')
#plt.legend()
#plt.savefig("pycbc_" + approximant + "_{}_{}_{}_{}_{}_{}_.png".format(m1, m2, distance, s1z, s2z, fl))

'''
# Storing the waveform data
'''
pycbc_output = np.vstack((hp.sample_times, hp, hc)).T # Combine the different arrays and transpose to get a (1,3) array
outfile_name = "pycbc_" + approximant + "_{}_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z, fl)
with h5py.File('/local/path/to/desired/hdf5/file/location', 'a') as f: # Creating a hdf5 file
    outfile = f.create_dataset(outfile_name, data = pycbc_output) # Making a dataset

'''
Reading the hdf5 file 
'''
pycbc_waveforms = h5py.File('/location/of/the/hdf5/file/generated/above','r')
# Print the groups we have in the root group for the first waveform
print(pycbc_waveforms.keys()) 

###
