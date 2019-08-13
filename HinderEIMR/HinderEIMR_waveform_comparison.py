#Copyright (C) 2019 Liam Brodie
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
'''
A way of computing the fitting factor, mismatch, optimal SNR, and measured SNR of two waveforms, analytic or numerical.
'''
#
#
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy
from scipy import signal 
import math

''' 
Enter range of eccentricities to loop over
'''
starting_eccentricity = 0
ending_eccentricity = 0.472
eccentricity_step_size = 0.001 # Can be found as: number of samples = 1 + ((max_e - min_e) / step_size) if endpoint = True and ((max_e - min_e) / step_size) if endpoint = False
#ecc_array = np.linspace(starting_eccentricity, ending_eccentricity, num = int((ending_eccentricity - starting_eccentricity) / eccentricity_step_size), endpoint = False) 
ecc_array = np.linspace(starting_eccentricity, ending_eccentricity, num = 472, endpoint = False)


for i in ecc_array:
    
    '''
    List the names and parameters of the models used. It is assumed that each generated model will be of the same mass, spin, and distance.
    '''
    eccentricity = "%.3f" % round(i, 3) 
    w1_name = "HinderEIMR" # Name of the first waveform
    w2_name = "SEOBNRv4" # Name of the second waveform
    m1 = 30 # Mass of the first black hole [Solar masses]
    m2 = 30 # Mass of the second black hole [Solar masses]
    distance = 410 # Luminosity distance to the binary [Parsecs]
    s1z = 0 # z-component of the spin for the first black hole [Dimensionless spin]
    s2z = 0 # z-component of the spin for the second black hole [Dimensionless spin]
    Mass_total = m1 + m2 # Total mass of the system [Solar masses]
    M_sun_kg = 1.98892e30 # [Solar mass in kg]
    GG = 6.67428e-11 # Universal gravitational constant [m^3/(kg*s^2)]
    CC = 2.99792458e8 # Speed of light in vacuum [m/s]
    M_sun_sec = GG * M_sun_kg / CC**3 # Solar mass in seconds [seconds]
    M_sun_m = GG * M_sun_kg / CC**2 # Solar mass in meters [meters]
    Psc_to_m = 3.08567782e22 # Conversion factor from parsecs to meters [meters]
    
    # Opening the first waveform. The first waveform is considered to be the template in this code.
    waveform_1 = "/local/path/to/HinderEIMR_waveform_generation.hdf5" 
    
    # Opening the second waveform. The second waveform is considered to be the signal in this code.
    waveform_2 = "/local/path/to/pycbc_waveforms.hdf5"  
    
    # Reading the first hdf5 waveform file 
    w1_data = h5py.File(waveform_1,'r')
    #Print the groups we have in the root group for the first waveform
    #for group in sorted(w1_data): 
    #    print(group)
    idx_w1 = 'hinderEIMR_' + str("%.3f" % round(i, 3)) + '_1_1_0_0_rh_.dat' # Choose a file from the printed groups and enter it here
    raw_data_w1 = w1_data[idx_w1][:,:] # Take all the rows and columns from the group chosen above
    
    # Reading the second hdf5 waveform file 
    w2_data = h5py.File(waveform_2,'r')
    # Print the groups we have in the root group for the second waveform
    #for group in sorted(w2_data): 
    #    print(group)   
    idx_w2 = 'pycbc_SEOBNRv4_30_30_410_0_0_10_.dat' # Choose a file from the printed groups and enter it here
    raw_data_w2 = w2_data[idx_w2][:,:] # Take all the rows and columns from the group chosen above
              
    # Naming the data columns for the first waveform
    sim_time_w1 = Mass_total * M_sun_sec * w1_data[idx_w1][:,0] # Simulation time [seconds]
    h_w1 = w1_data[idx_w1][:,1] / (distance * Psc_to_m) # Strain [dimensionless] h(t) = h+(t) ... HinderEIMR only generated h+(t)
    # Naming the data columns for the second waveform
    sim_time_w2 = w2_data[idx_w2][:,0] # Simulation time [seconds]
    h_w2 = (w2_data[idx_w2][:,1] + 1j * w2_data[idx_w2][:,2]) # Strain [dimensionless] h(t) = h+(t) + i*hx(t) 
    
    # Find the maximum value of the strain for the first waveform
    max_real_w1 = max(np.real(h_w1))
    # Find the merger time for the first waveform
    merger_time_w1 = sim_time_w1[[k for k, j in enumerate(np.real(h_w1)) if j==max_real_w1]]
    # Convert from counting up until merger to counting down until merger for the first waveform
    time_from_merger_w1 = sim_time_w1 - merger_time_w1
    
    # Find the maximum value of the strain for the second waveform
    max_real_w2 = max(np.real(h_w2))
    # Find the merger time for the second waveform
    merger_time_w2 = sim_time_w2[[k for k, j in enumerate(np.real(h_w2)) if j==max_real_w2]]
    # Convert from counting up until merger to counting down until merger for the second waveform
    time_from_merger_w2 = sim_time_w2 - merger_time_w2
    
    # Plotting the 'raw' first waveform data (with adjusted time scale)
    #plt.plot(time_from_merger_w1, np.real(h_w1), label='Real_w1')
    #plt.plot(time_from_merger_w1, np.imag(h_w1), label='Imag_w1')
    #plt.plot(time_from_merger_w1,  np.abs(h_w1), label='Abs_w1')

    # Plotting the 'raw' second waveform data (with adjusted time scale)
    #plt.plot(time_from_merger_w2, np.real(h_w2), label='Real_w2')
    #plt.plot(time_from_merger_w2, np.imag(h_w2), label='Imag_w2')
    #plt.plot(time_from_merger_w2,  np.abs(h_w2), label='Abs_w2')
    
    '''
    Interpolation of the first waveform
    '''
    # Amount to pad each size of the first waveform with zeros
    symmetric_padding_w1 = 500000 # Arbitrarily large amount choosen. The padding happens later on, but it needs to be defined here.
    # Interpolation x axis variable for the first waveform
    xnew_length_w1 = 2**22 - symmetric_padding_w1 # Length choosen so that we can Fourier transform a 2**n length waveform
    min_time_from_merger_w1 = np.min(time_from_merger_w1) 
    max_time_from_merger_w1 = np.max(time_from_merger_w1)
    min_time_from_merger_w2 = np.min(time_from_merger_w2)
    max_time_from_merger_w2 = np.max(time_from_merger_w2)    
    # If one waveform is shorter than the other, the longer waveform will be trimmed to the duration of the shorter one
    if min_time_from_merger_w1 > min_time_from_merger_w2:
        min_time_from_merger_w2 = min_time_from_merger_w1
    if min_time_from_merger_w2 > min_time_from_merger_w1:
        min_time_from_merger_w1 = min_time_from_merger_w2
    if max_time_from_merger_w1 < max_time_from_merger_w2:
        max_time_from_merger_w2 = max_time_from_merger_w1
    if max_time_from_merger_w2 < max_time_from_merger_w1:
        max_time_from_merger_w1 = max_time_from_merger_w2
    xnew_w1 = np.linspace(min_time_from_merger_w1, max_time_from_merger_w1, num = xnew_length_w1, endpoint = True) 
    xnew_w1_step_size = (np.max(xnew_w1) - np.min(xnew_w1)) / (len(xnew_w1) - 1)
    # Cubic spline interpolation for the first waveform
    cubic_w1 = interp1d(time_from_merger_w1, np.real(h_w1), kind = 'cubic')
    cubic_strain_w1 = cubic_w1(xnew_w1)
    
    '''
    Interpolation of the second waveform
    '''
    # Amount to pad each size of the second waveform with zeros
    symmetric_padding_w2 = 500000 # Arbitrarily large amount choosen. The padding happens later on, but it needs to be defined here.
    # Interpolation x axis variable for the second waveform
    xnew_length_w2 = 2**22 - symmetric_padding_w2 # Length choosen so that we can Fourier transform a 2**n length waveform
    xnew_w2 = np.linspace(min_time_from_merger_w2, max_time_from_merger_w2, num = xnew_length_w2, endpoint = True) 
    xnew_w2_step_size = (np.max(xnew_w2) - np.min(xnew_w2)) / (len(xnew_w2) - 1)
    # Cubic Spline Interpolation for the second waveform
    cubic_w2 = interp1d(time_from_merger_w2, np.real(h_w2), kind='cubic')
    cubic_strain_w2 = cubic_w2(xnew_w2)

    '''
    Windowing the first interpolated waveform
    '''    
    window_w1 = scipy.signal.tukey(len(cubic_strain_w1), alpha = 0.015, sym = False) # MANUALLY need to change alpha if there is too much spectral leakage...possibly automate in future
    #plt.plot(xnew_w1, window_w1, label = 'Window w1') # Plotting the window function for the first waveform
    windowed_w1 = cubic_strain_w1 * window_w1
    
    '''
    Windowing the second interpolated waveform
    '''
    window_w2 = scipy.signal.tukey(len(cubic_strain_w2), alpha = 0.015, sym = False) # MANUALLY need to change alpha if there is too much spectral leakage...possibly automate in future
    #plt.plot(xnew_w2, window_w2, label = 'Window w2') # Plotting the window function for the second waveform
    windowed_w2 = cubic_strain_w2 * window_w2
    
    # Plotting interpolated and windowed waveforms
    plt.plot(xnew_w1, cubic_w1(xnew_w1), label = w1_name + ' interpolation(cubic)')
    plt.plot(xnew_w2, cubic_w2(xnew_w2), label = w2_name + ' interpolation(cubic)')
    plt.plot(xnew_w1, windowed_w1, '--', label = 'Windowed ' + w1_name)
    plt.plot(xnew_w2, windowed_w2, '--', label = 'Windowed ' + w2_name)
    plt.title(w1_name + ' and ' + w2_name)
    plt.xlabel('$(t-t_{merger})$ [s]')
    plt.ylabel('$h_{+}^{2,2}$')
    plt.legend(loc = 'best')
    plt.savefig('interpolation_and_window_for_' + str("%.3f" % round(i, 3)) + '.png')
    plt.close()
    
    '''
    Zero padding the first windowed waveform
    '''
    left_pad_w1 = right_pad_w1 = symmetric_padding_w1 # This line can be rearranged if different amounts of zeros need to be padded on either side
    padded_strain_w1 = np.pad(windowed_w1, (left_pad_w1, right_pad_w1), 'constant')
    # Half the length difference of the amount of zeros padded on the first waveform
    strain_pad_diff_w1 = (len(padded_strain_w1) - len(xnew_w1)) / 2
    # Amount that the padded x variable needs to be increased on each side to match the padded waveform to the windowed one for the first waveform
    xpad_append_w1 = xnew_w1_step_size * strain_pad_diff_w1
    # Padding x axis variable for the first waveform
    xpad_w1 = np.linspace(np.min(xnew_w1) - xpad_append_w1, np.max(xnew_w1) + xpad_append_w1, num = len(padded_strain_w1), endpoint = True)
    #plt.plot(xpad_w1, padded_strain_w1, label = 'Padded w1') # Plot of the first zero padded waveform
    
    '''
    Zero padding the second windowed waveform
    '''
    left_pad_w2 = right_pad_w2 = symmetric_padding_w2 # This line can be rearranged if different amounts of zeros need to be padded on either side
    padded_strain_w2 = np.pad(windowed_w2, (left_pad_w2, right_pad_w2), 'constant')
    # Half the length difference of the amount of zeros padded on the second waveform
    strain_pad_diff_w2 = (len(padded_strain_w2) - len(xnew_w2)) / 2
    # Amount that the padded x variable needs to be increased on each side to match the padded waveform to the windowed one for the second waveform
    xpad_append_w2 = xnew_w2_step_size * strain_pad_diff_w2
    # Padding x axis variable for the second waveform
    xpad_w2 = np.linspace(np.min(xnew_w2) - xpad_append_w2, np.max(xnew_w2) + xpad_append_w2, num = len(padded_strain_w2), endpoint = True)
    #plt.plot(xpad_w2, padded_strain_w2, label ='Padded w2') # Plot of the second zero padded waveform
    
    '''
    Opening the detector noise file
    '''
    detector_sensitivity_file = "/local/path/to/aLIGO_detector_sensitivity.txt"
    detector_data = np.loadtxt(detector_sensitivity_file, skiprows = 7)
    detector_frequency = detector_data[48:,0] # [Hz]
    aLIGO_design_PSD = np.square(detector_data[48:,5])
    detector_freq_spacing = (np.max(detector_frequency) - np.min(detector_frequency)) / (len(detector_frequency) - 1)
    #plt.loglog(detector_frequency, aLIGO_design_PSD, label = 'aLIGO design sensitivity') # Plot of the detector PSD vs. frequency on log-log scale 

    '''
    Fourier transforming the first zero-padded, windowed waveform
    '''
    # Number of sample points for the first waveform
    N_w1 = len(padded_strain_w1)
    # Sample spacing for the first waveform 
    T_w1 = (np.max(xpad_w1) - np.min(xpad_w1)) / (len(xpad_w1) - 1)
    f_padded_strain_w1 = np.fft.fft(padded_strain_w1)
    f_xpad_w1 = np.fft.fftfreq(N_w1) / T_w1 # This is an array in units of Hz of the fourier frequencies    
    min_valid_freq_value_w1 = (next(x for x, val in enumerate(f_xpad_w1) if val > np.min(detector_frequency))) + 1 # Index of the minimum fft frequency that is inside the detector frequency range for the first waveform
    max_valid_freq_value_w1 = (next(x for x, val in enumerate(f_xpad_w1) if val > np.max(detector_frequency))) - 1 # Index of the maximum fft frequency that is inside the detector frequency range for the first waveform
    min_pos_freq_w1 = f_xpad_w1[min_valid_freq_value_w1] 
    max_pos_freq_w1 = f_xpad_w1[max_valid_freq_value_w1] 
    pos_freq_w1 = f_xpad_w1[min_valid_freq_value_w1:max_valid_freq_value_w1]
    valid_f_strain_w1 = 1/N_w1*np.abs(f_padded_strain_w1[min_valid_freq_value_w1:max_valid_freq_value_w1]) # Restricting frequencies only to those in the aLIGO band
    
    '''
    Fourier transforming the second zero-padded, windowed waveform
    '''
    # Number of sample points for the second waveform
    N_w2 = len(padded_strain_w2)
    # Sample spacing for the second waveform
    T_w2 = (np.max(xpad_w2) - np.min(xpad_w2)) / (len(xpad_w2) - 1)
    f_padded_strain_w2 = np.fft.fft(padded_strain_w2)
    f_xpad_w2 = np.fft.fftfreq(N_w2) / T_w2 # This is an array in units of Hz of the fourier frequencies
    min_valid_freq_value_w2 = (next(x for x, val in enumerate(f_xpad_w2) if val > np.min(detector_frequency))) + 1 # Index of the minimum fft frequency that is inside the detector frequency range for the second waveform
    max_valid_freq_value_w2 = (next(x for x, val in enumerate(f_xpad_w2) if val > np.max(detector_frequency))) - 1 # Index of the maximum fft frequency that is inside the detector frequency range for the second waveform
    min_pos_freq_w2 = f_xpad_w2[min_valid_freq_value_w2] 
    max_pos_freq_w2 = f_xpad_w2[max_valid_freq_value_w2] 
    pos_freq_w2 = f_xpad_w2[min_valid_freq_value_w2:max_valid_freq_value_w2] 
    valid_f_strain_w2 = 1/N_w2*np.abs(f_padded_strain_w2[min_valid_freq_value_w2:max_valid_freq_value_w2]) # Restricting frequencies only to those in the aLIGO band
    
    # Plotting the Fourier transformed waveforms
    plt.loglog(pos_freq_w1, valid_f_strain_w1, label = 'fft_' + w1_name) # loglog scale plot of the first waveform fft in the valid detector freq range
    plt.loglog(pos_freq_w2, valid_f_strain_w2, label = 'fft_' + w2_name) # loglog scale plot of the second waveform fft in the valid detector freq range
    plt.title(w1_name + ' and ' + w2_name + ' FFTs')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$h_{+}^{2,2}$')
    plt.legend(loc = 'best')
    plt.savefig('fft_plot_e_' + str("%.3f" % round(i, 3)) + '.png')
    plt.close()
    
    '''
    Interpolating the detector data
    '''
    # Interpolation x axis variable for the detector noise data
    freq_interp_noise = np.linspace(min_pos_freq_w1, max_pos_freq_w1, num = len(pos_freq_w1), endpoint = True) 
    freq_interp_noise_step_size = (np.max(freq_interp_noise) - np.min(freq_interp_noise)) / (len(freq_interp_noise) - 1)
    # Interpolation for the detector noise
    interp_noise = interp1d(detector_frequency, aLIGO_design_PSD, kind = 'cubic') 
    detector_noise_interp = interp_noise(freq_interp_noise)
    #plt.loglog(freq_interp_noise, detector_noise_interp, label = 'Interpolated Noise') # Plotting the interpolated detector noise
    
    '''
    Computing the inner products, optimal SNR, measured SNR, mismatch, and fitting factor
    '''
    # Inner product integrands: h(f) is the template, s(f) is the signal, and Sn(f) is the detector noise
    h_star_s_over_noise = 4 * (np.conj(valid_f_strain_w1) * valid_f_strain_w2) / detector_noise_interp # h(f)*s(f)/Sn(f)
    h_star_h_over_noise = 4 * (np.conj(valid_f_strain_w1) * valid_f_strain_w1) / detector_noise_interp # h(f)*h(f)/Sn(f)
    s_star_s_over_noise = 4 * (np.conj(valid_f_strain_w2) * valid_f_strain_w2) / detector_noise_interp # s(f)*s(f)/Sn(f)
            
    # Using the Trapezoidal rule for integration:
    # For h(f)*s(f)/Sn(f)
    h_star_s_over_noise_integral = np.real(scipy.integrate.trapz(h_star_s_over_noise, freq_interp_noise, freq_interp_noise_step_size))
    # For h(f)*h(f)/Sn(f)
    h_star_h_over_noise_integral = np.real(scipy.integrate.trapz(h_star_h_over_noise, freq_interp_noise, freq_interp_noise_step_size))
    # For s(f)*s(f)/Sn(f)
    s_star_s_over_noise_integral = np.real(scipy.integrate.trapz(s_star_s_over_noise, freq_interp_noise, freq_interp_noise_step_size))
    # Optimal SNR 
    op_snr = math.sqrt(s_star_s_over_noise_integral)
    # Measured SNR 
    m_snr = h_star_s_over_noise_integral / math.sqrt(h_star_h_over_noise_integral)
    # Mismatch 
    mismatch = 1 - m_snr/op_snr
    # Fitting Factor maximized over merger time
    fitting_factor = h_star_s_over_noise_integral / (math.sqrt(h_star_h_over_noise_integral) * math.sqrt(s_star_s_over_noise_integral))
    
    print('Eccentricity = ', eccentricity)
    print('Optimal SNR = ', op_snr)
    print('Measured SNR = ', m_snr)
    print('Mismatch = ', mismatch, '    (0.0 means exact match)')
    print('Fitting Factor (maxized over merger time) = ', fitting_factor, '    (1.0 means exact match)')
    
    '''
    Storing the analysis data
    '''
    with h5py.File('/local/path/to/HinderEIMR_SEOBNRv4_comparison.hdf5', 'r') as read_analysis_file: # Reading in a hdf5 file
        print('Keys from hdf5 file: ', read_analysis_file.keys()) 
        idx_analysis = read_analysis_file[w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)] # Reading a specific key from the file
        ecc_array = idx_analysis[:,0] # Eccentricity array
        op_snr_array = idx_analysis[:,1] # Optimal SNR array
        m_snr_array = idx_analysis[:,2] # Measured SNR array
        mismatch_array = idx_analysis[:,3] # Mismatch array
        ff_array = idx_analysis[:,4] # Fitting factor array
    # Appending the arrays that were read in with new data        
    ecc_array = np.append(ecc_array, eccentricity)
    op_snr_array = np.append(op_snr_array, op_snr)
    m_snr_array = np.append(m_snr_array, m_snr)
    mismatch_array = np.append(mismatch_array, mismatch)
    ff_snr_array = np.append(ff_array, fitting_factor)
    # Combining and transposing the arrays that were appended
    analysis_data = np.vstack((ecc_array, op_snr_array, m_snr_array, mismatch_array, ff_snr_array)).T
    analysis_data = analysis_data.astype(np.float)
    
    # Creating the name for the file to be written in the hdf5 file
    outfile = w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)
    with h5py.File('//local/path/to/HinderEIMR_SEOBNRv4_comparison.hdf5', 'w') as f: # Writing to the hdf5 file that was previously read in 
        outfile = f.create_dataset(outfile, data = analysis_data, dtype='f8') # Making an 8 bit float-type dataset in the hdf5 file 

'''
Printing out the final arrays from the above computations
'''
with h5py.File('/local/path/to/HinderEIMR_SEOBNRv4_comparison.hdf5', 'r') as read_analysis_file: # Reading the hdf5 file
    idx_analysis = read_analysis_file[w1_name + "_" + w2_name + "_{}_{}_{}_{}_{}_.dat".format(m1, m2, distance, s1z, s2z)] # Reading a specific key from the file
    ecc_array = idx_analysis[:,0] # Eccentricity array
    op_snr_array = idx_analysis[:,1] # Optimal SNR array
    m_snr_array = idx_analysis[:,2] # Measured SNR array
    mismatch_array = idx_analysis[:,3] # Mismatch array
    ff_array = idx_analysis[:,4] # Fitting factor array
    print('Final Eccentricity Array: ', ecc_array)
    print('Final Optimal SNR Array: ', op_snr_array)
    print('Final Measured SNR Array: ', m_snr_array)
    print('Final Mismatch Array: ', mismatch_array)
    print('Final Fitting Factor Array: ', ff_snr_array)
    
    plt.plot(ecc_array, ff_array, label = 'Fitting Factor')
    plt.xlabel('Eccentricity')
    plt.title(str(w1_name) + ' vs. ' + str(w2_name))
    plt.legend(loc = 'best')
    plt.ylabel('Fitting Factor')
    plt.savefig(str(w1_name) + '_' +  str(w2_name) + '_ff.png')
    plt.close()
    plt.plot(ecc_array, mismatch_array, label = 'Mismatch')
    plt.xlabel('Eccentricity')
    plt.title(str(w1_name) + ' vs. ' + str(w2_name))
    plt.legend(loc = 'best')
    plt.ylabel('Mismatch')
    plt.savefig(str(w1_name) + '_' + str(w2_name) + '_mm.png')    
    plt.close()
    plt.plot(ecc_array, op_snr_array, label = 'Optimal SNR')
    plt.xlabel('Eccentricity')
    plt.title(str(w1_name) + ' vs. ' + str(w2_name))
    plt.legend(loc = 'best')
    plt.ylabel('Optimal SNR')
    plt.savefig(str(w1_name) + '_' + str(w2_name) + '_op_snr.png') 
    plt.close()
    plt.plot(ecc_array, m_snr_array, label = 'Measured SNR')
    plt.xlabel('Eccentricity')
    plt.title(str(w1_name) + ' vs. ' + str(w2_name))
    plt.legend(loc = 'best')
    plt.ylabel('Measured SNR')
    plt.savefig(str(w1_name) + '_' + str(w2_name) + '_m_snr.png') 
    plt.close()

###