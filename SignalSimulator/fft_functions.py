import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import os
import csv
from scipy.interpolate import LSQUnivariateSpline

def multiclassflux(t, opt, x):
    """ Takes a time array and a parameter array to output a simulated flux array from a given class
        
    INPUTS: 
        
    t: time array where to evaluate the flux

    opt: variable that controles the type of signal to output.
        0: White noise  (Null class)
        1: Transits of constant depth (Planet class)
        2: Two transits per orbit, with different depths (Binary class)
        3: Two sinusoids, one at Porb and one at Porb/2 (Pulsation class)

    x: parameter array
        x[0] = sigma: amount of scatter   (class 0-3)
        x[1] = period: orbital period in units of time  (class 1-3)
        x[2] = t0: time of primary transit (class 1-2) phase of main sinusoid (class 3)
        x[3] = delta: primary eclipse depth (class 1-2) amplitude of main sinusoid (class 3)
        x[3] = dur: primary eclipse duration (class 1-2) phase of P/2 sinusoid (class 3)
        x[4] = delta2: secondary eclipse depth (class 2) amplitude of P/2 sinusoid (class 3)
        
    OUTPUT:
    
    A flux series that represents the selected model evaluate at t, with a white noise component
    
    """
    
    if opt == 0:
        sigma = x
        return sigma *np.random.random_sample(len(t))
    elif opt == 1:
        sigma, period, t0, delta, dur = x
        
        flux = sigma *np.random.random_sample(len(t))
        
        orbital_phase = (t-t0)/period # orbital phase = 0 at t0
        m = np.round(orbital_phase) # takes the integer part of the array
        t_m = t0 + m*period# t(m) = t0 + m(t) * period, time of transit at each orbit
        
        inside = np.abs(t-t_m) < dur/2.0
        
        flux[inside] = flux[inside] - delta
        
        return flux
    elif opt == 2:
        sigma, period, t0, delta1, dur, delta2 = x
        
        flux = sigma *np.random.random_sample(len(t))
        
        orbital_phase = (t-t0)/period # orbital phase = 0 at t0
        m = np.round(orbital_phase) # takes the integer part of the array
        t_m = t0 + m*period# t(m) = t0 + m(t) * period, time of transit at each orbit
        
        inside1 = np.abs(t-t_m) < dur/2.0
        
        flux[inside1] = flux[inside1] - delta1
        
        inside21 = np.abs(t+0.5*period-t_m) < (dur/2.0)
        flux[inside21] = flux[inside21] - delta2
        
        inside22 = np.abs(t-0.5*period-t_m) < (dur/2.0)
        flux[inside22] = flux[inside22] - delta2
        
        return flux
    else:
        sigma, period, t1, A1, t2, A2 = x
        
        flux = sigma *np.random.random_sample(len(t))

        phase1 = (t-t1)/period 
        phase2 = (t-t2)/(period/2.0) 
        
        return flux + A1*np.sin(2.0*np.pi*phase1) + A2*np.sin(2.0*np.pi*phase2)


    
################################################################################
################################################################################
#################################### FFT part ##################################
################################################################################
################################################################################

def fft_part(time_cad, flux_cad, extra_fact):
    """ Takes a time array, a series of fluxes, and an oversampling factor and computes the fft
        
    INPUTS: 
        
    time_cad: time array where the flux is obtained
    flux_cad: flux array with flux measurements
    extra_fact: Oversampling factor

        
    OUTPUT:
    
    A flux series that represents the selected model evaluate at t, with a white noise component
    
    freq: Frequencies where the FFT is evaluated, in 1/units of time_cad
    power: The power of the FFT power spectrum
    n: Is the new number of data points, which is equal to 2^(round(log2(N_orig)) + extra_fact), 
        where N_orig is the number of points of time_cad
    bin_sz: is the final distance between consecutive frequencies in the FFT
    peak_width: the expected total width of a peak representing a delta function

    """

    # oversampling
    N = len(time_cad)
    N_log = np.log2(N) # 2 ** N_log = N
    exp = np.round(N_log)
    if exp < N_log:
        exp += 1 #compensate for if N_log was rounded down
    
    newN = 2**(exp + extra_fact)
    n = np.round(newN/N)
    diff = newN - N
    mean = np.median(flux_cad)
    voidf = np.zeros(diff) + mean
    newf = np.append(flux_cad, voidf)
    
    norm_fact = 2.0 / newN # normalization factor 
    f_flux = fft(newf) * norm_fact
        
    freq = fftfreq((len(newf)))
    d_pts = (np.amax(time_cad) - np.amin(time_cad)) / (N-1)
    freq_fact = 1.0 / d_pts #frequency factor 
    freq *= freq_fact
    
    postivefreq = freq > 0 # take only positive values
    freq, f_flux = freq[postivefreq], f_flux[postivefreq]
    
    power = np.abs(f_flux)
    
    bin_sz = 1./len(newf) * freq_fact # distance between consecutive points in cycles per day
    peak_width = 2 * bin_sz * 2**extra_fact #in cycles per day
    return freq, power, n, bin_sz, peak_width

################################################################################
########################### Normalizing the FFT ################################
################################################################################

def fft_normalize(freq, power, n, bin_sz, cut_peak, knot1, knot2):
    """ Takes the values of fft at given frequencies and normalizes the power spectrum using a second order spline
        
    INPUTS: 
        
    freq: frequencies where the power of the FFT spectrum has been evaluated
    power: power of the FFT spectrum evaluated at freq
    n: Number of points in freq
    bin_sz: Distance between consecutive frequencies
    cut_peak: Maximum relative amplitude for a given peak to be used in second and final spline fit
    knot1: Knot distance for the preliminary second order spline with the peaks on it
    knot2: Knot distance for the final second order spline where the peaks have been removed
        
    OUTPUT:
    
    power_rel: The FFT spectrum normalized using a second order spline fit and normalized by its median.
    """
    
    knot_w = knot1*n*bin_sz # difference in freqs (cycles/day) between each knot
    first_knot_i = np.round(knot_w/bin_sz) #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_i = len(freq) - first_knot_i #index of the last knot is the first point that is knot_w away from the last value of x
    knots = np.arange(freq[first_knot_i], freq[last_knot_i],knot_w)
    spline = LSQUnivariateSpline(freq, power, knots, k=2) #the spline, it returns the piecewise function
    fit = spline(freq) #the actual y values of the fit
    if np.amin(fit) < 0:
        fit = np.ones(len(freq))
    pre_power_rel = power/fit
    
    # second fit -- by deleting the points higher than cut_peak times the average value of power
    pre_indexes = constraint_index_finder(cut_peak, power)
    power_fit = np.delete(pre_power_rel, pre_indexes)
    freq_fit = np.delete(freq, pre_indexes)
    
    knot_w1 = knot2*n*bin_sz
    first_knot_fit_i = np.round(knot_w1/bin_sz) #index of the first knot is the first point that is knot_w away from the first value of x 
    last_knot_fit_i = len(freq_fit) - first_knot_fit_i#index of the last knot is the first point that is knot_w away from the last value of x
    knots_fit = np.arange(freq_fit[first_knot_fit_i], freq_fit[last_knot_fit_i], knot_w1)
    spline = LSQUnivariateSpline(freq_fit,power_fit,knots_fit, k=2) #the spline, it returns the piecewise function
    fit3 = spline(freq) #the actual y values of the fit applied to freq
    if np.amin(fit3) < 0:
        fit3 = np.ones(len(freq))
    
    # relative power 
    power_rel = pre_power_rel / fit3    
    return power_rel / np.median(power_rel) # so the median of power_rel is 1
    
################################################################################
################################################################################
############################### Feature extraction #############################
################################################################################   
################################################################################


def getSNR(sigma, flux):
    """ Function that obtains the Signal to noise of a given time-series signal

    Input: 

    sigma: Standard deviation of the gaussian used to generate the noise of the flux series
    flux: The flux series

    Output:

    SNR: The signal to noise as defined as the square root of the difference between the 
    chi^2 cost function using a median-model and the expected value for only white noise,
    equal to the number of degrees of freedom (the length of flux)

    """
    newchi2 = np.sum( (flux-np.median(flux))**2)/sigma**2
    oldchi2 = len(flux)
    if newchi2 <= oldchi2:
        return 0
    else:
        return np.sqrt(newchi2-oldchi2)

def featureextraction(x, freq, power, power_rel):
    """ Function that retrieves the relevant features of the FFT of given time-series signal

    Input:
    x parameter vector
        x[0] = lower_freq, the lowest frequency searched (assumed as 1.0 cycles/day)
        x[1] = upper_freq, the highest frequency searched for the main peak (assumed as less than 2xlower_freq)
        x[2] = numharms, number of harmonics for which we want to know the amplitude of the FFT

    freq: The frequencies where the FFT is evaluated
    power: The FFT power spectrum with no normalization
    power_rel: The normalized FFT power spectrum

    Output:
    feature vector feat:
        feat[0] = the highest peak period between lower_freq and upper_freq
        feat[1] = the mean relative amplitude of the first nharm harmonics (~ SNR)
        feat[2:nharms-1] = the relative amplitude of each individual peak 
    """
    lower_freq, upper_freq, numharms = x

    dfreq = freq[0] # distance between frequencies
    indexlow = np.floor(lower_freq/dfreq)#position of the low limit
    indexhigh = np.floor(upper_freq/dfreq) #position of the upper limit

    indexmax = np.argmax(power[indexlow:indexhigh+1]) + indexlow

    amplitudes = [np.max(power[indexmax*i-10:indexmax*i+10]) for i in np.arange(1, numharms+1)]    
    amplitudes_rel = [np.max(power_rel[indexmax*i-10:indexmax*i+10]) for i in np.arange(1, numharms+1)]    


    return 1.0/freq[indexmax], np.append(np.mean(amplitudes_rel), amplitudes/np.mean(amplitudes))
