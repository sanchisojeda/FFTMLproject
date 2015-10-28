import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import os
import csv
from scipy.interpolate import LSQUnivariateSpline

def multiclassflux(t, opt, x):
    """ Takes a time array and a parameter array to output a flux array
        
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


def constraint_index_finder(constraint, y):
    """ 
    Parameters:
    constraint = the number times mean of y
    y =  y values -- power
    Return value: 
    val_indexes = array with indexes where y[indexes] output values greater than constraint * mean value
    What it does:
    The mean of the array is multiplied with the constraint, and the index of any value in y (power) 
    that is above constraint*mean value is stored to an array --> That array is then returned
    """
    return np.where(y >= constraint*np.median(y))[0]
    
def max_peaks(index_arr, arr):
    """ Returns an array of the indexes where the indexes indicate where the peaks are"""
    return [index_arr[i] for i in range(1, len(index_arr)-1) if arr[index_arr[i-1]]<arr[index_arr[i]] and arr[index_arr[i+1]]<arr[index_arr[i]]]
    
def limit_applier(arr, lower_limit = 1.0, upper_limit = 10.0):
    """ Takes an array and a lower and upper limit and returns an array of the indices that 
        are below the lower limit or higher than the upper limit -- it is the delete_arr"""
    return np.append(np.where(arr < lower_limit)[0], np.where(arr > upper_limit)[0])

################################################################################
############################# read in data #####################################
################################################################################

def read_files(path):
    return os.listdir(path)
    
def read_data(filename):
    # ex. filename = path_of_file + "LCfluxesepic201637175star00.txt" 
    # ex. epicname = "201637175"
    return np.transpose(np.loadtxt(filename)) # == data -- data[0] = time, data[1] = flux

def read_csv(filename): #mainly for Ruth's FFT
    spamreader = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='|')
    #assuming that there is a title row
    lines = np.transpose(list(spamreader)[1:]) #lines[0] = freq, lines[1] = power; lines[0][0] = "Frequency"
    return np.asarray([float(i) for i in lines[0]]), np.asarray([float(i) for i in lines[1]])
    
################################################################################
################################################################################
## Determine the appropriate cadence to which the data should be interpolated ##
######################### Get the GAP array ####################################
################################################################################
################################################################################

def gap_filler(data):
    t, f = data[0], data[1]
    num_time = len(t) # the original number of data time points (w gaps)
        
    delta_t = t[1:num_time]-t[0:num_time-1] # has a length = len(t) - 2 (bc it ignores endpoints)
    cadence = np.median(delta_t) # what the time interval between points with gaps currently is
    #Fill gaps in the time series (according to 2012 thesis method)
    gap = np.append(np.around(delta_t/cadence), 1).astype(int)
        
    gap_cut = 1.1   #Time differences greater than gap_cut*cadence are gaps    
    gap_loc = [i for i in np.arange(len(gap)) if (gap[i] > gap_cut)]
    num_gap, biggest_gap = len(gap_loc), np.amax(gap) # the number of gaps present 
    
    ################################################################################
    ########## Create a new time and flux array to account for the gaps ############
    ################################################################################
    
    num_cad = sum(gap)  # the number of data points
    time_cad = np.arange(num_cad)*cadence + np.amin(t)
    flux_cad = np.empty(num_cad) 
            
    oldflux_st, newflux_st = 0, 0 # this is an index
    
    for n in np.arange(num_gap):
        oldflux_ed = gap_loc[n]
        gap_sz     = gap[gap_loc[n]]
        newflux_ed = newflux_st + (oldflux_ed-oldflux_st)
        
        flux_cad[newflux_st:newflux_ed] = f[oldflux_st:oldflux_ed]
            
        gap_fill = (np.ones(gap_sz-1))*f[oldflux_ed]
        flux_cad[newflux_ed+1:newflux_ed+gap_sz] = gap_fill
        flux_cad[newflux_ed] = np.mean([flux_cad[newflux_ed-1], flux_cad[newflux_ed+1]])
        
        oldflux_st = oldflux_ed + 1
        newflux_st = newflux_ed + gap_sz    
    
    #account for last part where there is no gap after
    flux_cad[newflux_st:num_cad] = f[oldflux_st:num_time]
    
    return time_cad, flux_cad, biggest_gap/len(time_cad) >= .2
    
################################################################################
################################################################################
#################################### FFT part ##################################
################################################################################
################################################################################

def fft_part(time_cad, flux_cad, extra_fact):
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
    # first fit
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
############################### Peak Limits ####################################
################################################################################   
################################################################################

def peak_finder(x, freq, power_rel):
    
    peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq = x
    
    val_indexes = constraint_index_finder(peak_constraint, power_rel) #peaks
    val_indexes1 = constraint_index_finder(harmonics_constraint, power_rel) #harmonics

    peak_indexes = max_peaks(val_indexes, power_rel)
    harmonics_indexes = max_peaks(val_indexes1, power_rel)
    
    # keep all of the original peak_indexes/harmonics_indexes to check later on 
    # if it's a longer period planet
    if lower_freq != lowest_freq:
        original_peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes],lowest_freq,upper_freq))
        original_harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes],lowest_freq,upper_freq))
    else:
        original_peak_indexes, original_harmonics_indexes = np.empty(0), np.empty(0)
    
    # we only want peaks that are between freqs of [lower_freq, upper_freq] cycles/day
    peak_indexes = np.delete(peak_indexes, limit_applier(freq[peak_indexes],lower_freq,upper_freq))
    harmonics_indexes = np.delete(harmonics_indexes, limit_applier(freq[harmonics_indexes],lower_freq,upper_freq))
    
    return peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes
    
################################################################################
################################################################################
############ Determining potential periods based on the FFT ####################
################################################################################
################################################################################
 
def find_freq(inds, n, freq, power_rel):
    
    peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = inds
    # potential arr has the format of a list of lists where the 0th index of each elem 
    # of potential_arr is the peak index and any other numbers in elem are harmonic indexes for
    # the peak index
    # ex. [ [1000, 2001, 3004, 4008], [2001, 4008], [3004], [4008] ] 
    potential_arr = []
    for elem in peak_indexes:
        number = len(harmonics_indexes) + 1
        poss_indexes = np.arange(2, number) * elem - 1 #possible indexes
        poss_indexes_lower, poss_indexes_upper = poss_indexes - n, poss_indexes + n # lower/upper bound of possible indexes
        poss_indexes_bound = np.transpose(np.array([poss_indexes_lower, poss_indexes_upper]))
        #attempt to do the nested for loop in a line - works, but is much slower
        #temp_arr = [elem]+[elem1 for elem1 in harmonics_indexes for lower,upper in poss_indexes_bound if elem1>=lower and elem1<=upper elif lower>elem1 break]
        #print temp_arr
        temp_arr = [elem]
        for elem1 in harmonics_indexes:
            for lower, upper in poss_indexes_bound:
                if elem1 >= lower and elem1 <= upper:
                    temp_arr.append(elem1)
                    break # break because no elem1 will only satisfy this condition
                elif lower > elem1:
                    break # won't ever satisfy the condition

        potential_arr.append(temp_arr)
        
    # rel_power_sums is a list where each element is the sum of relative power 
    # for each elem in potential_arr -- sum is 0 if elem of potential_arr is only the peak index
    rel_power_sums = [np.sum(power_rel[elem]) if len(elem) != 1 else 0 for elem in potential_arr]
            
    # booleans
    has_peaks, longer_period = len(peak_indexes) > 0, False
    # good peak means that there is at least one peak with one other harmonic
    good_peak = has_peaks and np.amax(rel_power_sums) > 0
       
################################################################################
################# checking if it might be longer period planet #################
################################################################################

    if has_peaks and good_peak:
        relevant_index = potential_arr[np.argmax(rel_power_sums)][0]
        ### this 'for loop' goes through the peaks whose freq are [lowest_freq,upper_freq]
        ### and checks to see if one of those freqs could potentially be
        ### the relevant freq
        potential_indexes_longer_period = [] # indicies of potential revelant freqs
        for elem in original_peak_indexes: 
            number = len(harmonics_indexes) + 1
            poss_indexes = 1. / np.arange(2, number) * relevant_index - 1
            poss_indexes_lower, poss_indexes_upper = poss_indexes - n, poss_indexes + n # lower/upper bound of possible indexes
            poss_indexes_bound = np.transpose(np.array([poss_indexes_lower, poss_indexes_upper]))
            #attempt to do the nested for loop in a line - works, but is much slower
            #potential_indexes_longer_period = [elem for elem in original_peak_indexes for lower,upper in poss_indexes_bound if elem>=lower and elem<=upper]
            for lower, upper in poss_indexes_bound:
                if elem >= lower and elem <= upper:
                    potential_indexes_longer_period.append(elem)
                    break # break because elem only satisfies this condition
                elif upper > elem1:
                    break # won't ever satisfy the condition
                    
        ### this for loop goes through the freqs in the interval [lowest_freq,upper_freq] 
        ### that may potentially be the relevant freq
        ### follows exact algorithm as the for loop that goes through each peak
        ### in peak_indexes and tries to see if they are potentially good freq

        potential_arr1 =[]
        for elem in potential_indexes_longer_period:
            number = len(original_harmonics_indexes) + 1
            poss_indexes = np.arange(2, number) * elem - 1
            poss_indexes_lower, poss_indexes_upper = poss_indexes - n, poss_indexes + n # lower/upper bound of possible indexes
            poss_indexes_bound = np.transpose(np.array([poss_indexes_lower, poss_indexes_upper]))
            #attempt to do the nested for loop in a line - works, but is much slower
            #temp_arr = [elem]+[elem1 for elem1 in harmonics_indexes for lower,upper in poss_indexes_bound if elem1>=lower and elem1<=upper]
            temp_arr = [elem]
            for elem1 in harmonics_indexes: # or original_harmonics_indexes?
                for lower, upper in poss_indexes_bound:
                    if elem1 >= lower and elem1 <= upper:
                        temp_arr.append(elem1)
                        break # break because elem1 only satisfies this condition
                    elif lower > elem1:
                        break # won't ever satisfy the condition
            potential_arr1.append(temp_arr)
            
        if len(potential_arr1)>0:
            rel_power_sums1 = [np.sum(power_rel[elem]) if len(elem) != 1 else 0 for elem in potential_arr1]
            if np.amax(rel_power_sums1) > np.amax(rel_power_sums):
                longer_period, relevant_index = True, potential_arr1[np.argmax(rel_power_sums1)][0]
        
        relevant_freq = freq[relevant_index]
        #relevant_period = 24./relevant_freq # period is in hours
        relevant_period = relevant_freq**(-1) # period is in days
    else:
        relevant_index, relevant_freq, relevant_period= 0,0,0

    return relevant_index, relevant_freq, relevant_period, [has_peaks, good_peak, longer_period], potential_arr, rel_power_sums

################################################################################
################################################################################
############################ SAVING THE FIGURE #################################
################################################################################
################################################################################
def get_figure(time_cad, flux_cad, bools, inds, x, freq, power_rel, power, n, relevant_index, relevant_freq, relevant_period,epicname):
   
    huge_gap, has_peaks, good_peak, longer_period = bools
    peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq = x
    peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = inds
    
    fig = plt.figure(figsize=(20,15))
    
    ## Lightcurve (top left)
    ax1 = fig.add_subplot(221)
    ax1.scatter(time_cad, flux_cad, s=5, c='black')
    ax1.plot(time_cad, flux_cad, 'black', linewidth = .75)
    ax1.set_title("Lightcurve " + str(epicname), fontsize = 16)
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Numerical Flux")
    ax1.set_xticks(np.arange(np.round(np.amin(time_cad),-1), np.round(np.amax(time_cad),-1)+1, 5.0))
    ax1.set_xlim([np.amin(time_cad), np.amax(time_cad)])
    delta = np.amax(flux_cad) - np.amin(flux_cad)
    ax1.set_ylim([1 - 1.5 * delta, 1. + .5 * delta])
    ax1.grid(True)

    ## Original FFT (bottom left)
    ax2 = fig.add_subplot(223)
    ax2.plot(freq, power, 'black',linewidth = .75)
    if has_peaks:
        ax2.scatter(freq[peak_indexes], power[peak_indexes], s=30, c="black")
        ax2.set_title("Numfreq = " + str(len(peak_indexes)), fontsize = 16)
    else:
        ax2.set_title("NO PEAKS DETECTED")
    ax2.set_xlabel("Frequency (cycles/day)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim([0,upper_freq])
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(np.arange(upper_freq))
    ax2.grid(True)
    
    ## Normalized FFT (bottom right)
    ax3 = fig.add_subplot(224)
    ax3.plot(freq, power_rel,'black',linewidth = .75)
    if has_peaks and good_peak:
        ax3.scatter(freq[relevant_index], power_rel[relevant_index], c='black', s=50)
        ax3.set_title("PEAK FREQ = " + str(relevant_freq) + " Period: " + str(relevant_period) + " days", fontsize =16)
    else:
        ax3.set_title("NO PEAKS DETECTED")
    ax3.set_xlabel("Frequency (cycles/day)")
    ax3.set_ylabel("Relative Amplitude")
    ax3.set_xlim([0, upper_freq])
    ax3.set_ylim([0, 1.5*np.amax(power_rel[n:])])
    ax3.set_xticks(np.arange(upper_freq))
    ax3.grid(True)

    ### Folded Light Curve (top right)
    ax4 = fig.add_subplot(222)
    ax4.set_xlabel("Orbital phase")
    ax4.set_ylabel("Relative Flux")
    ax4.grid(True)
    if has_peaks and good_peak:
        phases, orbit = np.modf(time_cad/relevant_period)
        ax4.plot(phases,flux_cad,'k.')
        ax4.set_title("Folded light curve")
    else:
        ax4.set_title("Folded light curve - no peaks detected")
    plt.close(fig)
    return fig

################################################################################
################################################################################

def get_info(bools, inds, epicname, relevant_freq, relevant_period):
    
    huge_gap, has_peaks, good_peak, longer_period = bools
    peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = inds

    if not has_peaks:
        info = [epicname, 1, 0, 0]#, 'WARNING: no peaks detected'
    elif has_peaks and not good_peak:
        info = [epicname, 2, 0, 0]#, 'WARNING: a peak with no harmonics'
    elif has_peaks and good_peak and harmonics_indexes[0] < peak_indexes[0]:
        info = [epicname, 3, relevant_freq, relevant_period]#,'WARNING: there may be a harmonic peak that came before the first main peak'
    elif longer_period:
        info = [epicname, 4, relevant_freq, relevant_period]#,'WARNING: may be longer period'
    elif huge_gap:
        info = [epicname, 5, relevant_freq, relevant_period]#,'WARNING: huge gap'
    else:
        info = [epicname, 0, relevant_freq, relevant_period]# this one is normal
    return info
