from fft_functions import *
import time
import os


#Variables
extra_fact = 5
cut_peak = 3.5
knot1, knot2 = 60,120
x = [peak_constraint, harmonics_constraint, lower_freq, upper_freq, lowest_freq] = [4.0, 3.0, 1, 12.0, 1]
bools = [huge_gap, has_peaks, good_peak, longer_period] = [False, False, False, False]

script_dir = os.path.dirname(os.path.abspath(__file__))
figurepath = os.path.join(script_dir, "Plots") #destination of where to save graph
information_path = os.path.join(figurepath,  "information.txt") #saves all the information for the files

information_file = open(information_path, 'w')

epicname = 1000
count = 0
start_time1 = time.clock()


start_time = time.clock() #start the clock

#Part I - Define the parameters

time_int = 29.4/(60.0*24.0) # distance between points
a, b = 0, 100  #a,b : min and max of time
N = np.round((b-a) / time_int) #float
time_cad = np.linspace(a, b, N, endpoint = False) #hours

sigma = 0.05 #scatter
x0 = [sigma]

delta = .1 # depth    # (Rplanet / Rstar)**2
t0 = 0.7 # time of transit 
period = 15.0/24.0 # orbital period in hours
dur = 0.1
x1 = [sigma, period, t0, delta, dur]

delta2 = 0.05
x2 = [sigma, period, t0, delta, dur, delta2]

t2 = 0.12
x3 = [sigma, period, t0, delta, t2, delta2]
time1 = time.clock() - start_time


# Part II - Get flux values
flux_cad = multiclassflux(time_cad, 1, x1)
time2 = time.clock() - time1 - start_time


# Part III - FFT and FFT Normalization
freq, power, n, bin_sz, peak_width = fft_part(time_cad, flux_cad, extra_fact)
power_rel = fft_normalize(freq, power, n, bin_sz, cut_peak, knot1, knot2)
time3 = time.clock() - time2 - start_time
print "time3", time3

# Part IV - find the indexes of the peaks
inds = peak_indexes, harmonics_indexes, original_peak_indexes, original_harmonics_indexes = peak_finder(x, freq, power_rel)
time4 = time.clock() - time3 - start_time
print "time4", time4

# Part V - find the relevant index,freq,period
relevant_index, relevant_freq, relevant_period, bools[1:], potential_arr, rel_power_sums  = find_freq(inds, n, freq, power_rel)
huge_gap, has_peaks, good_peak, longer_period = bools
time5 = time.clock() - time4 - start_time
print "time5", time5

# Part VI - return (and save) the figure
fig = get_figure(time_cad, flux_cad, bools, inds, x, freq, power_rel, power, n, relevant_index, relevant_freq, relevant_period, epicname)
time6 = time.clock() - time5- start_time
print "time6", time6
fig.savefig(os.path.join(figurepath, "image50.png"), dpi = 50)
time7 = time.clock() - time6- start_time
print "time7", time7

#Part VII - return information about the data
info = get_info(bools, inds, epicname, relevant_freq, relevant_period)
print info
information_file.write("%s\n" % info)

stop_time = time.clock() - start_time
print stop_time, "seconds"
count+=1

stop_time1 = time.clock() - start_time1
print "Total time:", stop_time1, "seconds"
print "Time per LC:", stop_time1/count, "seconds"

#information_file.close()
#interest_file.close()