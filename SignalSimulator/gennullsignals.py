from fft_functions import *
import time


#Variables
extra_fact = 1
cut_peak = 3.5
knot1, knot2 = 60,120
xnew = [lower_freq, upper_freq, numharms] = [1, 2.0, 6]

script_dir = os.path.dirname(os.path.abspath(__file__))
information_path = os.path.join(script_dir, "nullsignalssmall.csv") #saves all the information for the files

f1 = open(information_path, "wb")
planets = csv.writer(f1)
planets.writerow(["Noise_in", "Period_out", "A", "A1", "A2", "A3", "A4", "A5", "A6"])


information_file = open(information_path, 'w')

count = 0
start_time1 = time.clock()

numsim = int(np.round(25000*0.4*3.0))

minfact = 0.01
maxfact = 0.10

np.random.seed(200)
fact= np.random.rand(numsim)*(maxfact-minfact) +minfact


for i in range(0, numsim):
	#Part I - Define the parameters

	time_int = 5.0/(60.0*24.0) # distance between points
	a, b = 0, 100  #a,b : min and max of time
	N = np.round((b-a) / time_int) # number of points
	time_cad = np.linspace(a, b, N, endpoint = False) #days

	sigma = fact[i] #  noise

	# Part II - Get flux values, and SNR
	flux_cad = multiclassflux(time_cad, 0, sigma)

	# Part III - FFT and FFT Normalization
	freq, power, n, bin_sz, peak_width = fft_part(time_cad, flux_cad, extra_fact)
	power_rel = fft_normalize(freq, power, n, bin_sz, cut_peak, knot1, knot2)

	# # Part IV - find the the highest peak and the amplitudes at the harmonics
	simple_period, amplitudes = featureextraction(xnew, freq, power, power_rel)


	#Part VII - write the information and features for the signal
	entries = np.append(np.array([fact[i], simple_period]), amplitudes)
	planets.writerow(entries)

	count+=1


stop_time1 = time.clock() - start_time1
print "Total time:", stop_time1, "seconds"
print "Time per LC:", stop_time1/count, "seconds"

f1.close()
