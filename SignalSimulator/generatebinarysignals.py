from fft_functions import *
import time


#Variables
extra_fact = 1
cut_peak = 3.5
knot1, knot2 = 60,120
xnew = [lower_freq, upper_freq, numharms] = [1, 2.0, 6]

script_dir = os.path.dirname(os.path.abspath(__file__))
information_path = os.path.join(script_dir, "binariessmall.csv") #saves all the information for the files

f1 = open(information_path, "wb")
planets = csv.writer(f1)
planets.writerow(["SNR", "Duration_in", "Noise_frac_in", "Period_in",  "Period_out", "A","A1", "A2", "A3", "A4", "A5", "A6"])


information_file = open(information_path, 'w')

count = 0
start_time1 = time.clock()

numsim = 25000


np.random.seed(300)
fact= np.random.lognormal(mean=1.0, sigma=0.09, size=numsim)

allperiods = np.random.rand(numsim)*(1.0/lower_freq - 1.0/upper_freq) + 1.0/upper_freq
allt0 = np.random.rand(numsim)

durfracmin = 0.05
durfracmax = 0.15

alldurfrac = np.random.rand(numsim)*(durfracmax-durfracmin)+durfracmin

mindepthfact = 0.3
maxdepthfact = 0.7

depthfact= np.random.rand(numsim)*(maxdepthfact-mindepthfact) +mindepthfact



for i in range(0, numsim):
	#Part I - Define the parameters

	time_int = 5.0/(60.0*24.0) # distance between points
	a, b = 0, 100  #a,b : min and max of time
	N = np.round((b-a) / time_int) # number of points
	time_cad = np.linspace(a, b, N, endpoint = False) #days

	sigma = 0.001 #scatter
	delta = sigma*fact[i] # depth    # (Rplanet / Rstar)**2
	t0 = allt0[i] # time of transit 
	period = allperiods[i] # orbital period in hours
	dur = alldurfrac[i]*period
	delta2 = delta*depthfact[i]
	x2 = [sigma, period, t0, delta, dur, delta2]


	# Part II - Get flux values
	flux_cad = multiclassflux(time_cad, 2, x2)
	snr = getSNR(sigma, flux_cad)


	# Part III - FFT and FFT Normalization
	freq, power, n, bin_sz, peak_width = fft_part(time_cad, flux_cad, extra_fact)
	power_rel = fft_normalize(freq, power, n, bin_sz, cut_peak, knot1, knot2)

	# # Part IV - find the the highest peak and the amplitudes at the harmonics
	simple_period, amplitudes = featureextraction(xnew, freq, power, power_rel)


	#Part VII - write the information and features for the signal
	entries = np.append(np.array([snr, alldurfrac[i], fact[i],  period, simple_period]), amplitudes)
	planets.writerow(entries)

	count+=1


stop_time1 = time.clock() - start_time1
print "Total time:", stop_time1, "seconds"
print "Time per LC:", stop_time1/count, "seconds"

f1.close()
