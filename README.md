# FFTMLproject
In this repository I share a series of tools designed to transform the problem of finding transiting exoplanets in time-series data into a more tractable machine learning classification problem. In this project I am focusing on planets with orbital periods shorter than 24 hours, so that the Fourier Transform technique can be used to speed up the process. So far only the simulations part of the project is advanced enough. In the SignalSimulator folder several python scripts can be found that do the following taks:

* "fft_functions.py": Contains the main functions used in these simulations. The functions are:
    * Multiclassflux: Is the most important function. Given a series of times, this function can simulate a signal that looks like a planet, an eclipsing binary, a classical pulsation (two sinusoids) or simply white noise.
    * fft_par: Obtains the Fast Fourier Transform of the time series
    * fft_normalize: Normalizes the fft using the scipy LSQUnivariateSpline routine
    * get_SNR: Obtains the Signal/Noise ratio of a signal created with the function Multiclassflux
    * Featureextraction: Obtains the most relevant features from the FFT of a signal: The main periodicity (from 12 to 24 hours by definition),the median amplitude of the first n harmonic peaks respect to the noise in the FFT, and the relative amplitude of each peak respect to median amplitude. 

* "genXXXsignals.py": A series of four scripts that generate the simulated signals, and that use the functions in "fft_functions.py" to extract the relevant features from each signal. The features of each simulated signal are stored in a csv file with a name similar to "xxxsmall.csv". XXX can be either planet, binary, sinu (sinusoid), null (for white noise).

*"initialSVMfits.py": Main code where the simulated signals for the four classes are combined into a big data frame. Low signal to noise simulations for each of the classes are assimilated into the null class, to generate the "no detection" class. The data is split into a training, cross validation and test sets and the features are reescaled using the preprocessed routine from sklearn. A radial basis function kernel is chosen for the SVM models, and the penalty parameter is chosen to be large based on trial and error (C=1000). The cross validation data set is chosen to evaluate the performance with different values of gamma. With or without the median relative amplitude as a feature, the code achieves accuracies of 99.9% within the simulations. Tests with real data must be performed to see the real accuracy of the code.