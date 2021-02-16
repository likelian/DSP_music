
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

##################Question 1: Generating sinusoids [15]#########################

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    """
    generating a sinusoidal according to the parameters
    The outputs x and t are the generated signal and the corresponding time in seconds
    (NumPy arrays of the same length)
    """
    return t, x

t, x = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2) #generate a sine wave

#Plot the first 5 ms of the sinusoid
#label the axes correctly, time axis must be in seconds



######Question 2. Sinusoids to generate waveforms with complex spectra [25]#####
def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    """
    generates a square wave approximated with 10 sinusoidal
    (https://en.wikipedia.org/wiki/Square_wave)
    The outputs x and t are the generated signal and the corresponding time in seconds.
    (NumPy arrays of the same length)
    """
    #generateSinusoidal()
    return t, x

t, x = generateSquare(1.0, 44100, 400, 0.5, 0)
#Plot the first 5 ms of the generated square waves in Part 2.2.



##################Question 3. Fourier Transform [25]############################

"""


"""


def computeSpectrum(x, sample_rate_Hz):
    """
    computes the FFT of the complete input signal vector x
    returns:
        the magnitude spectrum XAbs
        the phase spectrum XPhase
        the real part XRe
        the imaginary part XIm
        the frequency of the bins f

    Return only the non-redundant part (without symmetry)
    """
    #You may use the NumPy fft function in order to compute the FFT
    return f,XAbs,XPhase,XRe,XIm


#f,XAbs,XPhase,XRe,XIm = computeSpectrum(x, sample_rate_Hz) #spectrum in Question1.2
#f,XAbs,XPhase,XRe,XIm = computeSpectrum(x, sample_rate_Hz) #spectrum in Question2.2

#Plot the magnitude and phase spectra for each signal (2 sub-plots for magnitude and phase in 1 plot)
#(label the axes correctly, frequency axis must be in Hz) (There will be two plots. One for each signal.)

"""
What is the frequency resolution (difference in frequency between the bins) of the FFT obtained above?

How will the frequency resolution change in this case if you zero-pad the input signal
with the same of zeros as the length of the input signal?

(Answer 3.4 & 3.5 in a text file in results folder)
"""


#######################Question 4. Spectrogram [30]#############################


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    """
    blocks a given signal x according to the given parameters

    The output t is a NumPy array containing the time stamps of the blocks,
    and X is a matrix (block_size x N) where each column is a block of the signal.

    (You may need to zero-pad the input signal appropriately for the last block.)
    """
    return t, X

def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    """
    computes the FFT per block windowed using the window type specified

    freq_vector and time_vector are both column vectors
    containing the frequency of the bins in Hz (block_size/2 x 1)
    and the time stamps of the blocks in seconds (N x 1)
    N is the number of blocks.

    magnitude_spectrogram is a (block_size/2 x N) matrix,
    where each column is the FFT of a signal block.

    ‘rect’ for a rectangular window, ‘hann’ for a Hann window.

    plot the magnitude spectrogram
    Note: You may use the NumPy fft, hanning and Matplotlib specgram methods.
    You can use the generateBlocks and computeSpectrum methods which you created earlier.
    """
    return freq_vector, time_vector, magnitude_spectrogram

#plot the magnitude spectrogram of the square wave using the rectangular window
#freq_vector, time_vector, magnitude_spectrogram = mySpecgram()

#plot the magnitude spectrogram of the square wave using the hann window
#freq_vector, time_vector, magnitude_spectrogram = mySpecgram()

#Take the block_size as 2048 and the hop_size as 1024. In a text file in results directory,
#compare the differences in the two plots due to the different windows used.









"""
Question 5. BONUS: Sine-Sweep (10 points, capped at 100 points)

How would you approach generating a sine sweep in the spectral domain using only a single spectrum (not a spectrogram).
Write a script that generates a sine-sweep in the frequency domain using one single spectrum.
You may not use the SciPy chirp function.
"""
