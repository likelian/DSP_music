
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
    #create time steps
    t = np.arange(sampling_rate_Hz*length_secs)
    t /= sampling_rate_Hz
    #create the empty signal array
    x = np.zeros(int(sampling_rate_Hz * length_secs))
    #compute the signal
    x = amplitude * np.sin(2 * np.pi * frequency_Hz * t + phase_radians)

    return t, x

t_sin, x_sin = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2) #generate a sine wave

#Plot the first 5 ms of the sinusoid
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.plot(t_sin[:int(0.005*44100)], x_sin[:int(0.005*44100)])
plt.savefig('results/sine_time.png')
plt.clf()



######Question 2. Sinusoids to generate waveforms with complex spectra [25]#####
def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    """
    generates a square wave approximated with 10 sinusoidal
    The outputs x and t are the generated signal and the corresponding time in seconds.
    (NumPy arrays of the same length)
    """
    x = np.zeros(int(sampling_rate_Hz * length_secs)) #create the empty signal array

    n = 1
    while n <= 10:
        t_n, x_n = generateSinusoidal(amplitude/(2*n - 1), sampling_rate_Hz, frequency_Hz*(2*n - 1), length_secs, phase_radians)
        x += x_n #add each sinusoidal
        n += 1
    return t_n, x


t_square, x_square = generateSquare(1.0, 44100, 400, 0.5, 0)

#Plot the first 5 ms of the generated square waves in Part 2.2.
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.plot(t_square[:int(0.005*44100)], x_square[:int(0.005*44100)])
plt.savefig('results/square_time.png')
plt.clf()


##################Question 3. Fourier Transform [25]############################

def computeSpectrum(x, sample_rate_Hz):
    """
    computes the FFT of the complete input signal vector x
    returns:
        the magnitude spectrum XAbs
        the phase spectrum XPhase
        the real part XRe
        the imaginary part XIm
        the frequency of the bins f
    """
    #print(x)
    #print(len(x))
    X = np.fft.fft(x)[:int(len(x)/2)] #complex spectrum (half)
    XRe = abs(X)
    #XRe = 20 * np.log10(abs(X) / 8192)
    XIm = np.angle(X)
    XAbs = np.array([XRe, np.arange(len(XRe))*sample_rate_Hz/len(XRe)/2])
    XPhase = np.array([XIm, XAbs[1]])
    f = len(X)*2/sample_rate_Hz
    return f, XAbs, XPhase, XRe, XIm

f, XAbs_sin, XPhase_sin, XRe_sin, XIm_sin = computeSpectrum(x_sin, 44100) #spectrum in Question1.2
f, XAbs_square, XPhase_square, XRe_square, XIm_square = computeSpectrum(x_square, 44100) #spectrum in Question2.2

#Plot the magnitude and phase spectra for sine wave
fig_sin, axs_sin = plt.subplots(2)
fig_sin.suptitle('Sine')
axs_sin[0].plot(XAbs_sin[1], XAbs_sin[0])
axs_sin.flat[0].set(ylabel="magnitude")
axs_sin[1].plot(XPhase_sin[1], XPhase_sin[0])
axs_sin.flat[1].set(xlabel="frequency(Hz)", ylabel="phase")
plt.savefig('results/sin_frequency.png')
plt.clf()

#Plot the magnitude and phase spectra for square wave
fig_square, axs_square = plt.subplots(2)
fig_square.suptitle('Square')
axs_square[0].plot(XAbs_square[1], XAbs_square[0])
axs_square.flat[0].set(ylabel="magnitude")
axs_square[1].plot(XPhase_square[1], XPhase_square[0])
axs_square.flat[1].set(xlabel="frequency(Hz)", ylabel="phase")
plt.savefig('results/square_frequency.png')
plt.clf()


#######################Question 4. Spectrogram [30]#############################


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    """
    blocks a given signal x according to the given parameters

    The output t is a NumPy array containing the time stamps of the blocks,
    and X is a matrix (block_size x N) where each column is a block of the signal.

    (You may need to zero-pad the input signal appropriately for the last block.)
    """
    x_pad = np.append(x, np.zeros(hop_size - (len(x) - block_size) % hop_size)) #zero_padding
    t = np.zeros(int((len(x_pad) - block_size)/hop_size + 1)) #create the time stamp array
    X = np.zeros((len(t), block_size)) #create X matrix
    for i in range(len(X)):
        X[i] = x_pad[i * hop_size : i * hop_size + block_size]
        t[i] = i * hop_size
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

    time_vector, block = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    time_vector /= sampling_rate_Hz

    if window_type == "hann": block = np.multiply(block, np.hanning(block_size))

    magnitude_spectrogram = np.zeros((len(time_vector), int(block_size/2)))
    for i in range(len(block)-1):
        f, XAbs, XPhase, XRe, XIm = computeSpectrum(block[i], sampling_rate_Hz)
        magnitude_spectrogram[i] = XAbs[0]
    freq_vector = XAbs[1]
    return freq_vector, time_vector, magnitude_spectrogram


def plotSpecgram(freq_vector, time_vector, magnitude_spectrogram):
  if len(freq_vector) < 2 or len(time_vector) < 2:
    return

  Z = 20. * np.log10(magnitude_spectrogram)
  Z = np.flipud(Z)

  pad_xextent = (time_vector[1] - time_vector[0]) / 2
  xmin = np.min(time_vector) - pad_xextent
  xmax = np.max(time_vector) + pad_xextent
  extent = xmin, xmax, freq_vector[0], freq_vector[-1]
  plt.xlabel('time (seconds)')
  plt.ylabel('frequency (Hz)')
  im = plt.imshow(Z, None, extent=extent,
                           origin='upper')
  plt.axis('auto')
  plt.show()
  #plt.savefig('results/spectrogram(rect).png')
  #plt.savefig('results/spectrogram(hann).png')



#plot the magnitude spectrogram of the square wave using the rectangular window
freq_vector, time_vector, magnitude_spectrogram = mySpecgram(x_square, 2048, 1024, 44100, "rect")
plotSpecgram(freq_vector, time_vector, magnitude_spectrogram.transpose())

#plot the magnitude spectrogram of the square wave using the hann window
freq_vector, time_vector, magnitude_spectrogram = mySpecgram(x_square, 2048, 1024, 44100, "hann")
plotSpecgram(freq_vector, time_vector, magnitude_spectrogram.transpose())
