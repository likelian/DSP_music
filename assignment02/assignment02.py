
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time

##############Question 1: Time Domain Convolution: [30]#########################

#If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ?
#y = len(x) + len(y) - 1 = 299

def myTimeConv(x,h):
    """
    Computes the sample by sample time domain convolution of two signals.
    'x' and 'h' are the signal and impulse response as NumPy arrays.
    The convolution output "y" is a single channel NumPy array.
    """
    y = np.zeros(len(x)+len(h)-1) #empty array
    x = np.pad(x, (0, len(y) - len(x)), 'constant') #zeropadding
    h = np.pad(h, (0, len(y) - len(h)), 'constant')[::-1] #zeropadding

    idx = 0
    while idx < len(y):
        y[idx] = np.multiply(x[:idx+1], h[len(h)-idx-1:]).sum() #convolution
        idx += 1
    return y

x = np.ones(200) #DC offset signal
h = signal.triang(51) #symmetric triangular signal

y_time = myTimeConv(x, h)

plt.xlabel('time (samples)')
plt.ylabel('amplitude')
plt.plot(y_time)
plt.savefig('results/myTimeConv.png')


#############Question 2. Compare with SciPy Implementation: [20]################

def loadSoundFile(filename):
    """
    takes the filename and outputs a numpy array of floats
    if the file is multichannel, it grab the left channel
    """
    samplerate, data = wavfile.read(filename)
    if len(data.shape) == 2: data = data[:, 0]
    buffer = data.astype(np.float32)
    max_int16 = 2**15
    buffer_normalized = buffer / max_int16
    return buffer_normalized


def CompareConv(x, h):
    """
    compares the output of myTimeConv() with SciPy convolve()
    'x' and 'h' are the signal and impulse response as NumPy arrays.

    output: m, mabs, stdev, time
    m: float of the mean difference of the output compared to convolve()
    mabs: float of the mean absolute difference
    stdev: float standard deviation of the difference
    time: 2-lengthed array containing the running time of each method.
    """
    global time
    start = time.time()
    freqConv = signal.convolve(x, h)
    end = time.time()
    freqTime = end - start

    start = time.time()
    timeConv = myTimeConv(x, h)
    end = time.time()
    timeTime = end - start

    time = np.array([freqTime, timeTime])

    m = np.mean(freqConv - timeConv) #mean difference
    mabs = np.mean(np.abs(freqConv - timeConv)) #mean absolute difference
    stdev = np.std(freqConv - timeConv) #standard deviation of the difference

    return m, mabs, stdev, time


filename_x = "sounds/piano.wav"
filename_y = "sounds/impulse-response.wav"

x_buffer = loadSoundFile(filename_x)
h_buffer = loadSoundFile(filename_y)

m, mabs, stdev, time = CompareConv(x_buffer, h_buffer)
results = ("mean difference:", m,
    "mean absolute difference:", mabs,
    "standard deviation:", stdev,
    "running time of SciPy convolve() and myTimeConv():", time)

#report the output in the results folder
text_file = "results/Q2_results.txt"
f = open(text_file, "w")
for i in results:
    f.write(str(i)+"\n")
f.close()
