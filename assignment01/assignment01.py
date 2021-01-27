from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

#######################################################
#Question1
def crossCorr(x, y):
    """
    compute the correlation between two arrays
    """
    return np.correlate(x, y, "full")


def loadSoundFile(filename):
    """
    takes the filename and outputs a numpy array of floats
    if the file is multichannel, it grab the left channel
    """
    samplerate, data = wavfile.read(filename)
    buffer = data[:, 0].astype(np.float32)
    max_int16 = 2**15
    buffer_normalized = buffer / max_int16
    return buffer_normalized


def main(filename_x, filename_y):
    """
    load the sound files and compute the correlation between them
    plotting the result to file results/01-correlation.png
    """
    x = loadSoundFile(filename_x)
    y = loadSoundFile(filename_y)
    z = crossCorr(x, y)
    plt.plot(z)
    plt.show()
    plt.savefig('results/01-correlation.png')



filename_x = "sounds/snare.wav"
filename_y = "sounds/drum_loop.wav"

main(filename_x, filename_y)


#######################################################
#Question2
def findSnarePosition(snareFilename, drumloopFilename):
    """
    Outputs a regular python list of sample positions of the best guess
    for the snare position in the drumloop
    Save the result in a text file results/02-snareLocation.txt
    """

    x = loadSoundFile(filename_x)
    y = loadSoundFile(filename_y)
    z = crossCorr(x, y)
    pos = []
    for idx, val in enumerate(z):
        if val > 150:
            pos.append(idx)

    text_file = "results/02-snareLocation.txt"
    f = open(text_file, "w")
    for i in pos:
        f.write(str(i)+"\n")
    f.close()
    return pos

findSnarePosition(filename_x, filename_y)
