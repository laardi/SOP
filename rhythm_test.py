from scipy.io import wavfile
#import os
#import sys
#import rhythm.py as rhythm
import sys

from rhythm import rhythm_shit, bark_scale,spectral_masking,rhythm_patterns
samplerate, wavedata = wavfile.read(sys.argv[1])
#samplerate = wavedata.getframerate()
#print samplerate
#print wavedata
print "as",wavedata
asd, kek = rhythm_shit(wavedata, samplerate)
foo,bar = bark_scale(asd, kek)
#print foo
barista = spectral_masking(foo,bar)
#print bar
der = rhythm_patterns(barista)
for i in der:
    for j in i:
        print j
