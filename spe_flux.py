import wave
from yaafelib import *


def _SpectralFlux(audio_location, sample_rate):
    # This function behaves the same as 'python yaafe.py -r SAMPLERATE -f \
    #               "flux: SpectralFlux PARAMETERS" WAV-LOCATION'
    # SAMPLERATE : Samplerate of the file being processed 
    # - FFTLength (default=0): Frame's length on which perform FFT. Original 
    #   frame is padded with zeros or truncated to reach this size. If 0 then
    #   use original frame length.
    # - FFTWindow (default=Hanning): Weighting window to apply before fft. Hanning|Hamming|None
    # - FluxSupport (default=All): support of flux computation. if 'All' then 
    #   use all bins (default), if 'Increase' then use only bins which are increasing
    # - blockSize (default=1024): output frames size
    # - stepSize (default=512): step between consecutive frames

    # Build a dataflow object using FeaturePlan
    fp = FeaturePlan(sample_rate=sample_rate)

    # Using *.addFeature() multiple extractions can be called with a
    # single call
    fp.addFeature('flux: SpectralFlux')
    #('flux: SpectralFlux FFTLength=0 FFTWindow=Hanning FluxSupport=All\
    # blockSize=1024 stepSize=512')

    # Get dataflow
    df = fp.getDataFlow()

    # Configure engine
    engine = Engine()
    engine.load(df)

    # extract features from audio using AudioFileProcessor
    afp = AudioFileProcessor()
    afp.processFile(engine, audio_location)

    # features array holds all the extracted features
    features = engine.readAllOutputs()
    
    # returns the array of features extracted
    return features
        
def take_wav(fileloc):
    audio = wave.open(fileloc)
    fr = audio.getframerate()
    return audio, fr

def main():
    fileloc = '/tmp/wav/aanipankki_mono/kissa2.wav'
    a, b = take_wav(fileloc)
    results = _SpectralFlux(fileloc, b)
    print (results)
    
    
if __name__ == "__main__":
    main()