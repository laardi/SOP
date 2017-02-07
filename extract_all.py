import wave
from yaafelib import *

def _ExtractAll(audio_location, sample_rate):
    # Build a dataflow object using FeaturePlan
    fp = FeaturePlan(sample_rate=sample_rate)

    # Using *.addFeature() multiple extractions can be called with a
    # single call
    fp.addFeature('zcr: ZCR')
    fp.addFeature('mffc: MFCC')
    fp.addFeature('flux: SpectralFlux')

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
def _mfcc(audio_location, sample_rate):
    # This function behaves the same as 'python yaafe.py -r SAMPLERATE -f \
    #               "mfcc: MFCC PARAMETERS" WAV-LOCATION'
    # SAMPLERATE : Samplerate of the file being processed 
    #- CepsIgnoreFirstCoeff (default=1): 0 keeps the first cepstral coeffcient, 1 ignore it
    #- CepsNbCoeffs (default=13): Number of cepstral coefficient to keep.
    #- FFTWindow (default=Hanning): Weighting window to apply before fft. Hanning|Hamming|None
    #- MelMaxFreq (default=6854.0): Maximum frequency of the mel filter bank
    #- MelMinFreq (default=130.0): Minimum frequency of the mel filter bank
    #- MelNbFilters (default=40): Number of mel filters
    #- blockSize (default=1024): output frames size
    #- stepSize (default=512): step between consecutive frames

    # Build a dataflow object using FeaturePlan
    # blockSize, stepSize could be added too. 1024, 512 default
    fp = FeaturePlan(sample_rate=sample_rate)

    # Using *.addFeature() multiple extractions can be called with a
    # single call
    fp.addFeature('mfcc: MFCC')
    fp.addFeature('zrc: ZCR')
    #('mfcc: MFCC CepsIgnoreFirstCoeff=0 \
    #CepsNbCoeffs=13 FFTWindow=Hanning MelMaxFreq=6854\
    #MelMinFreq=130 MelNbFilters=40 blockSize=1024 stepSize=512')

    # Get dataflow
    df = fp.getDataFlow()

    # Or load it from a file
    # df = DataFlow()
    # df.load(dataflow_file)

    # Configure engine
    engine = Engine()
    engine.load(df)

    # extract features from audio using AudioFileProcessor
    afp = AudioFileProcessor()
    afp.processFile(engine, audio_location)

    # features array holds all the extracted features
    features = engine.readAllOutputs()

    # extract features from an audio file and write results to csv files
    # afp.setOutputFormat('csv','output',{'Precision':'8'})
    # afp.processFile(engine,audiofile)
    #  this creates output/myaudio.wav.mfcc.csv,
    #               output/myaudio.wav.mfcc_d1.csv and
    #               output/myaudio.wav.mfcc_d2.csv files.
    
    # Clear the engine so it can be used again | Is this needed?
    #engine.reset()
    
    # returns the array of features extracted
    return features
def zcr(audio, fr):
    #Build a dataflow object using FeaturePlan
    fp = FeaturePlan(sample_rate=fr)
    fp.addFeature('ZCR: ZCR')
    
    df = fp.getDataFlow()
    
    #Configure engine
    engine = Engine()
    engine.load(df)
    
    #extract features from audio using AudioFileProcessor
    afp = AudioFileProcessor()
    afp.processFile(engine,audio)
    
    features = engine.readAllOutputs()
    return features
    
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
    results = _ExtractAll(fileloc, b)
    #print (results)
    res1 = _SpectralFlux(fileloc, b)
    res2 = _mfcc(fileloc, b)
    res3 = zcr(fileloc, b)
    #print (res1, res2, res3)
    
    
if __name__ == "__main__":
    main()