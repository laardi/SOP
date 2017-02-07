

import wave
from yaafelib import Engine
from yaafelib import FeaturePlan
from yaafelib import AudioFileProcessor
from numpy import zeros
from numpy import abs

class Audio():
    def __init__(self):
        pass

    def pre_process(self, wav):
        try:
            self.wav = wave.open(wav)
        except IOError:
            return False
        except wave.Error:
            # TODO
            # print if debug set
            print("Invalid file format %s" % wav)

            return False

        self.sample_rate = self.wav.getframerate()
        self.filename = wav

        return True

    def process(self):
        pass

    def classify(self):
        pass

    def zcr(self):
        return self._zcr(self.filename, self.sample_rate)["ZCR"]

    def zcr_avg(self):
        zcr = self._zcr(self.filename, self.sample_rate)["ZCR"]
        zcr_avg = 0
        for i in zcr:
            zcr_avg += i

        return zcr_avg / len(zcr)

    def _zcr(self, audio_location, sample_rate):
        # This function behave the same as 'python yaafe.py -r SAMPLERATE -f \
        #               "zcr: ZCR blockSize=1024  stepSize=512" WAV-LOCATION'
        # SAMPLERATE = samplerate of the file being processed
        # zcr        = name for the process (zcr1, zcr2... )
        # ZCR        = the feature that is being extracted
        # blockSize  = output frames size
        # stepSize   = step between consecutive frames

        # Build a dataflow object using FeaturePlan
        # blockSize, stepSize could be added too. 1024, 512 default
        fp = FeaturePlan(sample_rate=sample_rate)

        # Using *.addFeature() multiple extractions can be called with a
        # single call
        fp.addFeature('ZCR: ZCR')

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
        
        # Clear the engine so it can be used again
        #engine.reset()
        
        # returns the array of features extracted
        return features
        
    def mfcc(self):
        return self._mfcc(self.filename, self.sample_rate)["mfcc"]
        
    def _mfcc(self, audio_location, sample_rate):
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
        fp = FeaturePlan(sample_rate=sample_rate)

        # Using *.addFeature() multiple extractions can be called with a
        # single call
        fp.addFeature('mfcc: MFCC')
        #('mfcc: MFCC CepsIgnoreFirstCoeff=0 \
        #CepsNbCoeffs=13 FFTWindow=Hanning MelMaxFreq=6854\
        #MelMinFreq=130 MelNbFilters=40 blockSize=1024 stepSize=512')

        # Get dataflow
        df = fp.getDataFlow()

        engine = Engine()
        engine.load(df)

        # extract features from audio using AudioFileProcessor
        afp = AudioFileProcessor()
        afp.processFile(engine, audio_location)

        # features array holds all the extracted features
        features = engine.readAllOutputs()

        # returns the array of features extracted
        return features
        
    def SpectralFlux(self):
        return self._SpectralFlux(self.filename, self.sample_rate)["Flux"] 
        
    def _SpectralFlux(self, audio_location, sample_rate):
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
        fp.addFeature('Flux: SpectralFlux')
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
        
    def energy(self):
        return self._energy(self.filename, self.sample_rate)["Energy"] 
    
    def _energy(audio_location, sample_rate):
        # This function behaves the same as 'python yaafe.py -r SAMPLERATE -f \
        #               "energy: Energy PARAMETERS" WAV-LOCATION'
        # SAMPLERATE : Samplerate of the file being processed 
        #- blockSize (default=1024): output frames size
        #- stepSize (default=512): step between consecutive frames

        # Build a dataflow object using FeaturePlan
        # blockSize, stepSize could be added too. 1024, 512 default
        fp = FeaturePlan(sample_rate=sample_rate)

        # Using *.addFeature() multiple extractions can be called with a
        # single call
        fp.addFeature('Energy: Energy')
        #('energy: Energy blockSize=1024 stepSize=512')

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
        
    def ExtractAll(self):
        return self._ExtractAll(self.filename, self.sample_rate)
        
    def _ExtractAll(self, audio_location, sample_rate):
        # Build a dataflow object using FeaturePlan
        fp = FeaturePlan(sample_rate=sample_rate)

        # Using *.addFeature() multiple extractions can be called with a
        # single call
        fp.addFeature('zcr: ZCR')
        fp.addFeature('mfcc: MFCC')
        fp.addFeature('mfcc_D1: MFCC > Derivate DOrder=1')
        #fp.addFeature('mfcc_D2: MFCC > Derivate DOrder=2')
        fp.addFeature('flux: SpectralFlux')
        fp.addFeature('energy: Energy')
        fp.addFeature('loudness: Loudness')
        fp.addFeature('obsi: OBSI')
        fp.addFeature('sharpness: PerceptualSharpness')
        fp.addFeature('spread: PerceptualSpread')
        fp.addFeature('rolloff: SpectralRolloff')
        fp.addFeature('variation: SpectralVariation')
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
        #print features["zcr"]
        # returns the array of features extracted
        return features
    def ExtractZCRAndFlux(self):
        return self._ExtractZCRAndFlux(self.filename, self.sample_rate)

    def _ExtractZCRAndFlux(self, audio_location, sample_rate):
        fp = FeaturePlan(sample_rate=sample_rate)
        fp.addFeature('zcr: ZCR')
        fp.addFeature('flux: SpectralFlux')
        df = fp.getDataFlow()
        engine = Engine()
        engine.load(df)
        afp = AudioFileProcessor()
        afp.processFile(engine, audio_location)
        features = engine.readAllOutputs()
        return features
        



    def rhythm_patterns():
        # calculate fft window-size
        fft_size = 2**(nextpow2(matrix.shape[1]))

        rhythm_patterns = zeros((matrix.shape[0], fft_size), dtype=complex128)

        # calculate fourier transform for each bark scale
        for b in range(0,matrix.shape[0]):

            rhythm_patterns[b,:] = fft(matrix[b,:], fft_size)
                
            # normalize results
            rhythm_patterns = rhythm_patterns / 256

            # take first 60 values of fft result including DC component
            feature_part_xaxis_rp = range(0,60)    

            rp = abs(rhythm_patterns[:,feature_part_xaxis_rp])
