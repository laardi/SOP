import wave
from yaafelib import *


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
    fp.addFeature('energy: Energy')
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
        
def take_wav(fileloc):
    audio = wave.open(fileloc)
    fr = audio.getframerate()
    return audio, fr

def main():
    fileloc = '/tmp/wav/aanipankki_mono/kissa2.wav'
    a, b = take_wav(fileloc)
    results = _energy(fileloc, b)
    print (results)
    
    
if __name__ == "__main__":
    main()