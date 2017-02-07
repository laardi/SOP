import wave
from yaafelib import *

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
    
    print (features)
    pass
      
def take_wav(fileloc):
    audio = wave.open(fileloc)
    fr = audio.getframerate()
    return audio, fr

def main():
    fileloc = '/tmp/wav/aanipankki_mono/kissa2.wav'
    a, b = take_wav(fileloc)
    zcr(fileloc, b)
    
if __name__ == "__main__":
    main()