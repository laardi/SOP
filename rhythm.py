import numpy as np
import scipy


from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
#from scikits.talkbox              import segment_axis
#from scikits.talkbox.features     import mfcc
import matplotlib.pyplot as plt
from numpy.lib                    import stride_tricks



def zero_crossing_rate_BruteForce(wavedata):
    
    zero_crossings = 0
    
    for i in range(1, number_of_samples):
        
        if ( wavedata[i - 1] <  0 and wavedata[i] >  0 ) or \
           ( wavedata[i - 1] >  0 and wavedata[i] <  0 ) or \
           ( wavedata[i - 1] != 0 and wavedata[i] == 0):
                
                zero_crossings += 1
                
    zero_crossing_rate = zero_crossings / float(number_of_samples - 1)

    return zero_crossing_rate

def zero_crossing_rate(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)

def root_mean_square(wavedata, block_length, sample_rate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    rms = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))
        rms.append(rms_seg)
    
    return np.asarray(rms), np.asarray(timestamps)

def spectral_centroid(wavedata, window_size, sample_rate):
    
    magnitude_spectrum = stft(wavedata, window_size)
    
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sc = []

    for t in range(timebins-1):
        
        power_spectrum = np.abs(magnitude_spectrum[t])**2
        
        sc_t = np.sum(power_spectrum * np.arange(1,freqbins+1)) / np.sum(power_spectrum)
        
        sc.append(sc_t)
        
    
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    
    return sc, np.asarray(timestamps)

def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sr = []

    spectralSum    = np.sum(power_spectrum, axis=1)
    
    for t in range(timebins-1):
        
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]
        
        sr.append(sr_t)
        
    sr = np.asarray(sr).astype(float)
    
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)
    
    return sr, np.asarray(timestamps)

def spectral_flux(wavedata, window_size, sample_rate):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2, axis=1)) / freqbins
    
#---------------------------------------------------------------
def sadf():
    # MFCC STUFF
    # Pre-emphasis filter.

    # Parameters
    nwin    = 256
    nfft    = 1024
    fs      = 16000
    nceps   = 13

    # Pre-emphasis factor (to take into account the -6dB/octave
    # rolloff of the radiation at the lips level)
    prefac  = 0.97

    # MFCC parameters: taken from auditory toolbox
    over    = nwin - 160

    filtered_data = lfilter([1., -prefac], 1, input_data)

    windows     = hamming(256, sym=0)
    framed_data = segment_axis(filtered_data, nwin, over) * windows

    # Compute the spectrum magnitude
    magnitude_spectrum = np.abs(fft(framed_data, nfft, axis=-1))
    # Compute triangular filterbank for MFCC computation.

    lowfreq  = 133.33
    linsc    = 200/3.
    logsc    = 1.0711703
    fs = 44100

    nlinfilt = 13
    nlogfilt = 27

    # Total number of filters
    nfilt    = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs            = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights          = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    filterbank  = np.zeros((nfilt, nfft))

    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nfilt):
        
        low = freqs[i]
        cen = freqs[i+1]
        hi  = freqs[i+2]

        lid    = np.arange(np.floor(low * nfft / fs) + 1,
                           np.floor(cen * nfft / fs) + 1, dtype=np.int)

        rid    = np.arange(np.floor(cen * nfft / fs) + 1,
                           np.floor(hi * nfft / fs)  + 1, dtype=np.int)

        lslope = heights[i] / (cen - low)
        rslope = heights[i] / (hi - cen)

        filterbank[i][lid] = lslope * (nfreqs[lid] - low)
        filterbank[i][rid] = rslope * (hi - nfreqs[rid])

    # apply filter
    mspec = np.log10(np.dot(magnitude_spectrum, filterbank.T))

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    MFCCs = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

#.----------------------------------- -----------------------------------------------------------------
def rhythm_shit(data, fs):
    # ANNETAAN TIEDOSTO
    # data = aanitiedosto WAV-muodossa
    # fs   = aanitiedoston samplerate 

    # Parameters
    skip_leadin_fadeout = 1
    step_width          = 3

    segment_size        = 2**18
    fft_window_size     = 1024   # for 44100 Hz

    # Pre-calculate required values

    #duration  =  data.shape[0]/fs

    # calculate frequency values on y-axis (for bark scale calculation)
    freq_axis = float(fs)/fft_window_size * np.arange(0,(fft_window_size/2) + 1)

    # modulation frequency x-axis (after 2nd fft)
    mod_freq_res  = 1 / (float(segment_size) / fs) # resolution of modulation 
                                                   # frequency axis (0.17 Hz)             
    mod_freq_axis = mod_freq_res * np.arange(257)  # modulation frequencies along
                                                   # x-axis from index 1 to 257)

    fluct_curve   = 1 / (mod_freq_axis/4 + 4/mod_freq_axis)

    skip_seg = skip_leadin_fadeout

    seg_pos = np.array([1, segment_size])

    if ((skip_leadin_fadeout > 0) or (step_width > 1)):
        
     #   if (duration < 45):
      #      
            # if file is too small, don't skip leadin/fadeout and set step_width to 1
       #     step_width = 1
        #    skip_seg   = 0
        
        #else:
        seg_pos = seg_pos + segment_size * skip_seg; # advance by number of skip_seg segments (i.e. skip lead_in)
            
    # values verified

    wavsegment = data[seg_pos[0]-1:seg_pos[1]]
    print "a",wavsegment,"b"
    
    wavsegment = 0.0875 * wavsegment * (2**15)

    #plot(wavsegment);

    # [S1] spectrogram: real FFT with hanning window and 50 % overlap

    # number of iterations with 50% overlap
    n_iter = wavsegment.shape[0] / fft_window_size * 2 - 1  
    w      = np.hanning(fft_window_size)

    spectrogr = np.zeros((fft_window_size/2 + 1, n_iter))

    idx = np.arange(fft_window_size)

    # stepping through the wave segment, 
    # building spectrum for each window
    for i in range(n_iter): 
        
        spectrogr[:,i] = periodogram(x=wavsegment[idx], win=w)
        idx = idx + fft_window_size/2

    Pxx = spectrogr

    return Pxx, freq_axis

def bark_scale(Pxx, freq_axis):
    # ----------------- bark scale ----------------------
    # border definitions of the 24 critical bands of hearing
    bark = [100,   200,  300,  400,  510,  630,   770,   920, 
            1080, 1270, 1480, 1720, 2000, 2320,  2700,  3150,
            3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

    eq_loudness = np.array(
        [[ 55,   40,  32,  24,  19,  14, 10,  6,  4,  3,  2,  
            2,    0,  -2,  -5,  -4,   0,  5, 10, 14, 25, 35], 
         [ 66,   52,  43,  37,  32,  27, 23, 21, 20, 20, 20,  
           20,   19,  16,  13,  13,  18, 22, 25, 30, 40, 50], 
         [ 76,   64,  57,  51,  47,  43, 41, 41, 40, 40, 40,
         39.5, 38,  35,  33,  33,  35, 41, 46, 50, 60, 70], 
         [ 89,   79,  74,  70,  66,  63, 61, 60, 60, 60, 60,  
           59,   56,  53,  52,  53,  56, 61, 65, 70, 80, 90], 
         [103,   96,  92,  88,  85,  83, 81, 80, 80, 80, 80,  
           79,   76,  72,  70,  70,  75, 79, 83, 87, 95,105], 
         [118,  110, 107, 105, 103, 102,101,100,100,100,100,  
           99,   97,  94,  90,  90,  95,100,103,105,108,115]])

    loudn_freq = np.array(
        [31.62,   50,  70.7,   100, 141.4,   200, 316.2,  500, 
         707.1, 1000,  1414,  1682,  2000,  2515,  3162, 3976,
         5000,  7071, 10000, 11890, 14140, 15500])

    # calculate bark-filterbank
    loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))

    i = 0
    j = 0

    for bsi in bark:

        while j < len(loudn_freq) and bsi > loudn_freq[j]:
            j += 1
        
        j -= 1
        
        if np.where(loudn_freq == bsi)[0].size != 0: # loudness value for this frequency already exists
            loudn_bark[:,i] = eq_loudness[:,np.where(loudn_freq == bsi)][:,0,0]
        else:
            w1 = 1 / np.abs(loudn_freq[j] - bsi)
            w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
            loudn_bark[:,i] = (eq_loudness[:,j]*w1 + eq_loudness[:,j+1]*w2) / (w1 + w2)
        
        i += 1

    matrix = np.zeros((len(bark),Pxx.shape[1]))

    barks = bark[:]
    barks.insert(0,0)

    for i in range(len(barks)-1):

        matrix[i] = np.sum(Pxx[((freq_axis >= barks[i]) & (freq_axis < barks[i+1]))], axis=0)

    return matrix, bark

def spectral_masking(matrix, bark):

    #------- SPREADING FUNCTION FOR SPECTRAL MASKING ------------------
    # CONST_spread contains matrix of spectral frequency masking factors
    n_bark_bands = len(bark)

    CONST_spread = np.zeros((n_bark_bands,n_bark_bands))

    for i in range(n_bark_bands):
        CONST_spread[i,:] = 10**((15.81+7.5*((i-np.arange(n_bark_bands))+0.474)-17.5*(1+((i-np.arange(n_bark_bands))+0.474)**2)**0.5)/10)
        
    spread = CONST_spread[0:matrix.shape[0],:]
    matrix = np.dot(spread, matrix)
    
    return matrix


def decimal_scale(matrix):
    #map to decimal scale
        

    matrix[np.where(matrix < 1)] = 1

    matrix = 10 * np.log10(matrix)

    return matrix

def phon_scale(matrix):
    # Transfer to PHON SCALE
    # phon-mappings
    phon = [3, 20, 40, 60, 80, 100, 101]


    # number of bark bands, matrix length in time dim
    n_bands = matrix.shape[0]
    t       = matrix.shape[1]

    # DB-TO-PHON BARK-SCALE-LIMIT TABLE
    # introducing 1 level more with level(1) being infinite
    # to avoid (levels - 1) producing errors like division by 0


    table_dim = n_bands; 
    cbv       = np.concatenate((np.tile(np.inf,(table_dim,1)), 
                                loudn_bark[:,0:n_bands].transpose()),1)

    phons     = phon[:]
    phons.insert(0,0)
    phons     = np.asarray(phons) 

    # init lowest level = 2
    levels = np.tile(2,(n_bands,t)) 

    for lev in range(1,6): 
        db_thislev = np.tile(np.asarray([cbv[:,lev]]).transpose(),(1,t))
        levels[np.where(matrix > db_thislev)] = lev + 2

    # the matrix 'levels' stores the correct Phon level for each datapoint
    cbv_ind_hi = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-1]), order='F') 
    cbv_ind_lo = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-2]), order='F') 

    # interpolation factor % OPT: pre-calc diff
    ifac = (matrix[:,0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])

    ifac[np.where(levels==2)] = 1 # keeps the upper phon value;
    ifac[np.where(levels==8)] = 1 # keeps the upper phon value;

    matrix[:,0:t] = phons.transpose().ravel()[levels - 2] + (ifac * (phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2])) # OPT: pre-calc diff
    return matrix

def rhythm_patterns(matrix):
    # calculate fft window-size
    fft_size = 2**(nextpow2(matrix.shape[1]))

    rhythm_patterns = np.zeros((matrix.shape[0], fft_size), dtype=np.complex128)

    # calculate fourier transform for each bark scale
    for b in range(0,matrix.shape[0]):

        rhythm_patterns[b,:] = fft(matrix[b,:], fft_size)
        
    # normalize results
    rhythm_patterns = rhythm_patterns / 256

    # take first 60 values of fft result including DC component
    feature_part_xaxis_rp = range(0,60)    

    rp = np.abs(rhythm_patterns[:,feature_part_xaxis_rp])

    return rp


def periodogram(x,win,Fs=None,nfft=1024):
        
    if Fs == None:
        Fs = 2 * np.pi
   
    U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
    Xx = fft((x * win),nfft) # verified
    P  = Xx*np.conjugate(Xx)/U
    
    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    
    # Generate the one-sided spectrum [Power] if so wanted
    if nfft % 2 != 0:
        select = np.arange((nfft+1)/2)  # ODD
        P_unscaled = P[select,:] # Take only [0,pi] or [0,pi)
        P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
    else:
        select = np.arange(nfft/2+1);    # EVEN
        P = P[select]         # Take only [0,pi] or [0,pi) # todo remove?
        P[1:-2] = P[1:-2] * 2
    
    P = P / (2* np.pi)

    return P


def nextpow2(num):
    n = 2 
    i = 1
    while n < num:
        n *= 2 
        i += 1
    return i
