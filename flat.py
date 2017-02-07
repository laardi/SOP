def find_wavs(directory):
    wav_files = []

    for i in os.listdir(directory):
        i = os.path.join(directory, i)

        if valid_wav(i):
            wav_files.append(i)

        if os.path.isdir(i):
            wav_files += find_wavs(i)

    return wav_files

def main():
    for i in find_wavs('sounds'):
        print i
