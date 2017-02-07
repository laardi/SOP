
import os
import sys
import json
import argparse
import numpy

from audio_processing import Audio
from classifier import Classifier
from pprint import pprint

# NOTE
# 1. Use autocorrelation in classification
# https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar


VERBOSE = False
database_file = "classifier_db_alt.json"


def parse_args():
    parser = argparse.ArgumentParser(prog="bif", add_help=True)
    parser.add_argument("files", nargs="*",
                        help="Files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print more verbose output.")
    parser.add_argument("-l", "--list_database", action="store_true",
                        help="Print classifier database.")
    parser.add_argument("-d", "--database",
                        help="Select database to use.")
    parser.add_argument("-c", "--audio_class",
                        help="Create new audio class.")
    parser.add_argument("-r", "--remove_class",
                        help="Remove an audio class.")
                        
    args = parser.parse_args()

    if args.verbose:
        global VERBOSE
        VERBOSE = True

    return args


def find_wavs(directory):
    wav_files = []

    for i in os.listdir(directory):
        i = os.path.join(directory, i)

        if valid_wav(i):
            wav_files.append(i)

        if os.path.isdir(i):
            wav_files += find_wavs(i)

    return wav_files


def valid_wav(wav_file):
    if not os.path.isfile(wav_file):
        return False

    if not wav_file.endswith(".wav"):
        return False

    return True


def create_feature_list(features):
    audio_mfcc_mean = numpy.mean(features["mfcc"], axis = 0)
    audio_mfcc_D1_mean = numpy.mean(features["mfcc_D1"], axis = 0)
    #audio_mfcc_D1_mean = numpy.mean(audio_mfcc['mfcc_D1'], axis = 0)

    audio_mfcc_var = numpy.var(features["mfcc"], axis = 0)
    audio_mfcc_D1_var = numpy.var(features["mfcc_D1"], axis = 0)
    #audio_mfcc_D2_var = numpy.var(audio_mfcc['mfcc_D2'], axis = 0)

    zcr_avg = numpy.mean(features["zcr"])
    zcr_var = numpy.var(features["zcr"])
    flux_avg = numpy.mean(features["flux"])
    flux_var = numpy.var(features["flux"])

    feature_list = {}
    #feature_list["mfcc"] = numpy.concatenate([audio_mfcc_D1_mean,
    #                                  audio_mfcc_D1_var])
    feature_list["mfcc"] = {}
    feature_list["mfcc"]["D1_var"]  = list(audio_mfcc_D1_var)
    feature_list["mfcc"]["D1_avg"]  = list(audio_mfcc_D1_mean)
    #feature_list["mfcc"]["D2_var"]  = 
    #feature_list["mfcc"]["D2_avg"]  = 
    feature_list["zcr_avg"] = zcr_avg
    feature_list["zcr_var"] = zcr_var
    feature_list["flux_avg"] = flux_avg
    feature_list["flux_var"] = flux_var
    return feature_list


def main():
    if len(sys.argv) == 1:
        print("help")

    args = parse_args()
    wavs = []
    global database_file
    if args.database:
        database_file = args.database

    classifier = Classifier(database_file)

    
    if not os.path.isfile(database_file):
        # No database exists, so create a new one with
        # basic classes.
        database = {}
        #database["natural"] = {}
        database["music"] = {}
        database["speech"] = {}
        #database["non-natural"] = {}
        database["other"] = {}

        for i in database.keys():
            database[i]["metadata"] = {}
        # data["metadata"] = {}
        
        with open(database_file, "w") as f:
            json.dump(database, f)

    else:
        with open(database_file, "r") as f:
            database = json.load(f)

    if args.list_database:
        pprint(database)
        
    elif args.remove_class:
        main_class, sub_class = args.remove_class.split("/")
        if sub_class in database[main_class].keys():
            # REmove 
            database[main_class].pop(sub_class)
            print "Removed %s from %s."%(sub_class, main_class)
            with open(database_file, "w") as f:
                json.dump(database, f)
        else:
            print "%s not found in %s."%(sub_class, main_class)
        

    elif args.files:
        if args.audio_class:
            classifier.create_class(args.audio_class, args.files)

            for i in args.files:
                with open("classes/" + args.audio_class + "/files", "a") as f:
                    f.write(i)
                    f.write("\n")

        for wav in args.files:
            if os.path.isdir(wav):
                for i in find_wavs(wav):
                    new_audio = Audio()
                    if new_audio.pre_process(i):
                        wavs.append(new_audio)

            else:
                if not valid_wav(wav):
                    print("%s not a valid wav file." % wav)
                    continue

                new_audio = Audio()
                if new_audio.pre_process(wav):
                    print(wav)
                    wavs.append(new_audio)

        if VERBOSE:
            print("Found %d valid wav files." % len(wavs))

        zcr_avg_all = []
        zcr_var_all = []
        flux_avg_all = []
        flux_var_all = []
        mfcc_list = []
        for audio_file in wavs:
            features = audio_file.ExtractAll()
            feature_list = create_feature_list(features)
            #asd = [item for sitem in features["zcr"] for item in sitem]
            #a = numpy.correlate(asd, asd, mode="full")[-len(asd):]
            #print(numpy.abs(a))
            #print len(feature_list)
            #classifier.classify_DTW(features_list)

            zcr_avg_all.append(feature_list["zcr_avg"])
            zcr_var_all.append(feature_list["zcr_var"])
            flux_avg_all.append(feature_list["flux_avg"])
            flux_var_all.append(feature_list["flux_var"])
            mfcc_list.append(feature_list["mfcc"])

            #print feature_list
          

        # NOTE
        # Temporary "if"
        if args.audio_class:
            json_data = {}
            if len(wavs) > 1:
                json_data["zcr_avg"] = numpy.mean(zcr_avg_all)
                json_data["zcr_var"] = numpy.mean(zcr_var_all)
                json_data["flux_avg"] = numpy.mean(flux_avg_all)
                json_data["flux_var"] = numpy.mean(flux_var_all)
            else:
                json_data["zcr_avg"] = zcr_avg_all[0]
                json_data["zcr_var"] = zcr_var_all[0]
                json_data["flux_avg"] = flux_avg_all[0]
                json_data["flux_var"] = flux_var_all[0]
            
            #json_data["mfcc"] = mfcc_list
            dump_json(database, json_data, args.audio_class, len(wavs))
        else:
            # Luokitteluvertailu
            # Tunnistettiin luokka x, dumpataanko? dump_json(json_data, audio_class)
            pass
            
        
    else:
        print("help")


def dump_json(data, new_data, audio_class, count):
    main_class, sub_class = audio_class.split("/")

    #with open(database_file, "r") as f:
    #    data = json.load(f)

    if sub_class not in data[main_class].keys():
        # If there is no subclass with this name
        data[main_class][sub_class] = {}
        data[main_class][sub_class]["metadata"] = new_data
        data[main_class][sub_class]["count"] = count

    else:
        # Metadata of an existing subclass is compared to new data, and updated
        subclass_metadata = data[main_class][sub_class]["metadata"]
        subclass_metadata_count = data[main_class][sub_class]["count"]
        data[main_class][sub_class]["count"] += count
        
        #print "Original data: ",data[main_class][sub_class]["metadata"]

        #a = subclass_metadata["flux_avg"]
        #data[main_class][sub_class]["metadata"]["flux_avg"] = (a * (subclass_metadata_count) + count * new_data["flux_avg"])/(subclass_metadata_count+count)
        #print "Version late: ",subclass_metadata["flux_avg"]
        
        for i in data[main_class][sub_class]["metadata"].keys():
            if i == "mfcc":
                continue
            print i,subclass_metadata[i],new_data[i],data[main_class][sub_class]["count"]
            data[main_class][sub_class]["metadata"][i] = (subclass_metadata[i] * (subclass_metadata_count) + count * new_data[i])/(subclass_metadata_count+count)
        print "<<<<<<<<<<<<<<<<"
        #print "New data: ",data[main_class][sub_class]["metadata"]
        # New JSON values
        
    with open(database_file, "w") as f:
        json.dump(data, f)


def update_json(data, audio_class):
    pass


main()
