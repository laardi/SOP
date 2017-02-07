
import os
import sys
import json
import argparse
import time
from numpy import var
from numpy import mean
import pydot
#import graphviz
#from itertools import combinations
import itertools

from audio_processing import Audio
from classifier import Classifier
from pprint import pprint
#from graphviz import Digraph

# NOTE
# 1. Use autocorrelation in classification
# https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar


VERBOSE = False
DEMO = False
database_file = "database.json"
SAMPLE_LIMIT = 20

def parse_args():
    parser = argparse.ArgumentParser(prog="bif", add_help=True)

    parser.add_argument("files", nargs="*", help="Files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print more verbose output.")
    parser.add_argument("-l", "--list_database", action="store_true", help="Print classifier database.")
    parser.add_argument("-d", "--database", help="Select database to use.")
    parser.add_argument("-r", "--remove_class", help="Remove an audio class.")
    parser.add_argument("-f", "--features",  help="Audio features")
    parser.add_argument("-p", "--plot", help="Plot metadata of classes.")
    parser.add_argument("-t", "--target", help="Expected target class")
    parser.add_argument("-a", "--add_samples", help="Add sample(s) to database")
    parser.add_argument("-D", "--print-database-info", action="store_true", help="Prints information about the database")
    parser.add_argument("-n", "--neighbors-knn", help="Set neighbors for knn")
    parser.add_argument("-c", "--convert", help="Convert hierarchical database to flat.")
    parser.add_argument("-w", "--draw_classes", help="Draw class hierarchy to png.")
    parser.add_argument("-b", "--test_db", action="store_true", help="Test all samples in database.")
    parser.add_argument("-s", "--test_features", action="store_true", help="Test all feature combinations.") 
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


def calculate_features(features):
    mfcc_mean = mean(features["mfcc"], axis = 0)
    mfcc_D1_mean = mean(features["mfcc_D1"], axis = 0)
    mfcc_var = var(features["mfcc"], axis = 0)
    mfcc_D1_var = var(features["mfcc_D1"], axis = 0)

    for i, j in enumerate(features["flux"]):
        if j > 500:
            features["flux"][i] = 0
    #TODO: korjaa flux

    zcr_avg = mean(features["zcr"])
    zcr_var = var(features["zcr"])
    flux_avg = mean(features["flux"])
    flux_var = var(features["flux"])
    energy_avg =  mean(features["energy"])
    energy_var =  var(features["energy"])
    loudness_avg =  mean(features["loudness"])
    loudness_var =  var(features["loudness"])
    obsi_avg =  mean(features["obsi"])
    obsi_var =  var(features["obsi"])
    sharpness_avg =  mean(features["sharpness"])
    sharpness_var =  var(features["sharpness"])
    spread_avg =  mean(features["spread"])
    spread_var =  var(features["spread"])
    rolloff_avg =  mean(features["rolloff"])
    rolloff_var =  var(features["rolloff"])
    spectral_variations_avg =  mean(features["variation"])
    spectral_variations_var =  var(features["variation"])


    feature_list = {}
    feature_list["mfcc"] = mfcc_D1_mean.tolist() + mfcc_D1_var.tolist()
    feature_list["zcr_avg"] = zcr_avg
    feature_list["zcr_var"] = zcr_var
    feature_list["flux_avg"] = flux_avg
    feature_list["flux_var"] = flux_var
    feature_list["energy_avg"] = energy_avg
    feature_list["energy_var"] = energy_var
    feature_list["loudness_avg"] = loudness_avg
    feature_list["loudness_var"] = loudness_var
    feature_list["obsi_avg"] = obsi_avg
    feature_list["obsi_var"] = obsi_var
    feature_list["sharpness_avg"] = sharpness_avg
    feature_list["sharpness_var"] = sharpness_var
    feature_list["spread_avg"] = spread_avg
    feature_list["spread_var"] = spread_var
    feature_list["rolloff_avg"] = rolloff_avg
    feature_list["rolloff_var"] = rolloff_var
    feature_list["spectral_variations_avg"] = spectral_variations_avg
    feature_list["spectral_variations_var"] = spectral_variations_var
    
    return feature_list

def DEMO_feature_list(features):
    # FOR DEMO USE ONLY
    for i, j in enumerate(features["flux"]):
        if j > 500:
            features["flux"][i] = 0

    zcr_avg = mean(features["zcr"])
    zcr_var = var(features["zcr"])
    flux_avg = mean(features["flux"])
    flux_var = var(features["flux"])
    feature_list = {}
    feature_list["mfcc"] = [] 
    feature_list["zcr_avg"] = zcr_avg
    feature_list["zcr_var"] = zcr_var
    feature_list["flux_avg"] = flux_avg
    feature_list["flux_var"] = flux_var
    feature_list["energy_avg"] = [] 
    feature_list["energy_var"] = []
    return feature_list

def main():
    ts = time.time()
    if len(sys.argv) == 1:
        print("help")

    args = parse_args()
    wavs = []
    global database_file
    if args.database:
        database_file = args.database

    if  not os.path.isfile(database_file):
        # No database exists, so create a new
        database = {}

        with open(database_file, "w") as f:
            json.dump(database, f)


    with open(database_file, "r") as f:
        database = json.load(f)

    classifier = Classifier(database)
    if args.features:
        for j in args.features.split(","):
            classifier.used_features.append(j)

    #if VERBOSE:
    #    classifier.VERBOSE = True

    if args.list_database:
        pprint(database)

    if args.convert:
        flat_database = {}
        ignore = ["count","samples", "filenames"]

        def flat_conv(data, node):
            for i, j in data[node].items():
                if i == "samples" and j:
                    flat_database[node] = data[node] 
                elif i not in ignore:
                    flat_conv(data[node], i)
        
        for i in database.keys():
            flat_conv(database, i)
        #print flat_database
            
           # if i == "metadata":
           #     continue

           # for j in database[i].keys():
           #     if j == "metadata":
           #         continue
           #     flat_database[j] = database[i][j]

           #     for k in database[i][j].keys():
           #         if k == "metadata":
           #             continue
           #         flat_database[k] = database[i][j][k]

        with open(args.convert, "w") as f:
            json.dump(flat_database, f)
 
    if args.print_database_info:
        mc = 0
        nc = 0
        sc = 0

        for i in database.keys():
            mc += 1

            for j in database[i].keys():
                if j == "metadata":
                    continue
                nc += 1
                print database[i][j]

                for k in database[i][j].keys():
                    if k != "samples":
                        continue

                    for n in database[i][j][k]:
                        if n == "metadata":
                            continue
                        sc += 1

        print("Main classes:\t%d" % mc)
        print("Sub classes:\t%d" % nc)
        print("Samples:\t%d" % sc)
        sys.exit(0)

    elif args.plot:
        plot_file = args.plot
        plot_metadata(database, plot_file)
    
    elif args.remove_class:
        
        joink =  args.remove_class.split("/")
        #main_class, sub_class, sub_sub = args.remove_class.split("/")
        main_class = joink[0]
        sub_class = joink[1]
        try:
            sub_sub = joink[2]
        except IndexError:
            sub_sub = False
        if sub_sub:
            if sub_sub in database[main_class][sub_class].keys():
                database[main_class][sub_class].pop(sub_sub)
                print "Removed %s from %s/%s."%(sub_sub, main_class,sub_class)
                with open(database_file, "w") as f:
                    json.dump(database, f)


        elif sub_class in database[main_class].keys():
            # REmove 
            database[main_class].pop(sub_class)
            print "Removed %s from %s."%(sub_class, main_class)
            with open(database_file, "w") as f:
                json.dump(database, f)
        else:
            print "%s not found in %s."%(sub_class, main_class)
        

    elif args.files:
        if args.add_samples:
            classifier.create_class_structure(args.add_samples, args.files)

            files = args.files[0]
            print args.files
            if os.path.isdir(files):
                files = [i for i in find_wavs(files)]
            
            for i in files:
                listed = False

                if os.path.isfile("classes/" + args.add_samples + "/files"):
                    with open("classes/" + args.add_samples + "/files", "r") as f:
                        for line in f.readlines():
                            if i in line:
                                listed = True
                                break
                    if not listed:
                        with open("classes/" + args.add_samples + "/files", "a") as f:
                            f.write(i)
                            f.write("\n")

		else:
			with open("classes/" + args.add_samples + "/files", "a") as f:
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

        samplelist = []

        # extract & calculate features:
        for audio_file in wavs:
            features = audio_file.ExtractAll()
            features = calculate_features(features)
            audio_file.features = features

            #asd["flux_avg"] = _features["flux_avg"]
            #asd["flux_var"] = _features["flux_var"]
            #asd["zcr_avg"] = _features["zcr_avg"]
            #asd["zcr_var"] = _features["zcr_var"]
            #asd["mfcc"] = _features["mfcc"]
            #asd["flux_var"] = _features["flux_var"]
            #audio_file.features = calculate_features(_features)
            #samplelist.append(feature_list)
            #print "samplelist: ", samplelist

        if VERBOSE:
            print("Found %d valid wav files." % len(wavs))

        # add samples:
        if args.add_samples:
            classes = args.add_samples.split('/')
            for audio_file in wavs:
                add_sample(database, classes, 0, audio_file)
            with open(database_file, "w") as f:
                json.dump(database, f)

        # classification
        elif args.target:
            for audio_file in wavs:
                ts = time.time()
                #print("Result (KNN-Classification):")
                classifier.ignored_file = audio_file.filename
                n = 5
                if args.neighbors_knn:
                    n = args.neighbors_knn
                #audio_file.features, .class
                result = classifier.classify(audio_file, timestamp=ts, algorithm="knn", neighbors_knn=n)
                if result == args.target.split("/")[-1]:
                    sys.exit(0)
                for root, subdirs, _  in os.walk("classes"):
                    if result in subdirs:
                        if root.split("/")[1] == args.target.split("/")[0]:
                            sys.exit(2)

                sys.exit(1)
                #print("Took %.4f" % (time.time() - ts))

                #print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

                #ts = time.time()
                #print("Result (SVM-Classification):")
                #result = classifier.classify(audio_file, timestamp=ts, target=args.target, algorithm="svm")
                #print("Took %.4f" % (time.time() - ts))
        else:
            results = {}
            for audio_file in wavs:
                n = 5
                if args.neighbors_knn:
                    n = args.neighbors_knn
                result = classifier.classify(audio_file, timestamp=ts, algorithm="knn", neighbors_knn=n)
                #results[result] += 1 
                if result not in results.keys():
                    results[result] = 1
                else:
                    results[result] += 1
                #print results
            print results

    elif args.test_features:
        test_features(database, classifier, 'results4.txt')
        #test_features3(database, classifier)
        
    elif args.test_db:
        accuracy = test_database(database, classifier)
        print 'Total accuracy:', accuracy

    elif args.draw_classes:
        import pydot
        draw_hierarchy(database, args.draw_classes)

    else:
        print("help")

    ts = time.time() - ts
    print "Executed in %.2f s"%(ts)

# test all combinations of 3 features
def test_features3(database, classifier):
    features = ["zcr","flux","energy", "mfcc","loudness", "obsi",
                 "sharpness","spread","rollof","variation"]
    for i in features:
        for j in features:
            for k in features:
                if i == j or i == k or j == k:
                    continue
                tss = time.time()
                print "Running tests with features: %s, %s, and %s"%(i,j,k)
                classifier.used_features = [i, j, k]
                _stdout = sys.stdout
                sys.stdout = open("results_%s_%s_%s.txt"%(i,j,k),"w")
                test_database(database, classifier)
                sys.stdout.close()
                sys.stdout = _stdout
                tss = time.time() - tss
                print "Test took %.2f"%tss

# test all feature combinations
def test_features(database, classifier, filename):
    features = ['mfcc', 'zcr', 'energy', 'flux', "loudness", 
                "obsi", "sharpness", "spread", "rollof", "variation"]
    tss = time.time()
    results = []
    count = 0
    f = open(filename, 'w')
    for L in range(1, len(features)+1):
        for subset in itertools.combinations(features, L):
            count += 1
            feats = ",".join(subset)
            print count, feats
            classifier.used_features = list(subset)
            accuracy = test_database(database, classifier)
            result = [accuracy, feats]
            results.append(result)
    results.sort(key=lambda x: x[0], reverse=True)
    for result in results:
        f.write('%f|Features: %s\n' %(result[0],result[1]))
    tss = time.time() - tss
    print "Test took %.2f"%tss
    f.close()

# test database
def test_database(database, classifier):
    if VERBOSE:
        print 'Testing all samples of the database...'
    test_database.correct = 0
    test_database.n_samples = 0
    test_samples(database, classifier)
    r = float(test_database.correct)/float(test_database.n_samples) * 100.0
    if VERBOSE:
        print 'Total number of samples:', test_database.n_samples  
        print 'Classified Correctly:', test_database.correct    
    print 'Total accuracy: {0:.2f}%'.format(r)
    return r
    #sys.exit(r)

# classify every sample recursively
def test_samples(json_data, classifier):
    for node in json_data:
        if (type(json_data[node]) == dict) and ('samples' in json_data[node]):
            n_samples = len(json_data[node]['samples'])
            if n_samples > 0:
                results = {}
                for i in range(n_samples):
                    audio = Audio()
                    audio.features = json_data[node]['samples'][i]
                    audio.filename = json_data[node]['filenames'][i]
                    result = classifier.classify(audio, algorithm="knn", neighbors_knn=6)
                    #if VERBOSE:
                    #    print 'Sample:', audio.filename
                    #    print 'Result', result
                    if result not in results.keys():
                        results[result] = 1
                    else:
                        results[result] += 1
                result_msg = ''
                for key in sorted(results.keys()):
                    #result_msg += '%s: %s | ' % (key, results[key])
                    result_msg += ',%s,%s' % (key, results[key])
                r = 0.0
                if node in results.keys():  #atleast 1 correct?
                    r = float(results[node])/float(n_samples) * 100.0
                    test_database.correct += results[node]
                test_database.n_samples += n_samples
                if VERBOSE:
                    #print 'Class: %s | nr of samples: %d |' % (node, n_samples)
                    #print 'Results:', result_msg
                    #print 'Accuracy: {0:.2f}%'.format(r)
                    #print '----------------------'
                    print '%s%s%s'%(node,result_msg,',{0:.2f}%'.format(r) )
            test_samples(json_data[node], classifier)

# add sound sample to database
def add_sample(json_data, classes, i, audio_file):
    if classes[i] not in json_data:
        json_data[classes[i]] = {}
        json_data[classes[i]]['samples'] = []
        json_data[classes[i]]['filenames'] = []
        json_data[classes[i]]['count'] = 0
        if VERBOSE:
            print 'Created class: ', classes[i]
    node = classes[i]
    i += 1
    if i < len(classes):
        add_sample(json_data[node], classes, i, audio_file)  #go deeper
    else:
        json_data[node]['samples'].append(audio_file.features)
        json_data[node]['filenames'].append(audio_file.filename)
        json_data[node]['count'] += 1
        if VERBOSE:
            print 'Sample %s added to "%s".' % (audio_file.filename, node)

# draw class hierarchy to image
def draw_hierarchy(database, filename):
    graph = pydot.Dot(graph_type='graph')
#    asd_node = pydot.Node("asd", style='filled', fillcolor='green')
    draw_recursive(graph, database, 0, 0)
    graph.write_png(filename)

def draw_recursive(graph, json_data, previous_graph_node, depth):
    for node in json_data:
        if (type(json_data[node]) == dict) and ('samples' in json_data[node]):
            node_text = node
            if VERBOSE and len(json_data[node]['samples']) > 0:
                node_text += ' (' + str(len(json_data[node]['samples'])) +')'
            graph_node = pydot.Node(node_text, style='filled', fillcolor='green')
            graph.add_node(graph_node)
            if depth > 0:
                graph.add_edge(pydot.Edge(previous_graph_node, graph_node))
            draw_recursive(graph, json_data[node], graph_node, depth+1)

def update_mainclass_metadata(data):
    for main_class in data.keys():
        totcount = 0
        values=[]
        print len(data[main_class].keys())
        if len(data[main_class].keys()) == 1:
            continue
        for sub_class in data[main_class]:
            print main_class, sub_class
            if sub_class == "metadata":
                print totcount
                continue

            data_a = data[main_class][sub_class]["metadata"]
            totcount += data[main_class][sub_class]["count"]
            print totcount, sub_class
            values.append([data[main_class][sub_class]["count"],data_a["flux_avg"],data_a["zcr_avg"]])
        flux_total = 0
        zcr_total = 0
        for i, j, k in values:
            print i,j,k
            flux_total += i*j
            zcr_total += i*k
        
        print zcr_total, totcount, sub_class
        zcr_total = zcr_total/totcount
        flux_total = flux_total/totcount
        data[main_class]["metadata"]["flux_avg"] = flux_total
        data[main_class]["metadata"]["zcr_avg"] = zcr_total
    return data


def plot_metadata(data, plot_file):
    #from pylab import *
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.lines as mlines
    colors = [0]
    xs= []
    ys= []
    zs= []
    markers = ["x","+","s","*","D",".",",","o","<",">","v","1","2","3","4","8","p", "H"]
    #markers = mlines.Line2D.filled_markers
    ignore = ["faucet", "metadata","count","samples"]
    #markers = list(markers) 
    i = 0
    
    fig = plt.figure()
    
    for main_class in data.keys():
        marker = markers.pop()
        print main_class
        print marker
        for sub_class in data[main_class].keys():
            #print sub_class
            if sub_class in ignore:
                continue
            i += 1
            #print "heer"
            for sample in data[main_class][sub_class]["samples"]:
                #print sample
                if isinstance(sample, list):
                    #for i in sample:
                    sample = sample[0]    
                    print "Meneeko tama tanne?"
                y = sample["flux_avg"]
                x = sample["zcr_avg"]
                #plt.text(x, y, sub_class)
                colors.append(colors[-1]+1)
            axl = fig.add_subplot(1,1,1)
            axl.set_ylabel("Spectral Flux")
            axl.set_xlabel("ZCR")
            axl.scatter(x,y,  marker=marker, alpha=0.5, s=50)


            xs = []
            ys = []
            zs = []
    plt.savefig(plot_file)
    print "Saved plot as %s."%(plot_file)


main()

