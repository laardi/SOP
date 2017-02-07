#!/usr/bin/env python

import os
import sys
import random
import subprocess


def print_statistics(index, test, success, features):
    # TODO
    # Get output from program
    i = 0
    if features:
        i = 2

    target = test.split()[3 + i]
    audio_file = test.split()[4 + i]
    _success = "SUCCESS"
    if success == 2:
        _success = "ALMOST"
    elif success != 0:
        _success = "FAIlED"

    print("{: <8}{: <36}{: <16}{: <32}".format(index, target, _success, audio_file))


def run_test(test):
    p = subprocess.Popen(test.split(), stdout=subprocess.PIPE)
    p.wait()
    return p.returncode


def get_random_tests(number_of_tests, features):
    files = []
    tests = []
    targets = []

    for root, subdirs, _files in os.walk("classes"):
        if not _files:
            continue

        if _files[0] == "files":
            with open(os.path.join(root, _files[0]), "r") as f:
                    targets.append(root.split("/", 1)[1])
                    files.append(f.read().split("\n"))
                    for n in files[-1]:
                        if n == "":
                            files[-1].pop()

        else:
            print("No filelist in %s" % root)

    while len(tests) < number_of_tests:
        target = targets[random.randint(0, len(targets) - 1)]
        i = targets.index(target)
        j = random.randint(0, len(files[i]) - 1)
        audio_file = files[i][j]
        if not audio_file:
            print("Could not get audio file...")
            sys.exit(1)

        if features:
            tests.append("python main.py --features %s -t %s %s" % (features, target, audio_file))

        else:
            tests.append("python main.py -t %s %s" % (target, audio_file))

    return tests


def main():
    exit_codes = []
    number_of_tests = 8
    features = ""

    if len(sys.argv) >= 2:
        number_of_tests = int(sys.argv[1])

    if len(sys.argv) > 2:
        features = ",".join(sys.argv[2:])

    tests = get_random_tests(number_of_tests, features)

    print("Performing %d tests.\n" % len(tests))
    print("{i: <8}{t: <36}{s: <16}{f: <32}".format(i="#", t="Target", s="Status", f="File"))
    print("-" * 160)

    for index, test in enumerate(tests):
        exit_codes.append(run_test(test))
        print_statistics(index, test, exit_codes[index], features)

    print("-" * 160)
    print("\n")
    print("Overall result is %.2f%% success rate" % ((100 * (exit_codes.count(0) + (exit_codes.count(2) / 2))) / len(exit_codes)))

main()
