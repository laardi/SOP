#!/bin/sh
python main.py -c speech/male /tmp/puhe/M*
python main.py -c speech/female /tmp/puhe/F*
python main.py -c natural/steps /tmp/aanipankki_mono/askel*
python main.py -c natural/cat /tmp/aanipankki_mono/kissa*
python main.py -c non-natural/faucet /tmp/aanipankki_mono/vesihan*
python main.py -c non-natural/radio /tmp/aanipankki_mono/radio*
