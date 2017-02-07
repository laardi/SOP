#!/bin/bash


declare -a features
features=(zcr flux energy mfcc loudness obsi sharpness spread rollof variation)

for i in ${features[@]}; do
	for j in ${features[@]}; do
		for k in ${features[@]}; do
			if [[ $i == $j ]] || [[ $i == $k ]] || [[ $j == $k ]]; then
				continue
			fi

			echo "Running tests with features: $i, $j and $k"
			python main.py -d flat.json -f $i,$j,$k -b > results_$i\_$j\_$k.txt
		done
	done
done
