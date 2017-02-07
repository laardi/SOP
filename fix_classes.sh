#!/bin/sh

for i in $(find "classes/" -name files); do
	cat $i | sort | uniq > ${i}.sorted
	mv ${i}.sorted $i
done
