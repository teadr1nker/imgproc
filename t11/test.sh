#!/bin/bash
echo "" > results
numbers=(800 600 400 200 100)
for number in "${numbers[@]}"
do
echo $number >> results
python3 file3.py $number | grep -E "^. accuracy" >> results
done
