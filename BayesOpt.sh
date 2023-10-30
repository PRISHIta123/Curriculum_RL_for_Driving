#!/usr/bin/env bash

counter=5

echo "Enter Mode: 1.Turnrates BO Curriculum 2.Obstacles BO Curriculum 3.Both BO Curriculum "
read mode
echo "Selected Mode: ${mode}"

max=18

if [ $mode -eq 2 ]
then
  max=22
fi

while [ $counter -le $max ]
do
python BayesOpt.py $mode $counter
((counter++))
done
