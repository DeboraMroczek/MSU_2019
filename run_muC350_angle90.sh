#!bin/bash


#   w = 1.7 rho = 1.1 and 3.8 --- Acceptable 

for i in 17         
do
	for j in 11 38 
	do 
  	    	sed -e "s/ww/`echo "$i" | awk '{print $1/10}'`/g; s/rho/`echo "$j" | awk '{print $1/10}'`/g" parameters_muC350_angle90.dat  >> parameters_muC350_angle90_"$i"_"$j".dat
		./EOSP parameters_muC350_angle90_"$i"_"$j".dat
		rm parameters_muC350_angle90_"$i"_"$j".dat
	done
done



