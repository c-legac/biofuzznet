for f in ./CV_dataset/*/
do
	foldername="${f%*}"
	python optimisation_script.py --inputnetwork ../../../bionics/biofuzznet/example_networks/manual_network_reduced.tsv --outputfolder $foldername --inputdata $foldername --epochs 300 --learningrate 5e-3 --batchsize 200 --rounds 1
done
