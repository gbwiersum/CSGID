mkdir ./MARCO
cd ./MARCO
curl --progress-bar https://marco.ccr.buffalo.edu/data/archive/test-jpg-tfrecords.tar --output ./test.tar &
curl --progress-bar https://marco.ccr.buffalo.edu/data/archive/train-jpg-tfrecords.tar --output ./train.tar &
wait
tar -xvf ./test.tar &
tar -xvf ./train.tar &
wait
rm ./test.tar
rm ./train.tar
exit
