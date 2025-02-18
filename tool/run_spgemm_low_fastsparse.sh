files=$(ls -Sr /home/guochengxin/spgemm_low/*.mtx)

for file in ${files}; do
    echo $file
    ./SPMMMeasurements "${file}"
    echo
done
