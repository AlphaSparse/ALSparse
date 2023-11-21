files=$(ls -Sr /home/guochengxin/spgemm_low/*.mtx)

for file in ${files}; do
    echo $file
    ./spgemm -cuda -spgemm "${file}"
    echo
done
