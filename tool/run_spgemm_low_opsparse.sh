files=$(ls -Sr /home/guochengxin/spgemm_low/*.mtx)

for file in ${files}; do
    echo $file
    ./opsparse "${file}"
    echo
done
