files=$(ls -Sr /data/home/gcx/mtx/*.mtx)
codes=$(ls src/test/level3/)

echo 
for cols in 128 256 512 1024; do
  for file in ${files}; do
    for code in ${codes}; do
      if [[ $code == spsm_csr_r_f64_test* ]]; then
        echo $code
        echo $file
        timeout --preserve-status --signal=SIGTERM 720 ./build/src/test/${code::-3} --data-file="${file}" --cols=$cols
        echo
      fi
    done
  done
done
