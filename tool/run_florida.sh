files=$(ls -Sr /home/guochengxin/spgemm_low/test_mtx/*.mtx)
codes=$(ls src/test/level3/)

for file in ${files}; do
  for code in ${codes}; do
    if [[ $code == spgemm_csr_r_f64_test* ]]; then
      echo $code
      echo $file
      ./build/src/test/${code::-3} --data-file="${file}"
      echo
    fi
  done
done
