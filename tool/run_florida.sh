files=$(ls -Sr /home/zhangliqun/test_mtx/*.mtx)
codes=$(ls src/test/level2/)

for file in ${files}; do
  for code in ${codes}; do
    if [[ $code == spmv_csr_r_f32_test_metr* ]]; then
      echo $code
      echo $file
      ./build/src/test/${code::-3} --data-file="${file}"
      echo
    fi
  done
done
