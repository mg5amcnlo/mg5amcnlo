basedir=$(pwd)

dirs=$(ls -d P0*)
for x in $dirs
do
  echo $x
  cd $x
  ln -s ../input_check_sudakov.dat
  cp V0*/nsqso_born.inc .
  cp V0*/nsquaredSO.inc .
  make check_sudakov > check_sudakov_make.log
  ./check_sudakov < input_check_sudakov.dat > output_check_sudakov.dat 
  cd $basedir
done


