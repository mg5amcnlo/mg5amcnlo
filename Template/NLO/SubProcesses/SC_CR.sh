basedir=$(pwd)
for x in ls -s P0*
do
  echo $x
  cd $x
  ln -s ../input_check_sudakov.dat
  make check_sudakov > check_sudakov_make.log
  ./check_sudakov < input_check_sudakov.dat > output_check_sudakov.dat 
  cd $basedir
done


