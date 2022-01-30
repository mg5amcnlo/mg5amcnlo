name=$1

cp test_scale_header.gp tmp.gp
sed -i "s/<name>/$name/g" tmp.gp

#echo "plot \\" | tee -a tmp.gp;

count=0;
ls -1 ${name}*.dat | while read line ;
do
  echo ""; echo "";
  cat test_scale_template.gp;
  line2=$(echo $line | sed 's/scalup/scaleps/g')
  echo "set origin 0.0, ${count}.0";
  echo " plot \"$line\" u 1:2 w l lt 1, \"$line2\" u 1:2 w l lt 2";
  let count++;
done >> tmp.gp

#gnuplot tmp.gp
