name=$1

cp test_scale_template.gp tmp.gp
sed -i "s/<name>/$name/g" tmp.gp

echo "plot \\" | tee -a tmp.gp; ls -1 ${name}*.dat | while read line ; do echo " \"$line\" u 1:2 w l, \\"; done | tee -a tmp.gp

#gnuplot tmp.gp
