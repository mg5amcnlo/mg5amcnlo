proc=$1
nborn=$2

cd /home/prestel/work/2020/MG5/PY8meetsMG5aMC_release/$proc/Events/run_01
gunzip events.lhe.gz
cd -

rm main-test-scales
make main-test-scales

./main-test-scales /home/prestel/work/2020/MG5/PY8meetsMG5aMC_release/$proc/Events/run_01/events.lhe $nborn $proc
cp 1st-emission-scale_template.gp tmp.gp
sed -i "s/<name>/$proc/g" tmp.gp
sed -i "s/<file1>/${proc}_avgscale_vs_tS.dat/g" tmp.gp
sed -i "s/<file2>/${proc}_avgscale_vs_tH.dat/g" tmp.gp
sed -i "s/<file3>/${proc}_tS.dat/g" tmp.gp
sed -i "s/<file4>/${proc}_tH.dat/g" tmp.gp
gnuplot tmp.gp
