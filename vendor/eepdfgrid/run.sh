#! /bin/bash

function run_collider()
{
     collider_name=$1
cp gridpdfaux_$collider_name.f gridpdfaux.f
cp calcpdf_$collider_name.f calcpdf.f

make >/dev/null 2>&1
echo -n "Totaly luminosity: for $collider_name is"
./totlumi
cp eepdf.f eepdf_$collider_name.f
}

run_collider isronly
run_collider cepc240
run_collider fccee240
run_collider fccee365
run_collider ilc250
run_collider ilc500
run_collider clic3000
