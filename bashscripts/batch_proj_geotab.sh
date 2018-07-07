#!/bin/bash

folder=$1
cd $folder
for i in $( ls *geom.tab )
do

echo $i
nbase=`echo $i | cut -d. -f1`
echo $nbase
awk -F, '{print $4" "$3}' $i > $i.tmp
gdaltransform -s_srs '+proj=longlat +a=3396190 +b=3376200 +no_defs' -t_srs '+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs ' < $i.tmp | awk '{print ","$1", "$2}' > $i.tmp.prj
paste $i $i.tmp.prj > tmp

tr '\r' ' ' < tmp > $nbase-proj.tab

rm $i.tmp $i.tmp.prj tmp

done
mkdir ./proj-geom
mv *_geom-proj.tab ./proj-geom/
