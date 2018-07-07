#!/bin/bash

folder=$1
cd $folder
for id in $( cat ../datalist.txt )
do

echo $id

dataurl='http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/data'
browseurl='http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/browse'

echo $browseurl
echo $dataurl

if [ ! -f s_"$id"_rgram.img ]
then

head=`echo $id | cut -b1-4`
echo "$browseurl"/tiff/s_"$head"xx/s_"$id"_tiff.lbl
curl -O "$dataurl"/rgram/s_"$head"xx/s_"$id"_rgram.img
curl -O "$dataurl"/rgram/s_"$head"xx/s_"$id"_rgram.lbl
curl -O "$dataurl"/geom/s_"$head"xx/s_"$id"_geom.lbl
curl -O "$dataurl"/geom/s_"$head"xx/s_"$id"_geom.tab
curl -O "$browseurl"/tiff/s_"$head"xx/s_"$id"_tiff.lbl
curl -O "$browseurl"/tiff/s_"$head"xx/s_"$id"_tiff.tif

fi

done

mv s_*_tiff.* ./Radargrams

