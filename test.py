from SHARAD import SHARAD2d
from SHARAD import peakdetect
import numpy as np

#series = np.random.rand(100,1) 
#maxtab, mintab = peakdetect.peakdet(np.array(series),0.0001)
#print maxtab
#datafolder = '/Volumes/Si-Ting/3Dworkspace'
datafolder = '/Volumes/WD-2TB/CEPData'
s = SHARAD2d.sharadProc()
#s.setDataFolder(datafolder + '/Radargrams/others/Others')
s.setDataFolder(datafolder + '/Radargrams')
#s.setROI(datafolder + '/ROI/test-area-3.shp')
s.setROI(datafolder + '/ROI/study-area.shp')
#s.setDTM(datafolder + '/DTM/MOLA/megr_s_512.tif')
s.setDTM(datafolder + '/DTM/MOLA/megr00-44n090hb.tif')

with open(datafolder + '/datalist.txt') as file:
    if file is not None:
        datalist = file.read().splitlines()

elist = []
for id in datalist:
    print id
    s.setID(id)
    s.readData()
    if s.geoinfo is None:
        continue
    s.clipDatabyShape()
    if s.geoinfo is None:
        continue
    else:
        print s.geoinfo.shape
    try:
        s.chainProcess()
    except:
        elist.append(id)
        continue

np.savetxt('test.out',elist.asarray())

