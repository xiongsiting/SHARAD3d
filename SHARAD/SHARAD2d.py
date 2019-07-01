# -*- coding: utf-8 -*-
"""
    /***************************************************************************
    sharadProc
    A toolbox for 2D processing
    -------------------
    begin                : 2018-02-01
    copyright            : (C) 2018 by SITING XIONG
    email                : siting.xiong.14@ucl.ac.uk
    git sha              : $Format:%H$
    ***************************************************************************/
    
    /***************************************************************************
    *                                                                         *
    *   This program is free software; you can redistribute it and/or modify  *
    *   it under the terms of the GNU General Public License as published by  *
    *   the Free Software Foundation; either version 2 of the License, or     *
    *   (at your option) any later version.                                   *
    *                                                                         *
    ***************************************************************************/
    This script initializes the plugin, making it known to QGIS.
"""

import numpy as np
import os
import gdal
from gdalconst import *
import pyproj as pj
from math import *

from scipy import interpolate
from scipy import signal
from skimage import filters
from skimage import morphology

import subprocess
import logging
import time, datetime
    
import shapefile
from shapely.geometry import Point
from shapely.geometry import Polygon
import peakdetect as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# Create a class for SHARAD data processing
class sharadProc:
    '''
        For Processing SHARAD products derived by U.S. SHARAD Science Team, which can be downloaded from http://pds-geosciences.wustl.edu/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/
        
        Author: Si-Ting Xiong
        Latest Version : 25-Dec-2016
        
        The class takes as 4 input parameters:
        filepath : folder path which contains all SHARAD products
        track ID : one track No.
        roifile : a shapefile defining the interested area
        dtmfile : path to DTM file which used for simulating cluttergram
    '''
    
    # Initial function, set some constant parameters
    midimg = 1800
    C = 299792458
    sigma = 3.4
    tspace = 0.0375e-6
    MOLARADIUS = 3396000
    mars2000ll = pj.Proj("+proj=longlat +a=3396190 +b=3376200 +no_defs")
    molastere = pj.Proj("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs")
    molaeqc = pj.Proj("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs")
    sharadproj = pj.Proj("+proj=longlat +a=3396190 +b=3396190 +no_defs")
    
    def __init__(self,datapath = None, trackID = None, roifile = None):
        self.setDataFolder(datapath)
        self.setID(trackID)

        self.geoinfo = None
        self.img = None
        
        self.polygon = None
        self.dtm = None
        self.dtmgt = None
        
        self.filtim = None
        self.peakim = None
        self.clutim = None
        self.surfind = None
        self.surfecho = None
        self.points = None

        if datapath and trackID:
            self.readData()
            if :
                self.clipDatabyShape(roifile)
    
    def setDataFolder(self, datapath):
        self.datapath = datapath

        if datapath is not None:
            if datapath.endswith('/'):
                self.datapath = datapath[:-1]

            self.outpath = '/'.join(self.datapath.split('/')[:-1]) + '/Results'
            if not os.path.isdir(self.outpath):
                os.makedirs(self.outpath)

    def setID(self, trackID):
        self.trackID = trackID
    def readData(self):
        self.readGeo()
        self.readImg()
    def readGeo(self):
        '''
        Read out the *_geom.tab file, in which the column are:
        col, time, lat, lon, mrad, srad, rvel, tvel, sza, phase
        '''
        self.geoinfo = []
        geofile = self.datapath + '/s_' + self.trackID + '_geom.tab'
        try:
            with open(geofile,'r') as f:
                for line in f:
                    line = line.replace('\r\n','').split(',')
                    # read lat, lon, Hmars, Hsat
                    self.geoinfo.append([float(line[2]),float(line[3]),float(line[4])*1000,float(line[5])*1000])

            self.geoinfo = np.asarray(self.geoinfo)
            # delay time between satellite and the Mars aeroid
            time = 2 * (self.geoinfo[:,3] - self.geoinfo[:,2]) /self.C
            time = time.reshape((len(time),1))
            self.geoinfo = np.concatenate((self.geoinfo,time),axis = 1)
            # keep lat and lon to the end of self.geoinfo
            # before replacing them to mapx, mapy
            #self.geoinfo = np.concatenate((self.geoinfo, self.geoinfo[:,:2]), axis = 1)

            # transform from Mars 2000 lat/lon to MOLA sphere projection map coordinate x/y
            ptsx,ptsy = pj.transform(self.mars2000ll, self.molaeqc, self.geoinfo[:,1],self.geoinfo[:,0])
            self.geoinfo[:,0] = ptsx
            self.geoinfo[:,1] = ptsy
        except:
            self.geoinfo = None

    def readImg(self):   
        dict = {}
        lblfile = self.datapath + '/s_' + self.trackID + '_rgram.lbl'
        try:
            with open(lblfile ,'r') as f:
                for line in f:
                    linelist = line.replace('\r\n','').split('=')
                    if len(linelist) == 2:
                        key,val = line.split('=')
                        key = key.strip()
                        val = val.strip()
                        dict[key] = val

            ny = np.int16(dict['LINES'])
            nx = np.int16(dict['LINE_SAMPLES'])
            nbytes = np.int16(dict['SAMPLE_BITS'])
        
            if nbytes == 32:
                datatype = np.float32
            elif ifnbytes == 16:
                datatype = np.int16

            imgfile =  ('.').join(lblfile.split('.')[:-1]) + '.img'

            f = open(imgfile,'r')
            f.seek(0,os.SEEK_SET)
            img = np.fromfile(f ,dtype = datatype)
            f.close()
            self.img = img.reshape((ny,nx))
        except:
            self.img = None

# ############# CLIP DATA ###########################
    def setROI(self, roifile):

        try:
            shape = shapefile.Reader(roifile)
            #first feature of the shapefile
            feature = shape.shapeRecords()[0]
            first = feature.shape.__geo_interface__
        except:
            return -1
        
        latlon = list(first['coordinates'][0])
        polygonx,polygony = pj.transform(self.sharadproj, self.molaeqc, [i[0] for i in latlon],[i[1] for i in latlon])
        self.polygon = Polygon(zip(polygonx,polygony))
    def clipDatabyShape(self, roifile = None):
        '''
        Clip Data by using the shapefile
        '''
        if self.img is None or self.geoinfo is None:
            return
        
        if roifile is not None:
            self.setROI(roifile)
        
        # geoinfo latlon has been changed to x, y in self.readGeo()
        ptsx = self.geoinfo[:,0]
        ptsy = self.geoinfo[:,1]
            
        pts = [ Point(i) for i in zip(ptsx, ptsy) ]
        subind = np.where([self.polygon.contains(i) for i in pts])[0]

        if len(subind) <= 0:
            self.geoinfo = None
            self.img = None
        else:
            self.geoinfo = self.geoinfo[subind,:]
            self.img = self.img[:,subind]
            subind = subind.reshape((len(subind),1))
            self.geoinfo = np.concatenate((self.geoinfo,subind.astype(int)),axis = 1)
# ############# DTM AND CLUTGRAM ###########################
    def setDTM(self, dtmfile):
        if dtmfile is None:
            return None

        ## It is currently only appliable to MOLA DTM, need to check with HRSC DTM
        ds = gdal.Open(dtmfile,GA_ReadOnly)
        if ds is None:
            return None
        else:
            cols = ds.RasterXSize
            rows = ds.RasterYSize

            gt = ds.GetGeoTransform()
            band = ds.GetRasterBand(1)
            self.dtm = band.ReadAsArray()
            nx, ny = self.dtm.shape

            ## MOLA has problem here, need to check with HRSC data
            #ulx = - nx / 2 * gt[1]
            #uly = - ny / 2 * gt[5]
            ulx = gt[0]
            uly = gt[3]

            gt = [ulx,uly,gt[1],gt[5]]
            self.dtmgt = gt


    def dtmInterpxyz(self, x, y, method = 'nearest'):
        if self.dtm is None or self.dtmgt is None:
            return None

        dtm = self.dtm

        resx = self.dtmgt[2]
        resy = self.dtmgt[3]
        ulx = self.dtmgt[0]
        uly = self.dtmgt[1]

        nx, ny = self.dtm.shape
        '''
        X = np.arange(gt[0],gt[0] + (gt[2] * nx),gt[2])
        Y = np.arange(gt[1],gt[1] + (gt[3] * ny),gt[3])
    

        xx,yy = np.meshgrid(X,Y)
        f = interpolate.interp2d(xx,yy,dtm,kind = 'linear')
        vdtm = f(x,y)
        '''
        # Correct here for python cause' python interpolation is overflow with too many points
        vdtm = np.zeros(len(x),dtype='float')
        for i in xrange(len(x)):
            xi = int(floor((x[i] - ulx) / resx))
            yi = int(floor((y[i] - uly) / resy))

            u = ((x[i] - ulx) - xi * resx) / resx
            v = ((y[i] - uly) - yi * resy) / resy

            if xi >= 0 and xi + 1 < nx and yi >=0 and yi + 1 < ny:
                if method == 'nearest':
                    if u <= 0.5 and v <= 0.5:
                        vdtm[i] = dtm[yi, xi]
                    elif u <= 0.5 and v > 0.5:
                        vdtm[i] = dtm[yi + 1, xi]
                    elif u > 0.5 and v <= 0.5:
                        vdtm[i] = dtm[yi, xi + 1]
                    else:
                        vdtm[i] = dtm[yi + 1, xi + 1]
                elif method == 'linear':
                        vdtm[i] = (1-u) * (1-v) * dtm[yi,xi] + u * v * dtm[yi+1,xi+1] + u * (1-v) * dtm[yi,xi + 1] + (1-u)*v*dtm[yi + 1, xi]
                else:
                        vdtm[i] = None
            else:
                vdtm[i] = -32768

        return vdtm + self.MOLARADIUS
    

    def dtmTrack(self): 
        if self.dtm is None or self.dtmgt is None:
            return None

        if self.geoinfo is None or self.geoinfo.shape[1] < 5:
            return None
        else:
            z = self.dtmInterpxyz(self.geoinfo[:,0], self.geoinfo[:,1])
            return z

    def dtmPatch(self,dtmwidth):
        if self.geoinfo is None:
            return None

        if self.dtm is None or self.dtmgt is None:
            return None
        
        ## Calculate the four corners of the subarea on DTM used to simulate the cluttergram
        x = self.geoinfo[:,0]
        y = self.geoinfo[:,1]
        kt = (y[-1]-y[0]) / (x[-1]-x[0])
        k = -1/ kt
            
        xhalf = dtmwidth * cos(abs(atan(k)))
        yhalf = dtmwidth * sin(abs(atan(k)))
        xpixels = ceil(2 * xhalf / self.dtmgt[2])
        ypixels = ceil(2 * yhalf / -self.dtmgt[3])
        
        acrosspixels = max(xpixels, ypixels)
        
        X = np.zeros((acrosspixels,len(x)))
        Y = np.zeros((acrosspixels,len(x)))
        Z = np.zeros((acrosspixels,len(x)))
        
        for i in xrange(len(x)):
            if acrosspixels == xpixels:
                X[:,i] = np.arange(x[i]-abs(xhalf),x[i]+abs(xhalf),self.dtmgt[2])
                Y[:,i] = k * X[:,i] + y[i] - k * x[i]
            else:
                Y[:,i] = np.arange(y[i]+abs(yhalf),y[i]-abs(yhalf),self.dtmgt[3])
                X[:,i] = (Y[:,i] - y[i] + k * x[i]) / k

            Z[:,i] = self.dtmInterpxyz(X[:,i], Y[:,i])

        return [Z,X,Y]

    def cluttergram(self,dtmwidth = 6e4, mtype = 1):
        if self.geoinfo is None:
            return None
        if self.dtm is None or self.dtmgt is None:
            return None
        # Simulate the cluttergram
        # See from Adamo Ferro IEEE TGRS, VOL. 51 NO. 5 2013
        x = self.geoinfo[:,0]
        y = self.geoinfo[:,1]
        Hsat = self.geoinfo[:,3]
        time = self.geoinfo[:,4]
        
        dtm, X, Y = self.dtmPatch(dtmwidth)
        logging.info('Save the dtm patch used to simulate the cluttergram as %s_sim.tif'% self.trackID)

        ncol = len(self.geoinfo[:,-1])
        self.clutim = np.zeros((self.midimg * 2,ncol),dtype = 'float')
        self.surfind = np.zeros((ncol,1),dtype = 'int')
     
        for i in xrange(ncol):
            for j in xrange(1, dtm.shape[0] - 1):
                X0 = x[i]
                Y0 = y[i]
                h0 = Hsat[i]
                Xp = X[j,i]
                Yp = Y[j,i]
                hp = dtm[j,i]
                if np.isnan(hp):
                    continue
                
                Rxy = sqrt((Xp - X0)**2 + (Yp - Y0)**2 + (hp-h0)**2)
                t = 2 * Rxy / self.C;
                tdelay = self.midimg + int(round((t - time[i]) / self.tspace,0))
                if tdelay >= 3600:
                    continue

                if j == ceil(dtm.shape[0]/2):
                    self.surfind[i,0] = tdelay
            
                if mtype == 1:
                    self.clutim[tdelay,i] += 1.0/pow(Rxy,4)
                elif mtype == 2:
                    # Using local incidence angle and relative dielectric constant
                    pxspace = sqrt((X[j+1,i]-X[j,i])**2 + (Y[j+1,i]-Y[j,i])**2)
                    # Local incidence angle
                    theta = atan2(abs(dtm[j+1,i]-dtm[j,i]), pxspace)
                    # rho is the relative Fresnel coefficient of the surface
                    rho = ((cos(theta)-sqrt(self.sigma - sin(theta)**2))/(cos(theta) + sqrt(self.sigma - sin(theta)**2)))**2
                    self.clutim[tdelay,i] = self.clutim[tdelay,i] + rho * cos(theta)/pow(Rxy,4)
                elif mtype == 3:
                    ## The equation is (4) from M.G. Spagnuolo et al. Planetary and Space Science 59(2011) 1222-1230
                    K = sqrt(0.01)
                    pxspace = sqrt((X[j+1,i]-X[j,i])**2 + (Y[j+1,i]-Y[j,i])**2)
                    theta = atan2(abs(dtm[j+1,i]-dtm[j,i]), pxspace)
                    rho = ((cos(theta)-sqrt(self.sigma - sin(theta)**2))/(cos(theta) + sqrt(self.sigma - sin(theta)**2)))**2
                    rho = rho * K / 2 * pow(pow(cos(theta),4) + K * sin(theta)**2,-3/2)
                    self.clutim[tdelay,i] = self.clutim[tdelay,i] + rho * cos(theta)/pow(Rxy,4)

# ############# FILTERING ###########################
    def logGaborFilter(self, nscale = 15, mult = 2, norient = 6, sigmaOnf = 0.55, dThetaOnSigma = 1):
        if self.img is None:
            return

        minWaveLength = 2
        thetaSigma = np.pi/norient/dThetaOnSigma

        epsilon = 0.00001
        softness = 1
        k = 2

        imfft = np.fft.fft2(self.img)
        a = abs(imfft)
        [rows,cols] = imfft.shape

        if rows <= 0 or cols <= 0:
            return

        b = np.array(range(-cols/2,cols/2))/float(cols/2)
        x = np.repeat(b.reshape(1,cols),rows,axis = 0)
        b = np.arange(-rows/2, rows/2)/float(rows/2)
        y = np.repeat(b.reshape(1,rows),cols,axis = 0)
        y = np.transpose(y)
        x = x.reshape(rows*cols,1, order = 'F')
        y = y.reshape(rows*cols,1, order = 'F')

        radius = np.sqrt(x * x + y * y)
        radius[rows * int(ceil(cols / 2.0)) + rows/2] = 1

        theta = [ atan2(-j,i) for i,j in zip(x,y) ]

        totalEnergy = np.zeros((rows,cols),dtype=complex)

        for orient in xrange(1,norient + 1):
            print('Processing orientation %s' % str(orient))

            angl = (orient) * pi / norient
            wavelength = minWaveLength

            ds = np.sin(theta) * cos(angl) - np.cos(theta) * sin(angl) # Difference in sine.
            dc = np.cos(theta) * cos(angl) + np.sin(theta) * sin(angl) # Difference in cosine.

            dtheta = [ atan2(i,j) for i,j in zip(ds,dc) ]          # Absolute angular distance.
            spread = np.array([exp(-(x * x)/(2 * pow(thetaSigma,2))) for x in dtheta])

            for scale in xrange(1,nscale + 1):
                fo = 1.0/wavelength                 # Centre frequency of filter.
                rfo = fo/0.5

                logGabor = [exp((-pow(log(r/rfo),2)) / (2 * pow(log(sigmaOnf),2))) for r in radius]
                                     # Set the value at the center of the filter
                logGabor[rows * int(ceil(cols / 2.0)) + rows/2] = 0
                                                  # back to zero (undo the radius fudge).

                kernel = np.multiply(logGabor, spread)          # Multiply by the angular spread to get the filter.
                kernel = kernel.reshape(rows,cols, order = 'F')
                kernel = np.fft.fftshift(kernel)            # Swap quadrants to move zero frequency to the corners.

                # Convolve image with even an odd filters returning the result in EO
                EOfft = np.multiply(imfft, kernel)          # Do the convolution.
                EO = np.fft.ifft2(EOfft)                    # Back transform.
                aEO = abs(EO)

                #estMeanEn = []

                if scale == 1:
                    medianEn = np.median(aEO.reshape(1, rows * cols))
                    meanEn = medianEn * 0.5 * sqrt(-pi/log(0.5))
                    RayVar = (4 - pi) * (meanEn * meanEn) /pi
                    RayMean = meanEn

                    #estMeanEn = np.concatenate(estMeanEn, meanEn)
                    #sig = np.concatenate(sig,math.sqrt(RayVar))

                T = (RayMean + k * sqrt(RayVar))/pow(mult, (scale-1))

                V = np.divide(softness * T * EO, aEO + epsilon)
                V = np.ma.masked_array(V, aEO <= T).filled(0)
                V += np.ma.masked_array(EO, aEO > T).filled(0)

                #a = np.max(aEO[1:1000,:])
                #V = EO
                #V = np.ma.masked_array(V, aEO > a).filled(0)

                EO = EO - V
                totalEnergy += EO
                wavelength = wavelength * mult

        self.filtim = totalEnergy.real

# ############# PEAK DETECTION ###########################
    def firstReturn(self, im = None, method = 'mouginot'):
        # This function is used for picking the first Return echo position from SHARAD radargram
        flagreturn = 1

        if im is None:
            im = self.img
            flagreturn = 0

        if self.img is None or im.shape[1] == 0:
            return
        
        nlen, ntrace = im.shape
        
        surfecho = np.zeros(ntrace)
        for col in xrange(ntrace):
            iprofile = im[:,col]
            if method == 'mouginot':
                Si = iprofile[30:] ** 2
                mSi = [(iprofile[i-30:i-1] ** 2).mean() for i in xrange(30,nlen)]

                try:
                    C = Si/mSi
                except:
                    print mSi
                    continue
                surfecho[col] = np.argmax(C) + 30
            
            elif method == 'ferro':
                # This method is from Ferro et al. 2013
                Noise = iprofile[len(iprofile)-50:-1]
                mval = np.mean(Noise)
                sval = np.std(Noise)
                r = 6.0
                d = 0.8
                while r > 0:
                    thres = mval + r * sval
                    soN = np.where(iprofile > thres)[0]
                    if soN is not None:
                        surfecho[col] = soN[0]
                        break
                    else:
                        r = r * d
            else:
                return None

        if flagreturn == 0:
            self.surfecho = surfecho
        else:
            return surfecho
    def peakimCWT(self, scales, bgSkip = 20):
        # CWT-based peak detection
        if self.surfecho is None:
            return

        if self.filtim is not None:
            im = self.filtim
        elif self.img is not None:
            im = self.img
        else:
            return

        self.peakim = np.zeros_like(im)
        num = np.zeros((len(scales),im.shape[1]))
        for col in xrange(im.shape[1]):
            # direct peak detection after strong logGabor filtering is not working
            #peakmx, peakmn = pd.peakdet(im[:,col],0.000001)
            #peakind = peakmx[:,0].astype(int)
            #self.peakim[peakind,col] = 1

            cwtMatr = signal.cwt(im[:,col], signal.ricker, scales)
            idxStart = int(self.surfecho[col])
            for r in xrange(cwtMatr.shape[0]):
                cwtRow = cwtMatr[r,:]

                bgmax = 0
                # take the first pixels till surfecho to be the background
                bgSig = cwtRow[1:idxStart - bgSkip]
                bgmax = max(bgSig)

                if bgmax is None:
                    bgmax = 0

                peakmx, peakmn = pd.peakdet(cwtRow,0.000001)
                if peakmx.shape[0] <= 0:
                    continue
                peakind = peakmx[:,0].astype(int)
                ind = np.where(peakmx[:,1] > bgmax)[0]
                peakind = peakind[ind]
                #if col == 0 and r == 0:
                 #   plt.plot(im[:,col])
                 #   plt.plot(peakind, im[peakind,col],'ro')
                 #   plt.show()

                self.peakim[peakind,col] = self.peakim[peakind,col] + cwtRow[peakind]
                #num[peakind,col] = num[peakind,col] + 1
                num[r,col] = len(peakind)
        return num
    def removeClutter(self, clutRatio = 0.5, dist = 10):
        surfclut = self.firstReturn(self.clutim)
        bclutim = np.zeros_like(self.img)
        for col in xrange(self.clutim.shape[1]):
            iprofile = self.clutim[:,col]
            bgnoise = np.max(iprofile)
            peakmx, peakmn = pd.peakdet(iprofile,1e-26)
            if peakmx.shape[0] <= 0:
                continue
            peakind = peakmx[:,0].astype(int)
            ind = np.where(peakmx[:,1] > bgnoise * clutRatio)[0]
            peakind = peakind[ind]
            #peakind = np.where(iprofile > bgnoise * 0.5)[0].astype(int)
            bclutim[peakind,col] = 1

        plt.figure()
        xy = np.where(bclutim == 1)
        x = xy[0]
        y = xy[1]
        plt.scatter(y,x - 1300, color='r',marker = '.',zorder = 1)
        plt.xlim(0,self.clutim.shape[1])
        plt.imshow(self.clutim[1300:3600,:], zorder = 0)

        filename = self.outpath + '/s_' + self.trackID + '_rmclut.png'
        plt.savefig(filename)
        plt.close()

        for col in xrange(self.peakim.shape[1]):
            deltaH = int(self.surfecho[col] - surfclut[col])
            peakind = np.where(self.peakim[:,col] != 0)[0].astype(int)
            tmp = peakind
            peakind = np.delete(tmp, np.where(tmp - deltaH >= self.peakim.shape[0])[0])
            if len(peakind) <= 0 or abs(deltaH) > 15:
                continue
            if abs(deltaH) > 0:
                self.peakim[peakind - deltaH, col] = self.peakim[peakind,col]
                self.peakim[peakind,col] = 0

        for col in xrange(self.peakim.shape[1]):
            self.peakim[1:self.surfecho[col] + 1,col] = 0

            clind = np.where(bclutim[:,col] > 0)[0]
            pkind = np.where(self.peakim[:,col] > 0)[0]

            if len(pkind) <= 0:
                self.peakim[self.surfind[col], col] = 1
                continue

            #self.peakim[pkind[0], col] = 0

            for k in clind:
                anyind = np.where(abs(pkind - k) < dist)[0]

                if len(anyind) <= 0:
                    continue
                else:
                    self.peakim[pkind[anyind],col] = 0

            self.peakim[self.surfind[col], col] = 1


# ############# OUTPUT ###########################
    def savePoints(self, minpts = 2, outfile = None):

        if outfile is None:
            if self.outpath is not None:
                outfile = self.outpath + '/s_' + self.trackID + '_subpt.txt'
            else:
                return

        # remove the isolated points
        a = self.peakim > 0
        mask = morphology.remove_small_objects(a, minpts, connectivity=2)
        self.peakim = np.multiply(self.peakim, mask)

        dbim = 10 * np.log10(self.img)

        xy = np.where(self.peakim > 0)
        x = xy[0]
        y = xy[1]
        z = self.geoinfo[y,2] + (1800 - x )* self.tspace* self.C/2 - self.MOLARADIUS
        #z = (x - self.surfind) * self.tspace * self.C / (2 * self.sigma)
        self.points = np.zeros((len(x),5))
        self.points[:,0] = self.geoinfo[y, 0]
        self.points[:,1] = self.geoinfo[y, 1]
        self.points[:,2] = z
        self.points[:,3] = dbim[x,y]
        self.points[:,4] = self.peakim[x,y]
        np.savetxt(outfile, self.points,'%.4f', ',')

    def saveTiff(self, outfile, im):
        [nx, ny] = im[0].shape
        driver = gdal.GetDriverByName('GTiff')
        ds  = driver.Create(outfile, ny, nx, 3, gdal.GDT_Float32)
        for i in np.arange(3):
            ds.GetRasterBand(i+1).WriteArray(im[i])
        ds.FlushCache()
        ds = None

    def savePlot(self,outfile = None):
        if outfile is None:
            if self.outpath is None:
                return
            else:
                outfolder = self.outpath

        if outfile is None:
            outfile = outfolder + '/s_' + self.trackID + '.png'

        if self.peakim is not None and self.img is not None:
            plt.figure()
            plt.subplot(121)
            xy = np.where(self.peakim > 0)
            x = xy[0]
            y = xy[1]
            mx = x.min()
            mx = mx - 50
            if mx < 0:
                mx = 1300
            ind = np.where(np.logical_and(x > mx,x < 3600))
            x = x[ind]
            y = y[ind]
            plt.scatter(y,x - mx,color = 'k', marker = '.',zorder = 1)
            #plt.plot(self.surfind - mx + 50, color = 'g', marker = '.', zorder = 1)
            #plt.ylim(1300,1800)
            plt.xlim(0, self.img.shape[1])
            #ext = [0, self.img.shape[1], 0, 1800 - 1300]
            plt.imshow(10*np.log10(self.img[mx:3600,:]), zorder = 0)
            #plt.gca().invert_yaxis()
            #plt.colorbar()


            if self.clutim is not None:
                plt.subplot(122)
                plt.plot(self.surfind - mx, color = 'r', marker = '.',zorder = 1)
                plt.xlim(0,self.clutim.shape[1])
                plt.imshow(self.clutim[mx:3600,:])
                plt.savefig(outfile, bbox_inches = 'tight',
                pad_inches = 0)
                plt.close()
# ############# CHAIN PROCESS ###########################
    def chainProcess(self, outfile = None, params = None):
        if self.img is not None and self.geoinfo is not None:
            if params is None:
                scale = np.arange(1,5)
                dtmwidth = 60000
                pfilter = [5,3,6]
                minpts = 2
            else:
                pfilter = params[0]
                scale = params[1]
                dtmwidth = params[2]
                minpts = params[3]

            # convert the format of image
            #self.img = 10 * np.log10(self.img)
            #scaler =  MinMaxScaler()
            #self.img = scaler.fit_transform(self.img)

            #filename = self.outpath + '/s_' + self.trackID + '_filtim.txt'
            #if os.path.isfile(filename):
            #self.filtim = np.loadtxt(filename, delimiter = ',')
            #else:
            #self.logGaborFilter(pfilter[0],pfilter[1], pfilter[2])

            #filename = self.outpath + '/s_' + self.trackID + '_surfecho.txt'
            #if os.path.isfile(filename):
            #    self.surfecho = np.loadtxt(filename, dtype = 'int', delimiter = ',')
            #else:
            start = time.time()
            self.firstReturn()
            print "Time Comsuption (Find First Return):"
            print time.time() - start

            #filename = self.outpath + '/s_' + self.trackID + '_peakim.txt'
            #if os.path.isfile(filename):
            #    self.peakim = np.loadtxt(filename, delimiter = ',')
            #else:
            self.peakimCWT(scale)
            print "Time Comsuption (CWT Peak detection):"
            print time.time() - start

            if self.dtm is not None and self.dtmgt is not None:
                    self.cluttergram(dtmwidth)
            print "Time Comsuption (Cluttergram):"
            print time.time() - start
            #self.removeClutter(0.95,3)

            self.savePoints(minpts)
            self.savePlot()
            
            outgeofile = self.outpath + '/s_' + self.trackID + '_geom_clip.txt'
            outimgfile = self.outpath + '/s_' + self.trackID + '_rgram_clip.txt'
            np.savetxt(outgeofile, self.geoinfo, '%.4f', ',')
            np.savetxt(outimgfile, self.img, '%.4f', ',')
            filename = self.outpath + '/s_' + self.trackID + '_surfind.txt'
            np.savetxt(filename, self.surfind, '%d', ',')
            filename = self.outpath + '/s_' + self.trackID + '_surfecho.txt'
            np.savetxt(filename, self.surfecho, '%d')
            '''
            filename = self.outpath + '/s_' + self.trackID + '_filtim.txt'
            np.savetxt(filename, self.filtim, '%.4f', ',')
            outgeofile = self.outpath + '/s_' + self.trackID + '_geom_clip.txt'
            outimgfile = self.outpath + '/s_' + self.trackID + '_rgram_clip.txt'
            np.savetxt(outgeofile, self.geoinfo, '%.4f', ',')
            np.savetxt(outimgfile, self.img, '%.4f', ',')
            filename = self.outpath + '/s_' + self.trackID + '_dtmpatch.tif'
            self.saveTiff(filename, [dtm, X, Y])
            filename = self.outpath + '/s_' + self.trackID + '_surfind.txt'
            np.savetxt(filename, self.surfind, '%d', ',')
            #filename = self.outpath + '/s_' + self.trackID + '_clutim.txt'
            #np.savetxt(filename, np.log10(self.clutim), delimiter = ',')
            filename = self.outpath + '/s_' + self.trackID + '_surfecho.txt'
            np.savetxt(filename, surfecho, '%d')
            #filename = self.outpath + '/s_' + self.trackID + '_rmclut.txt'
            #np.savetxt(filename, self.peakim, '%.4f', ',')

            filename = self.outpath + '/s_' + self.trackID + '_peakim.txt'
            np.savetxt(filename, self.peakim, '%.4f', ',')
            filename = self.outpath + '/s_' + self.trackID + '_peaknm.txt'
            np.savetxt(filename, num, '%d',',')
            '''

if __name__ == '__main__':
    s = sharadProc()
