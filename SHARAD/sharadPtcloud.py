from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import matplotlib.pyplot as plt
import numpy as np
import gdal
#from scipy.interpolate import griddata
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri
#from scipy.spatial import Delaunay
import scipy.ndimage as ndimage
from osgeo import osr
from mpl_toolkits.axes_grid1 import make_axes_locatable

def clusterPoints(ptfile, outfolder, filtdB = -30, nlayer = 7, dbeps = 0.1, dbsamples = 10):
    
    if outfolder.endswith('/'):
        outfolder = outfolder[:-1]
    
    subpt = np.loadtxt(ptfile, delimiter = ',')
    subpt = subpt[np.where(subpt[:,6] > filtdB),:]
    subpt = subpt[0]
    xy = subpt[:,2:4]
    depth = subpt[:,5]
    value = subpt[:,6]

    data = np.concatenate((xy, depth.reshape((len(depth),1))), axis=1)

#   subpt = data[np.where(data[:,3] > filtdB),:]
#   subpt = subpt[0]
#    data = subpt[:,:3]
#    value = subpt[:,3]

    # Standalise the three metrices
    X = StandardScaler().fit_transform(data)
    # Compute DBSCAN
    db = DBSCAN(eps = dbeps, min_samples = dbsamples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    # nclass including the -1 which is noise
    nclass = len(set(labels))
    # clsslen is number of points in that class
    clsspts = np.zeros((nclass,1))
    
    # centroid of each class
    XCentroid = np.zeros((nclass - 1,3))

    for k in set(labels):
        if k == -1:
            continue
        class_member_mask = (labels == k)
        # core_samples_mask, used for mask out noise
        ind = np.where(class_member_mask & core_samples_mask)[0]
        clsspts[k] = len(ind)
        XCentroid[k,:] = np.mean(data[ind,:],axis=0)

    # Hierarchical clustering using the height only
    Hei = np.zeros((XCentroid.shape[0],1))
    Hei[:,0] = XCentroid[:,2]

    ac = AgglomerativeClustering(n_clusters= nlayer, affinity='euclidean',linkage='average').fit(Hei)
    newlabels = ac.labels_
    
    # get the final labels from class and subclass
    final_labels = np.zeros((subpt.shape[0],1)) + (-1)
    for k in set(newlabels):
        bcls = np.where(newlabels == k)[0]
        clust = []
        for n in bcls:
            clust.extend(np.where(labels == n)[0])
        final_labels[clust,0] = k

    # Output each class into an indenpendent txt file
    for k in set(newlabels):
        ind = np.where(final_labels == k)[0]
        x = data[ind,0]
        y = data[ind,1]
        #z = data[ind,2] - 3396000
        z = data[ind,2]
        v = value[ind]

        filename = "surface-%s.txt" % (k + 1)
        filename = outfolder + '/' + filename
        #np.savetxt(filename, np.transpose([x,y,z,v]), '%.6f', delimiter = ',')
        np.savetxt(filename, subpt[ind,:], '%.4f', delimiter = ',')
    
    
def filterPoints(ptfile, outfile, xblock = 10000, yblock = 10000, kz = 2, kv = 3):
    
    data = np.loadtxt(ptfile, delimiter = ',')
    xy = data[:,2:4]
    z = data[:,5]
    v = data[:,6]
        
    # find unique items in xy
    tmp = np.ascontiguousarray(xy).view(np.dtype((np.void, xy.dtype.itemsize * xy.shape[1])))
    _, idx = np.unique(tmp, return_index=True)
    uxy = xy[idx]
    uz = z[idx]
    uv = v[idx]

    uniq_data = data[idx,:]
    rep_data = np.delete(data,idx)
        
    # remove repeated point and deviated points...
    delete_list= []
    exclulist = []
    for xi, yi in uxy:
        # idx of xi and yi in uxy
        idx_in_uxy = np.where(np.logical_and(uxy[:,0] == xi, uxy[:,1] == yi))[0]
        # for each of these points, need to eliminate those far from the plane
        blkidx = np.where(np.logical_and(abs(uxy[:,0] - xi) < xblock, abs(uxy[:,1] - yi) < yblock))[0]
        if len(blkidx) == 0:
            continue
        blkz = uz[blkidx]
        blkv = uv[blkidx]
            
        # remove the repeat points!!!
        # check how many points having the same x- and y- coordinates
        repidx_in_xy = np.where(np.logical_and(xy[:,0] == xi, xy[:,1] == yi))[0]
        if len(repidx_in_xy) > 1:
            dist = []
            for repz in z[repidx_in_xy]:
                # check in uxy for a distance of
                # Derive a block around the x- and y- coordinates
                dist.append(np.mean([ i * i for i in (blkz - repz)]))
            
            #where in repidx having the smallest dist
            tidx_in_rep = np.argmin(np.array(dist))
            # put these value in uxy
            uxy[idx_in_uxy] = xy[repidx_in_xy[tidx_in_rep]]
            uz[idx_in_uxy] = z[repidx_in_xy[tidx_in_rep]]

            for item in np.delete(repidx_in_xy,tidx_in_rep):
            	if (z[item] - np.mean(blkz)) > kz * np.std(blkz):
                	exclulist.append(item)

        # smooth filter
        if abs(uz[idx_in_uxy] - np.mean(blkz)) > kz * np.std(blkz):
            delete_list.append(idx_in_uxy.item(0))
        #if abs(uv[idx_in_uxy] - np.mean(blkv)) > kv * np.std(blkv):
        #    delete_list.append(idx_in_uxy.item(0))

    print uxy.shape, uz.shape
    print len(delete_list)
    
    #exclu_xy = xy[exclulist,:]
    #exclu_z = z[exclulist]
    #exclu_ux = uxy[delete_list,0]
    #exclu_uy = uxy[delete_list,1]
    #exclu_uz = uz[delete_list]

    this_layer = np.delete(uniq_data, delete_list, axis = 0)
    exclu_data = data[exclulist,:]
    delete_data = uniq_data[delete_list,:]
    exclu_layer = np.concatenate((exclu_data, delete_data), axis = 0)

    #x = np.delete(uxy[:,0], delete_list)
    #y = np.delete(uxy[:,1], delete_list)
    #z = np.delete(uz,delete_list)
    
    #a = np.concatenate((exclu_xy,exclu_z.reshape((len(exclu_z),1))),axis = 1)
    #b = np.concatenate((exclu_ux,exclu_uy),axis = 1)
    #c = np.concatenate((b,exclu_uz),axis = 1)

    #np.savetxt(outfile, np.transpose([x,y,z]), '%.6f', delimiter = ',')
    np.savetxt(outfile, this_layer, '%.4f', delimiter = ',')
    outfile = "".join(outfile.split('.')[:-1]) + '-exclu.txt'
    #np.savetxt(outfile, np.concatenate((a,c),axis = 0), '%.6f', delimiter = ',')
    np.savetxt(outfile, exclu_layer, '%.4f', delimiter = ',')

def krigingInterp(ptfile, outfile, highres = 500, lowres = 10000):
    data = np.loadtxt(ptfile, delimiter = ',')
    x = data[:,2]
    y = data[:,3]
    z = data[:,5]

    # large region
    #xmin = -1336879
    #xmax = 218216
    #ymin = -285829
    #ymax = 748935

    # central deep region
    #xmin = -1052540
    #xmax = -699452
    #ymin = 178950
    #ymax = 398008

    # north deep region
    xmin = -844013
    xmax = -435095
    ymin = 294936
    ymax = 609489

    #xmin = min(x)
    #ymin = min(y)
    #xmax = max(x)
    #ymax = max(y)

    ## coarse interpolation
    #define grid
    xi = np.arange(xmin, xmax, highres)
    yi = np.arange(ymin, ymax, highres)

    newx = []
    newy = []
    newz = []
    for i in np.arange(xmin, xmax, lowres):
        for j in np.arange(ymin,ymax, lowres):
            idx = np.where(np.all([x > i ,x <= i + lowres,y>j,y<=j+lowres], axis = 0) == True)[0]

            if idx.size == 0:
                continue

            if len(idx) == 1:
                newx.append(x[idx].item())
                newy.append(y[idx].item())
                newz.append(z[idx].item())
            else:
                newx.append(np.mean(x[idx]))
                newy.append(np.mean(y[idx]))
                newz.append(np.mean(z[idx]))

    newx = np.array(newx)
    newy = np.array(newy)
    newz = np.array(newz) * np.sqrt(3.4)
    print newx.shape

    #define grid
    #xi = np.arange(xmin, xmax, highres)
    #yi = np.arange(ymin, ymax, highres)

    if newx.shape[0] >= 4000:
        return

    # kriging interpolation
    OK = OrdinaryKriging(newx,newy,newz, variogram_model = 'linear', verbose = True, 
        enable_plotting = False )
    #OK = OrdinaryKriging(x,y,z, variogram_model = 'linear', verbose = True, 
    #    enable_plotting = True )
    zi, ss = OK.execute('grid', xi, yi)

    [nx, ny] = zi.shape
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfile, ny, nx, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(zi)

    #ds.GetRasterBand(2).WriteArray(ss)
    ds.SetGeoTransform((
                        min(xi),    #0
                        xi[1]-xi[0], #1
                        0,          #2
                        min(yi),    #3
                        0,
                        yi[1]-yi[0]))

    sr = osr.SpatialReference()
    sr.ImportFromProj4("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs")
    sr_wkt = sr.ExportToWkt()
    ds.SetProjection(sr_wkt)

def interpolatePoints(ptfile, outfile, res = 300, smooth = 2):
    data = np.loadtxt(ptfile, delimiter = ',')
    x = data[:,2]
    y = data[:,3]
    z = data[:,5]
    npts = len(x)

    ## below is specific for study of PL in SPLD
    #xmin = 352229.121
    #ymin = -238184.108
    #xmax = 488030.041
    #ymax = -165350.708
    #res = 290.172906298066493

    xmin = min(x)
    ymin = min(y)
    xmax = max(x)
    ymax = max(y)
    # define grid.
    xi,yi = np.mgrid[xmin:xmax:res,ymin:ymax:res]

    tin = interpolate.LinearNDInterpolator(np.array([x,y]).T, z)
    zi = tin((xi,yi))
    filtz = ndimage.gaussian_filter(zi, sigma=(2,2), order=0)
    zi[~np.isnan(filtz)] = filtz[~np.isnan(filtz)]
    #zi = griddata((x, y), z, (xi,yi), method='cubic')
    zi = zi.T
    
    # Output file to Geotif
    # Add projection to the file
    [nx, ny] = zi.shape
    driver = gdal.GetDriverByName('GTiff')
    ds  = driver.Create(outfile, ny, nx, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(zi)
    ds.GetRasterBand(1).SetNoDataValue(-32768)
    ds.SetGeoTransform((xi.min(),res,0,yi.min(),0, res))
    
    sr = osr.SpatialReference()
    ## Projection below is for PL in SPLD
    #sr.ImportFromProj4("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs")
    sr.ImportFromProj4("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +a=3396000 +b=3396000 +units=m +no_defs")
    sr_wkt = sr.ExportToWkt()
    ds.SetProjection(sr_wkt)

    plotGeotiff(outfile)

    # output png file
    # needs add the point and x, y coordinates
def plotGeotiff(tiffile):
    ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
    geoinfo = ds.GetGeoTransform()
    zi = ds.GetRasterBand(1).ReadAsArray()
    [nx, ny] = zi.shape
    xmin = geoinfo[0]
    ymin = geoinfo[3]
    res = geoinfo[1]
    xmax = xmin + ny * res
    ymax = ymin + nx * res
    
    fig = plt.figure()
    ax = plt.gca()
    palette = plt.cm.jet
    palette.set_bad('w',1.0)
    im = ax.imshow(zi, cmap = palette, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(im, cax = cax)
    xarray = np.arange(xmin,xmax,res).astype(int)/1000
    yarray = np.arange(ymin,ymax,res).astype(int)/1000
    xticks = np.arange(0,ny,50).astype(int)
    yticks = np.arange(0,nx,50).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xarray[xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yarray[yticks])
    ax.set_xlabel('X coordinate (km)', fontsize = 12)
    ax.set_ylabel('Y coordinate (km)', fontsize = 12)
    ax.set_title('Interpolated Depth from MOLA DTM (m)')
    outfile = '.'.join(tiffile.split('.')[:-1]) + '.png'
    plt.savefig(outfile, transparent = True, bbox_inches = 'tight', pad_inches = 0)

#clusterPoints('/Users/xst/Desktop/CEP/subpt-thresh.txt', '/Users/xst/Desktop/CEP/SURFACE',-40,1,0.1,20)
#for k in xrange(1,2):
#   print k
#    ptfile = '/Users/xst/Desktop/CEP/SURFACE/surface-%s.txt' %k
#    outfile = '/Users/xst/Desktop/CEP/SURFACE/surface-%s_flt.txt' %k
#    filterPoints(ptfile, outfile, kz = 3, kv = 2)

#ptfile = '/Users/xst/Desktop/CEP/SURFACE/surface-1_flt.txt'
#outfile = '/Users/xst/Desktop/CEP/SURFACE/surface-1_flt-2.tif'
#krigingInterp(ptfile, outfile, highres = 1000, lowres = 3000)
#interpolatePoints(ptfile, outfile)

'''

clusterPoints('/Users/xst/Desktop/EP-all/Auto/A/subpt.txt', '/Users/xst/Desktop/EP-all/Auto/A/SURFACE',-40,1,0.1,10)
for k in xrange(1,2):
    ptfile = '/Users/xst/Desktop/EP-all/Auto/A/SURFACE/surface-%s.txt' %k
    outfile = '/Users/xst/Desktop/EP-all/Auto/A/SURFACE/surface-%s_flt.txt' %k
    filterPoints(ptfile, outfile, kz = 2, kv = 3)
'''

'''
clusterPoints('/Users/xst/Desktop/subpt.txt', '/Users/xst/Desktop/SURFACE',-40,1,0.05,5)
for k in xrange(1,2):
    ptfile = '/Users/xst/Desktop/SURFACE/surface-%s.txt' %k
    outfile = '/Users/xst/Desktop/SURFACE/surface-%s_flt.txt' %k
    filterPoints(ptfile, outfile, 30000, 30000, kz = 1, kv = 3)
'''
'''
clusterPoints('./Results-6/subpt.txt', './SURFACE-6')

for k in xrange(1,8):
    ptfile = './SURFACE-6/surface-%s.txt' %k
    outfile = './SURFACE-6/surface-%s_flt.txt' %k
    filterPoints(ptfile, outfile, kz = 2, kv = 3)
    ptfile = outfile
    outfile = './SURFACE-6/surface-%s_flt.tif' %k
    interpolatePoints(ptfile, outfile)
'''
'''
#clusterPoints('./Results-1/subpt.txt', './SURFACE-1')

for i in xrange(7):
    print('Interpolating %sth layers...' %(i+1))
    ptfile = './SURFACE-1/surface-%s.txt' %(i+1)
    outfile = './SURFACE-1/surface-%s%s.tif' %(i+1,i+1)
    krigingInterp(ptfile, outfile)



for i in xrange(7):
    print('Interpolating %sth layers...' %(i + 1))
    ptfile = './SURFACE-6/surface-%s.txt' %(i + 1)
    outfile = './SURFACE-6/surface-%s.tif' %(i + 1)
    krigingInterp(ptfile, outfile)
'''
