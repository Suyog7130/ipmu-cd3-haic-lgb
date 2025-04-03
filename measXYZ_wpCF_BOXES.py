import numpy as np
import numpy as np
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from astropy.table import Table
import sys


# Uchuu uses the following cosmology: 
# Planck2015 (table 4, rightmost column)
#Ω_m = 0.3089 Ω_L = 0.6911 h = 0.6774
#σ_8 = 0.8159 Ω_b = 0.0486 ns = 0.9667
#Linear Power Spectrum
#z_init = 127 (2LPT)
## Uchuu is 2000 Mpc/h on a side 
Om0 = 0.3089
h = 0.6774
H0 = 100*h
z_sim = 2.95
ns = 0.9667
sigma8 = 0.8159
Ob0 = 0.0486
Tcmb0 = 2.718

LOS = 'z' # Line-of-sight to shift into redshift-space
Lbox = 250
Lfull = 2000.0 #Mpc/h


ROOT = '/Users/ave_astro/Desktop/IPMU/PFS'
DATAdir = 'Uchuu/data'

GAL = Table.read('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_z2p95_sSFR_boxes.fits')
RAND = Table.read('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_randoms_z2p95_sSFR_boxes.fits')


idGbox = GAL['boxID'].astype(int)
idRbox = RAND['boxID'].astype(int)
IDboxes = np.unique(idRbox)

aa=1./(1.+z_sim)
sf=0.01/np.sqrt(Om0/aa+(1-Om0)*aa**2) # Calculates H(z)

## Parameters to measure projected correlations function in each sub-volume.
boxsize = 250
nthreads = 4
pimax = 100
nrpbins = 13
rpmin = -1.0
rpmax = 1.5
bins = np.logspace(rpmin, rpmax, nrpbins + 1)
autocorr = 1
crosscorr = 0


## Shift positions into redshift-space for the galaxies. Randoms do not need to be shifted.
# Assumes periodic boundary conditions were objects that are shifted out of the box show up on the other side, e.g., z' = z+vz > 2000 will be set to z'' = z' - 2000, etc.
## Shifting along each axis and averaging the result
LOS = ['x','y','z']
wp = []
rp = []
for jj in range(0,len(LOS)):
    xG = GAL['x']
    yG = GAL['y']
    zG = GAL['z']
    if LOS[jj] == 'x':
        xG = xG + GAL['vx']*sf
        xG[xG<0] = xG[xG<0]+2000
        xG[xG>2000] = xG[xG>2000]-2000
    if LOS[jj] == 'y':
        yG = yG + GAL['vy']*sf
        yG[yG<0] = yG[yG<0]+2000
        yG[yG>2000] = yG[yG>2000]-2000
    if LOS[jj] == 'z':
        zG = zG + GAL['vz']*sf
        zG[zG<0] = zG[zG<0]+2000
        zG[zG>2000] = zG[zG>2000]-2000

    rpTMP = []
    wpTMP = []
    for ii in range(0,len(IDboxes)):
        print('{}{}'.format(LOS[jj],IDboxes[ii]))
        xGtmp = xG[idGbox==IDboxes[ii]].value.astype(float)
        yGtmp = yG[idGbox==IDboxes[ii]].value.astype(float)
        zGtmp = zG[idGbox==IDboxes[ii]].value.astype(float)
        xRtmp = xR[idRbox==IDboxes[ii]].value.astype(float)
        yRtmp = yR[idRbox==IDboxes[ii]].value.astype(float)
        zRtmp = zR[idRbox==IDboxes[ii]].value.astype(float)
        Ngal = len(xGtmp)
        Nrand = len(xRtmp)
        
        DD_counts = DDrppi(autocorr, nthreads, pimax, bins, xGtmp, yGtmp, zGtmp, output_rpavg=True, boxsize=boxsize)
        DR_counts = DDrppi(crosscorr, nthreads, pimax, bins, xGtmp, yGtmp, zGtmp, X2=xRtmp, Y2=yRtmp, Z2=zRtmp, output_rpavg=True, boxsize=boxsize)
        RR_counts = DDrppi(autocorr, nthreads, pimax, bins, xRtmp, yRtmp, zRtmp, output_rpavg=True, boxsize=boxsize)
        
        wpTMP.append(convert_rp_pi_counts_to_wp(Ngal, Ngal, Nrand, Nrand, DD_counts, DR_counts, DR_counts, RR_counts, nrpbins, pimax))
        #for w in wp: print("{0:10.6f}".format(w))
        
        rpTMP.append(bins[0:-1]+(bins[1:]-bins[:-1])/2.)

    wp.append(wpTMP)
    rp.append(rpTMP)

wpAVG = np.average(wp,axis=0)
rp = rp[0][0]

cov_matrix = np.cov(wpAVG, rowvar=False, ddof=1)
wp_mean = np.mean(wpAVG, axis=0) 


import datetime
now = datetime.datetime.now()
NOW = now.strftime("%Y-%m-%d")
wp_out = np.vstack((rp,wp_mean)).T
comments = ["# Measurement of Uchuu UM projected correlation function in 512 sub-volumes with Lbox = 250 Mpc/h by KSM on {}".format(NOW),"#Averaged over LOS (X,Y,Z) and Nrand is 50xNgal"]
header = '\n'.join(comments)
np.savetxt(ROOT+'/LBG_Clustering/measured/'+'Uchuu_wpCF_sSFR_VLTSboxes_sftAVG.dat',wp_out,header=header)
comments = ["# Measurement of Uchuu UM covariance of wp in 512 sub-volumes with Lbox = 250 Mpc/h by KSM on {}".format(NOW),"#Averaged over LOS (X,Y,Z) and Nrand is 50xNgal"]
header = '\n'.join(comments)
np.savetxt(ROOT+'/LBG_Clustering/measured/'+'Uchuu_wpCOV_sSFR_VLTSboxes_sftAVG.dat',cov_matrix,header=header)


