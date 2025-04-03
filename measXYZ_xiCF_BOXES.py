import numpy as np
from Corrfunc.theory.DD import DD
from Corrfunc.io import read_catalog
from Corrfunc.utils import convert_3d_counts_to_cf
import traceback
from astropy.table import Table


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
xR = RAND['x']
yR = RAND['y']
zR = RAND['z']

idGbox = GAL['boxID'].astype(int)
idRbox = RAND['boxID'].astype(int)
IDboxes = np.unique(idRbox)

aa=1./(1.+z_sim)
sf=0.01/np.sqrt(Om0/aa+(1-Om0)*aa**2) # Calculates H(z)


nthreads = 5
Lbox = 250
rmin = -0.7
rmax = 1.477
nrbins = 11

bins = np.logspace(rmin, rmax, nrbins + 1)
ravgOUT = True
autocorr = 1
crosscorr = 0

LOS = ['x','y','z']
s = []
xi = []
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

    sTMP = []
    xiTMP = []
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

        DD_counts = DD(autocorr, nthreads, bins, xGtmp, yGtmp, zGtmp, output_ravg=ravgOUT, boxsize=Lbox)
        DR_counts = DD(crosscorr, nthreads, bins, xGtmp, yGtmp, zGtmp, X2=xRtmp, Y2=yRtmp, Z2=zRtmp, output_ravg=ravgOUT, boxsize=Lbox)
        RR_counts = DD(autocorr, nthreads, bins, xRtmp, yRtmp, zRtmp, output_ravg=ravgOUT,boxsize=Lbox)

        xiTMP.append(convert_3d_counts_to_cf(Ngal, Ngal, Nrand, Nrand, DD_counts,DR_counts,DR_counts, RR_counts))
        sTMP.append(DD_counts['ravg'])

    s.append(sTMP)
    xi.append(xiTMP)



xiAVG = np.average(xi,axis=0)
s = s[0][0]

cov_matrix = np.cov(xiAVG, rowvar=False, ddof=1)
xi_mean = np.mean(xiAVG, axis=0)


import datetime
now = datetime.datetime.now()
NOW = now.strftime("%Y-%m-%d")
xi_out = np.vstack((s,xi_mean)).T
comments = ["# Measurement of Uchuu UM 3D redshift-space 2PCF function in 512 sub-volumes with Lbox = 250 Mpc/h by KSM on {}".format(NOW),"#Averaged over LOS (X,Y,Z) and Nrand is 50xNgal"]
header = '\n'.join(comments)
np.savetxt(ROOT+'/LBG_Clustering/measured/'+'Uchuu_xiCF_sSFR_VLTSboxes_sftAVG.dat',xi_out,header=header)
comments = ["# Measurement of Uchuu UM covariance of 3D redshift-space 2PCF in 512 sub-volumes with Lbox = 250 Mpc/h by KSM on {}".format(NOW),"#Averaged over LOS (X,Y,Z) and Nrand is 50xNgal"]
header = '\n'.join(comments)
np.savetxt(ROOT+'/LBG_Clustering/measured/'+'Uchuu_xiCOV_sSFR_VLTSboxes_sftAVG.dat',cov_matrix,header=header)

