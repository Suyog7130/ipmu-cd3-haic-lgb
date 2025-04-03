# Python script to cut up the Uchuu box into smaller subboxes with length Lbox. Creates randoms for the Uchuu box and also finds the ID for the subbox. Prints .fits file with galaxy/random ID, X/Y/Z positions, and for the galaxies, vx, vy, vz for redshift-space measurements. -KSM
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from hmf import MassFunction  
from hmf import cosmo
import os
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import latex
plt.rcParams['text.usetex'] = False
import sys
import scipy.stats as scs
from astropy.table import Table

ROOT = '/Users/ave_astro/Desktop/IPMU/PFS'
DATAdir = 'Uchuu/data'


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
LOS = 'z'
Lbox = 250 # This produces 512 subvolumes from which we can create a covariance matrix for the summary statistics.
# 250 Mpc/h is about areaBOX = 10.7 deg^2 in surface area at z_sim = 2.95 
Lfull = 2000.0 #Mpc/h


CUT = 'VLTS'

## Simulation details
cosmoFLCDM=FlatLambdaCDM(H0=H0,Om0=Om0,Ob0=Ob0,Tcmb0=Tcmb0)
Dv = cosmoFLCDM.comoving_distance(z_sim).value*h

arclenF = Lfull/Dv*180./np.pi # degrees
areaF = arclenF**2 # deg^2 surface area 
AREAfull = areaF*60**2 # arcmin^2 
VOLfull=Lfull**3 # (Mpc/h)^3

arclenB = Lbox/Dv*180./np.pi # degrees
areaB = arclenB**2 # deg^2 surface area 
AREAbox = areaB*60**2 # arcmin^2
VOLbox=Lbox**3 # (Mpc/h)^3

Nbins = int(Lfull/Lbox)
BINcell = np.arange(0,Lfull+Lbox,Lbox)

## Data files
# Data1 ID,upid,Mvir,sm,icl,sfr,obs_sm,obs_sfr,obs_uv 
data1 =   h5py.File('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_z2p95_data1.h5','r')
# Data2 x,y,z
data2 = h5py.File('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_z2p95_data2.h5','r')
# Data3 ID,vx,vy,vz,Mpeak,Vmax_Mpeak,vmax,A_UV
data3 =  h5py.File('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_z2p95_data3.h5','r')

ID_UM = data1['id'][:]
xUM = data2['x'][:]
yUM = data2['y'][:]
zUM = data2['z'][:]
vxUM = data3['vx'][:]
vyUM = data3['vy'][:]
vzUM = data3['vz'][:]
obs_sfrUM = data1['obs_sfr'][:]
obs_smUM = data1['obs_sm'][:]

cut1 = obs_smUM > 0
cut2 = obs_sfrUM > 0
ID_UM = ID_UM[cut1*cut2]
xUM = xUM[cut1*cut2]
yUM = yUM[cut1*cut2]
zUM = zUM[cut1*cut2]
vxUM = vxUM[cut1*cut2]
vyUM = vyUM[cut1*cut2]
vzUM = vzUM[cut1*cut2]
obs_sfrUM = obs_sfrUM[cut1*cut2]
obs_smUM = obs_smUM[cut1*cut2]

## Select objects based on surface density abundance per sSFR to match observations of LBGs from Bielby+13, LBG targets observed to be ~0.2/arcmin^2

NbinsSFR = 100
obs_ssfrUM =np.log10(obs_sfrUM/obs_smUM)
sSFRbins=np.linspace(obs_ssfrUM.min(),obs_ssfrUM.max(),NbinsSFR)
sSFRcount,sSFRedge = np.histogram(obs_ssfrUM,bins=sSFRbins)
sSFRwid = sSFRedge[1:]-sSFRedge[:-1]
sSFRcen = sSFRedge[:-1]+sSFRwid/2.
sSFRden = sSFRcount/sSFRwid/VOLfull

sSFRcdf = np.zeros((len(sSFRcen)))
for i in range(0,len(sSFRcen)):
   sSFRcdf[i] = len(obs_ssfrUM[obs_ssfrUM>=sSFRbins[i]])/AREAfull

if CUT=='VLTS':
    sSFRcenCUT = sSFRcen[sSFRcdf>0.2][-1]


cut = obs_ssfrUM>=sSFRcenCUT
IDssfr = ID_UM[cut] 
xCssfr = xUM[cut]
yCssfr = yUM[cut]
zCssfr = zUM[cut]
vxCssfr = vxUM[cut]
vyCssfr = vyUM[cut]
vzCssfr = vzUM[cut]

## Split sSFR-selected galaxies into sub-volumes

Ngal = len(xCssfr)
Nbins = int(Lfull/Lbox)
BINcell = np.arange(0,Lfull+Lbox,Lbox)

IDboxGAL = np.zeros(Ngal) - 99

# Only using binned_statistic_2d to digitize the x and y coordinates into cells
stat,xE,yE,BINnumGAL = scs.binned_statistic_2d(xCssfr,yCssfr,IDssfr,bins=BINcell)
zDIGgal = np.digitize(zCssfr,bins=BINcell)

idBOXgal = []
numBOXgal = []
for i in range(1,Nbins+1):
    for j in range(1,Nbins+1):
        NUM = '{}{}'.format(i,j) 
        zDIGtmp = zDIGgal[BINnumGAL==int(NUM)]
        IDtmp = IDssfr[BINnumGAL==int(NUM)]
        for k in range(1,Nbins+1): 
             numBOXgal.append(NUM+'{}'.format(k))
             idBOXgal.append(IDtmp[zDIGtmp==k])


for i in range(0,len(idBOXgal)):
    IDboxGAL[np.isin(IDssfr,idBOXgal[i])] = numBOXgal[i]

Tgal = Table([IDssfr, IDboxGAL,xCssfr, yCssfr, zCssfr, vxCssfr, vyCssfr, vzCssfr],names=('ID','boxID','x', 'y','z', 'vx', 'vy', 'vz'))
Tgal.write('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_z2p95_sSFR_boxes.fits', overwrite=True)

## Create randoms and split into sub-volumes

Nrand = 50*Ngal
seed = 42
np.random.seed(seed)
Xrand = np.random.uniform(0, Lfull, Nrand)
Yrand = np.random.uniform(0, Lfull, Nrand)
Zrand = np.random.uniform(0, Lfull, Nrand)
IDrand = np.arange(0,Nrand,1)

stat,xE,yE,BINnumRAND = scs.binned_statistic_2d(Xrand,Yrand,IDrand,bins=BINcell)
zDIGrand = np.digitize(Zrand,bins=BINcell)

IDboxRAND = np.zeros(Nrand) - 99

idBOXrand = []
numBOXrand = []
for i in range(1,Nbins+1):
    for j in range(1,Nbins+1):
        NUM = '{}{}'.format(i,j)
        zDIGtmp = zDIGrand[BINnumRAND==int(NUM)]
        IDtmp = IDrand[BINnumRAND==int(NUM)]
        for k in range(1,Nbins+1):
             numBOXrand.append(NUM+'{}'.format(k))
             idBOXrand.append(IDtmp[zDIGtmp==k])


for i in range(0,len(idBOXrand)):
    IDboxRAND[np.isin(IDrand,idBOXrand[i])] = numBOXrand[i]

Trand = Table([IDrand, IDboxRAND,Xrand, Yrand, Zrand],names=('ID','boxID','x', 'y','z'))
Trand.write('{}'.format(ROOT)+'/Uchuu/UM/Uchuu_UM_randoms_z2p95_sSFR_boxes.fits', overwrite=True)



