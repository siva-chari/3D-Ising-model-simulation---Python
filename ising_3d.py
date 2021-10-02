#
# Siva
# 3 Sept, 2021
#
'''
This code simulates the Ising model in 3-dimensions.
'''

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#============= define functions here.======================#

@jit
def initialize_spin_config(lx,ly,lz):
	config = np.zeros((lx, ly, lz))
	
	for i in range(lx):
		for j in range(ly):
			for k in range(lz):
				if (np.random.uniform() > 0.5):
					config[i,j,k] = 1.0
				else:
					config[i,j,k] = -1.0
	return config
	
@jit
def initialize_all_ones(lx,ly,lz):
	config = np.ones((lx,ly,lz))
	return config


@jit
def get_energy(config,Jex,lx,ly,lz,mag_field):
	esum = 0.0
	#Jex = 1.0 # exchange interaction strength.
	
	for i in range(lx):
		for j in range(ly):
			for k in range(lz):
				# neighboring spin values, along x-dimension.
				iup = config[(i+1)%lx, j, k]
				idown = config[(i-1)%lx, j, k]
				
				# along y-dimension
				jup = config[i, (j+1)%ly, k]
				jdown = config[i, (j-1)%ly, k]
				
				# along z-dimension
				kup = config[i, j, (k+1)%lz]
				kdown = config[i, j, (k-1)%lz]
				
				# nearest neighbor sum
				nnsum = iup + idown + jup + jdown + kup + kdown
				esum = esum -Jex * config[i,j,k] * nnsum * 0.5  # 0.5 takes care of the double counting.
				
	# for any magnetic field is present, then.
	if (mag_field > 0):
		esum = esum - mag_field * np.sum(config[:,:,:])
		
	return esum

@jit
def do_one_MC_move(config,Jex,lx,ly,lz,mag_field,temperature):
	i = np.random.randint(0,lx)
	j = np.random.randint(0,ly)
	k = np.random.randint(0,lz)
	
	#energy before flipping the spin.
	E0 = get_energy(config,Jex,lx,ly,lz,mag_field)
	
	#make the flip.
	config[i,j,k] = -config[i,j,k]
	
	#new energy now is E1.
	E1 = get_energy(config,Jex,lx,ly,lz,mag_field)
	
	dE = E1-E0
	kB = 1.0
	bta = 1.0/(kB*temperature)
	
	#accept/ reject as per the Metropolis algorithm.
	# if ((dE < 0.0) or (np.random.uniform() < np.exp(-bta*dE))):
		# #accept.
		# E0 = E1
		# enrg = E1
	# else:
		# # reject the flip.
		# config[i,j,k] = -config[i,j,k]
		# enrg = E0
	
	if (dE < 0):
		# accept.
		E0 = E1
		enrg = E1
	else:
		if(np.random.uniform() > np.exp(-bta*dE)):
			# accept.
			E0 = E1
			enrg = E1
		else:
			# reject
			config[i,j,k] = -config[i,j,k]
			enrg = E0
	
	#magn = np.sum(config[:,:,:])
	return config

@jit
def make_one_MC_sweep(config,Jex,lx,ly,lz,mag_field,temperature):
	N = lx*ly*lz
	for istep in range(N):
		config = do_one_MC_move(config,Jex,lx,ly,lz,mag_field,temperature)
		
	enrg = get_energy(config,Jex,lx,ly,lz,mag_field)
	magn = np.abs(np.sum(config[:,:,:]))
	return [config,enrg,magn]
	

def open_out_files():
	# filenames.
	global out_thermo_file
	global fout_thermo
	
	out_thermo_file = "out_thermo.dat"
	
	# file pointers.
	fout_thermo = open(out_thermo_file, 'w+')
	return 


def close_all_files():
	global out_thermo
	
	fout_thermo.close()
	return 
	

#============= MAIN PROGRAM ===============================#

# system parameters.-----------------------------
[lx, ly, lz] = 4, 4, 4
Tinit = 1.5
Tfinal = 7.0
dT = 0.1  # because 
kB = 1.0
Jex = 1.0
mag_field = 0.0

N = lx*ly*lz

# simulation parameters ---------------------------
nMCsweeps = 15000
nEquil_Sweeps = 10000
nTempSteps = int(np.abs((Tfinal - Tinit)/dT))  # no. of steps from Tinit to Tmax, with dT.


# file names ---------------------------------------
#out_thermo_file = "out_thermo.dat"

#fout_thermo = open(out_thermo_file, "w+")

open_out_files()

#---------------------------------------------------
# initialize the system configuration.
config = np.zeros((lx, ly, lz))
#config = initialize_spin_config(lx,ly,lz)
config = initialize_all_ones(lx,ly,lz)

# equilibrate the initial configuration.
for i in range(100):
	[config,E0,M] = make_one_MC_sweep(config,Jex,lx,ly,lz,mag_field,Tinit)

# get the energy of the initial configuration.
#E0 = get_energy(config,Jex,lx,ly,lz,mag_field)

# Now, apply the metropolis algorithm.

for istep in range(nTempSteps+1):
	Tcurr = Tinit + istep*dT
	bta = 1.0/(kB*Tcurr)
	
	# zero out the average sums.
	eavg = 0.0
	mavg = 0.0
	eavg2 = 0.0
	mavg2 = 0.0
	
	for isweep in range(nMCsweeps):
		[config,E,M] = make_one_MC_sweep(config,Jex,lx,ly,lz,mag_field,Tcurr)
		
		if (isweep > nEquil_Sweeps):
			eavg = eavg + E
			mavg = mavg + M
			eavg2 = eavg2 + E**2.0
			mavg2 = mavg2 + M**2.0
	
	avgOver = (nMCsweeps - nEquil_Sweeps)
	eavg = eavg/avgOver
	eavg_perSpin = eavg/(N*1.0)
	
	eavg2 = eavg2/avgOver
	eavg2_perSpin = eavg2/(N**2.0)
	
	mavg = mavg/avgOver
	mavg_perSpin = mavg/(N*1.0)
	
	mavg2 = mavg2/avgOver
	mavg2_perSpin = mavg2/(N**2.0)
	
	varE = eavg2 - eavg**2
	varM = mavg2 - mavg**2
	
	sp_heat = varE * (bta**2.0)*kB
	cv = sp_heat/(N**2)
	suscep = varM * bta
	chi = suscep/(N**2)
	
	fout_thermo.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(Tcurr,eavg_perSpin,mavg_perSpin,cv,chi))
	fout_thermo.flush()
	#print(Tcurr,eavg,mavg,cv,chi)


# closing all the opened files.
close_all_files()
#fout_thermo.close()
