import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/home/frieren/Documents/projects/icenumerics/')
sys.path.insert(0,'../auxnumerics/')
sys.path.insert(0, '../')
import icenumerics as ice
from parameters import params
import vertices as vrt
import auxiliary as aux

ureg = ice.ureg
idx = pd.IndexSlice

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'xtick.labelsize':15,
    'ytick.labelsize':15,
    'axes.labelsize':20,
})

# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def geometrical_part(r,B):
    """ 
        Geometrical part of the force 
        (rxB)xB - 2r + 5r (rxB)^2
    """
    rcb = np.cross(r,B)
    return 2*np.cross(rcb,B) - 2*r + 5*r*np.dot(rcb,rcb)

def forces_from_positions(params, positions, Bhat = [1,0,0], periodic=False):
    """ Compute the total force on all the particles """
    
    # get the dimensional part in pN * nm
    factor = (3*params['mu0']*params['m']**2)/(4*np.pi)
    factor = factor.to(ureg.piconewton * ureg.micrometer**4)
    
    forces = [] # init

    L = params['lattice_constant'].magnitude * params['size']
    
    # loop all particles
    for i,r1 in enumerate(positions):

        force = np.array([0,0,0])*ureg.piconewton

        # get the contribution from all particles to particle i
        for j,r2 in enumerate(positions):

            if periodic:
                xij = r1 - r2
                ox = np.array([xij[0], xij[0]+L, xij[0]-L])
                oy = np.array([xij[1], xij[1]+L, xij[1]-L])
                oz = np.array([xij[2], xij[2]+L, xij[2]-L])
                
                ix = np.argmin(np.abs(ox))
                iy = np.argmin(np.abs(oy))
                iz = np.argmin(np.abs(oz))
                
                R = np.array([ox[ix], oy[iy], oz[iz]])
            else:
                R = r1 - r2 

            distance = np.linalg.norm(R)
            # the algorithm eventually gets to itself, 
            # so I just want to skip this case
            if distance == 0:
                continue

            rhat = R/distance 
            distance = distance*ureg.um
            force = force + factor/distance**4 * geometrical_part(rhat,Bhat)

        forces.append(force.magnitude) 
    return np.asarray(forces)

def forces_from_file(params, filepath, field_dir, periodic = False):
    # import the trj and correct dx,dx,dz to have trap_separation
    trj = vrt.trj2trj( pd.read_csv(filepath,index_col=['id']) )
    trj[['dx','dy','dz']] = trj[['dx','dy','dz']].apply(lambda x: params['trap_sep'].magnitude*x)

    # get the position vectors
    # so that each row is a vector [x,y,z]
    pos = np.vstack([
        (trj['x']+trj['cx']).to_numpy(),
        (trj['y']+trj['cy']).to_numpy(),
        (trj['z']+trj['cz']).to_numpy()
    ]).T

    # compute the forces, and get forces and magnitudes
    forces = forces_from_positions(params,pos,Bhat=field_dir, periodic=periodic)
    force_dir = np.asarray([f/np.linalg.norm(f) for f in forces])
    force_mag = np.round( np.array([np.linalg.norm(f) for f in forces]),4)

    return force_dir, force_mag

def choose_trj(name,extension, exclusion):
    a = name.endswith(extension)
    b = name not in exclusion
    return (a and b)

def draw_forces(params, filepath, targetfile, fdirs, norm_mags):

    trj = vrt.trj2trj( pd.read_csv(filepath,index_col=['id']) )
    trj[['dx','dy','dz']] = trj[['dx','dy','dz']].apply(lambda x: params['trap_sep'].magnitude*x)

    # get the position vectors
    # so that each row is a vector [x,y,z]
    pos = np.vstack([
        (trj['x']+trj['cx']).to_numpy(),
        (trj['y']+trj['cy']).to_numpy(),
        (trj['z']+trj['cz']).to_numpy()
    ]).T


    # plot the figure
    norm = plt.Normalize(0,1)
    fig, ax = plt.subplots(figsize=(5,5))
    ice.draw_frame(trj, frame_no=0,
                   radius=params["particle_radius"].magnitude,
                   cutoff=params["trap_sep"].magnitude/2,
                   particle_color='#75b7ea',
                   trap_color='gray',
                   ax = ax)
    ax.quiver(pos[:,0],pos[:,1],fdirs[:,0],fdirs[:,1], norm_mags, cmap='viridis', norm=norm)

    fig.savefig(targetfile,bbox_inches='tight',dpi=300)

# =============================================================================
# MAIN
# =============================================================================


DRIVE = '/home/frieren/Dropbox/'
PROJECT = 'mnt/thesis/src/vertex_configs'

excluded = ['3b3.csv','3b8.csv']
PERIODIC = ['10.csv']
trj_files = ['10.csv']

# in this section, i will compute the forces,
# and get the maximum of all forces for renormalization
dirs_dict = {}
mags_dict = {}

global_maxforce = 0 # initialize
for file in trj_files:
    filepath = os.path.join(DRIVE,PROJECT,file)

    pbc = file in PERIODIC
    print(file, pbc)

    force_dirs, force_mags = forces_from_file(params,filepath,field_dir=[1,0,0], periodic = pbc)
    dirs_dict[file] = force_dirs
    mags_dict[file] = force_mags

    # compute global magnitude maximum without saving to list
    local_maxforce = np.max(force_mags)
    if local_maxforce > global_maxforce:
        global_maxforce = local_maxforce

# go file by file again, but plotting with the correct scale
for file in trj_files:
    filepath = os.path.join(DRIVE,PROJECT,file)

    force_dirs = dirs_dict[file]
    force_mags = mags_dict[file]

    renorm_mags = force_mags/global_maxforce

    # forces = force_dirs * renorm_mags[:,np.newaxis]
    tgfile = file[:-4] + '.pdf'
    draw_forces(params,filepath,os.path.join(DRIVE,PROJECT,tgfile),force_dirs,renorm_mags)
