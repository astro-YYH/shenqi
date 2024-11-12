#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "physconst.h"
#include "walltime.h"
#include "cooling.h"
#include "density.h"
#include "treewalk.h"
#include "timefac.h"
#include "slotsmanager.h"
#include "timestep.h"
#include "utils.h"
#include "gravity.h"
#include "winds.h"
#include <cuda_runtime.h>
#include "slotsmanager_dev.h"

// Define a global device pointer
__device__ struct sph_particle_data* sph_particles;
__device__ struct star_particle_data* star_particles;
__device__ struct bh_particle_data* bh_particles;

/* Get the predicted velocity for a particle
 * at the current Force computation time ti,
 * which always coincides with the Drift inttime.
 * For hydro forces.*/
__device__ void
SPH_VelPred_device(int i, MyFloat * VelPred, const struct kick_factor_data * kf, struct particle_data *particles)
{
    int j;
    /* Notice that the kick time for gravity and hydro may be different! So the prediction is also different*/
    for(j = 0; j < 3; j++) {
        VelPred[j] = particles[i].Vel[j] + kf->gravkicks[particles[i].TimeBinGravity] * particles[i].FullTreeGravAccel[j]
            + particles[i].GravPM[j] * kf->FgravkickB + kf->hydrokicks[particles[i].TimeBinHydro] * SPHP_dev(i).HydroAccel[j];
    }
}

__device__ static void
density_reduce_device(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode, TreeWalk * tw, struct particle_data * particles)
{
    TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->NumNgb[place], remote->Ngb);
    TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->DhsmlDensityFactor[place], remote->DhsmlDensity);

    if(particles[place].Type == 0)
    {
        TREEWALK_REDUCE(SPHP_dev(place).Density, remote->Rho);

        TREEWALK_REDUCE(SPHP_dev(place).DivVel, remote->Div);
        int pi = particles[place].PI;
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][0], remote->Rot[0]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][1], remote->Rot[1]);
        TREEWALK_REDUCE(DENSITY_GET_PRIV(tw)->Rot[pi][2], remote->Rot[2]);

        MyFloat * gradrho = DENSITY_GET_PRIV(tw)->GradRho;

        if(gradrho) {
            TREEWALK_REDUCE(gradrho[3*pi], remote->GradRho[0]);
            TREEWALK_REDUCE(gradrho[3*pi+1], remote->GradRho[1]);
            TREEWALK_REDUCE(gradrho[3*pi+2], remote->GradRho[2]);
        }

        /*Only used for density independent SPH*/
        if(DENSITY_GET_PRIV(tw)->DoEgyDensity) {
            TREEWALK_REDUCE(SPHP_dev(place).EgyWtDensity, remote->EgyRho);
            TREEWALK_REDUCE(SPHP_dev(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
        }
    }
    else if(particles[place].Type == 5)
    {
        TREEWALK_REDUCE(BHP_dev(place).Density, remote->Rho);
        TREEWALK_REDUCE(BHP_dev(place).DivVel, remote->Div);
    }
}