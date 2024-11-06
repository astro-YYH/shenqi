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
            + particles[i].GravPM[j] * kf->FgravkickB + kf->hydrokicks[particles[i].TimeBinHydro] * SPHP(i).HydroAccel[j];
    }
}