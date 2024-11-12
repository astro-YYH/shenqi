/*Prototypes and structures for the wind model*/

#include <math.h>
#include <string.h>
#include <omp.h>
#include "winds.h"
#include "physconst.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "timebinmgr.h"
#include "walltime.h"
#include "density.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "blackhole.h"
#include "slotsmanager_dev.h"


/*Parameters of the wind model*/
__device__ static struct WindParams wind_params_dev;

// wrapper function to pass the wind parameters to the device
void run_assign_wind_params(struct WindParams * wind_params_ptr)
{
    cudaMemcpyToSymbol(wind_params_dev, wind_params_ptr, sizeof(struct WindParams));
}

__device__ int
winds_is_particle_decoupled_device(int i, struct particle_data * particles)
{
    if(HAS(wind_params_dev.WindModel, WIND_DECOUPLE_SPH)
        && particles[i].Type == 0 && SPHP_dev(i).DelayTime > 0)
            return 1;
    return 0;
}