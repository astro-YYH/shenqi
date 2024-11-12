#ifndef TREEWALK_KERNEL_H
#define TREEWALK_KERNEL_H

#include <cuda_runtime.h>
#include "treewalk.h"
#include "partmanager.h"  // To access particle_data structure
#include "gravity.h"

// Declaration of the GPU kernel
// __global__ void treewalk_kernel(TreeWalk *tw, struct particle_data *particles, int *workset, size_t workset_size);

void run_treewalk_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, const double GravitySoftening, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions);

void run_gravshort_fill_ntab(const enum ShortRangeForceWindowType ShortRangeForceWindowType, const double Asmth);

void run_treewalk_secondary_kernel(TreeWalk *tw, struct particle_data *particles, const struct gravshort_tree_params * TreeParams_ptr, char* databufstart, char* dataresultstart, const int64_t nimports_task);

// Host function to initialize the device pointer
void set_device_hydro_part(struct sph_particle_data * host_ptr, struct star_particle_data * star_ptr, struct bh_particle_data * bh_ptr);

void run_treewalk_density_kernel(TreeWalk *tw, struct particle_data *particles, const struct density_params * DensityParams_ptr, unsigned long long int *maxNinteractions, unsigned long long int *minNinteractions, unsigned long long int *Ninteractions);

#endif  // TREEWALK_KERNEL_H