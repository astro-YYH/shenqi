#ifndef SLOTSMANAGER_DEV_H
#define SLOTSMANAGER_DEV_H

extern __device__ struct sph_particle_data* sph_particles;
extern __device__ struct star_particle_data* star_particles;
extern __device__ struct bh_particle_data* bh_particles;

/* shortcuts for accessing different slots directly by the index */
#define SphP_dev ((struct sph_particle_data*) sph_particles)
#define StarP_dev ((struct star_particle_data*) star_particles)
#define BhP_dev ((struct bh_particle_data*) bh_particles)

/* shortcuts for accessing slots from base particle index */
#define SPHP_dev(i) SphP_dev[particles[i].PI]
#define BHP_dev(i) BhP_dev[particles[i].PI]
#define STARP_dev(i) StarP_dev[particles[i].PI]


#endif