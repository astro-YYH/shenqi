/*! \file gravtree.c
 *  \brief main driver routines for gravitational (short-range) force computation
 *
 *  This contains the cuda-specific function calls. A separate file only because the C++ compilers cannot always understand cuda.
 */
#include <stdlib.h>
#include <math.h>

#include "gravshort2.hpp"

#include "treewalk2.cuh"

class GravTreeWalkGPU : public TreeWalkGPU <GravTreeWalkGPU, GravTreeQuery, GravTreeResult, GravLocalTreeWalk, GravTopTreeWalk, GravTreeParams, GravTreeOutput> {
    using GPUBase = TreeWalkGPU <GravTreeWalkGPU, GravTreeQuery, GravTreeResult, GravLocalTreeWalk, GravTopTreeWalk, GravTreeParams, GravTreeOutput>;

    public:
    using TreeWalkGPU::TreeWalkGPU;

    int * ev_count_exports(int * WorkSet, const int64_t WorkSetSize, particle_data * const parts)
    {
        return GPUBase::ev_count_exports_cpu(WorkSet, WorkSetSize, parts);
    }

    void ev_free_exports(int * exportcounts)
    {
        GPUBase::ev_free_exports_cpu(exportcounts);
    }

    int64_t ev_toptree(int * WorkSet, const int64_t WorkSetStart, const int64_t WorkSetSize, particle_data * const particles, int * exportcounts, ExportMemory2 * const exportlist)
    {
        return GPUBase::ev_toptree_cpu(WorkSet, WorkSetStart, WorkSetSize, particles, exportcounts, exportlist);
    }

    void ev_primary(int * WorkSet, int64_t WorkSetSize, particle_data * const particles)
    {
        int device;
        cudaError_t status = cudaGetDevice(&device);
        if(status != cudaSuccess)
            endrun(5, "Failed to get CUDA device for gravity prefetch: %s\n", cudaGetErrorString(status));
        this->prefetch_if_managed(this->output->Accel, PartManager->NumPart * sizeof(this->output->Accel[0]), device, "gravity acceleration");
        GPUBase::ev_primary(WorkSet, WorkSetSize, particles);
    }
};

/*! CUDA treewalk.
 */
void
grav_short_tree_cuda(const ActiveParticles * act, ForceTree * tree, GravTreeParams * priv, GravTreeOutput * output, particle_data * const parts, const size_t MaxExportBufferBytes, MPI_Comm comm)
{
        GravTreeWalkGPU tw("GRAVTREE", tree, *priv, output);
        tw.MaxExportBufferBytes = MaxExportBufferBytes;
        tw.run_on_queue(act->ActiveParticle, act->NumActiveParticle, parts, comm);
        tw.print_stats("/Tree", comm);
}
