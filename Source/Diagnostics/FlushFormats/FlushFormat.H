#ifndef WARPX_FLUSHFORMAT_H_
#define WARPX_FLUSHFORMAT_H_

#include "Particles/MultiParticleContainer.H"
#include "Diagnostics/ParticleDiag/ParticleDiag.H"

class FlushFormat
{
public:
    /** Flush fields and particles to file */
    virtual void WriteToFile (
        const amrex::Vector<std::string> varnames,
        const amrex::Vector<amrex::MultiFab>& mf,
        amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<int> iteration, const double time,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
        const std::string prefix,
        bool plot_raw_fields,
        bool plot_raw_fields_guards,
        bool plot_raw_rho, bool plot_raw_F) const = 0;

     virtual ~FlushFormat() {}
};

#endif // WARPX_FLUSHFORMAT_H_
