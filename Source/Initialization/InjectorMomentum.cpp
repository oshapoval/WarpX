/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Maxence Thevenet,
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorMomentum.H"

#include <AMReX_GpuDevice.H>
#include <AMReX_OpenMP.H>

#include <memory>
#include <new>

void InjectorMomentum::clear () // NOLINT(readability-make-member-function-const)
{
    switch (type)
    {
    case Type::parser:
    case Type::gaussian:
    case Type::gaussianflux:
    case Type::uniform:
    case Type::juttner:
    case Type::constant:
    {
        break;
    }
    case Type::maxwellian:
    {
        object.maxwellian.clear();
        break;
    }
    }
#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    clearThreadCopies();
#endif
}

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
void InjectorMomentum::clearThreadCopies ()
{
    for (auto* q : inj_mom_omp) {
        q->~InjectorMomentum();
    }
    inj_mom_omp.clear();
    inj_mom_data.reset();
}
#endif

bool InjectorMomentum::needPreparation () const noexcept
{
    return type == Type::maxwellian && object.maxwellian.needPreparation();
}

void InjectorMomentum::prepare (
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    amrex::IntVect const& ngrow,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (!needPreparation()) {
        return;
    }

    object.maxwellian.prepare(grids, dmap, ngrow, get_zlab);

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    clearThreadCopies();
    if (distributed()) {
        auto const nthreads = amrex::OpenMP::get_max_threads();
        inj_mom_data = std::unique_ptr<void, amrex::DataDeleter>(
            amrex::The_Cpu_Arena()->alloc(sizeof(InjectorMomentum) * nthreads),
            amrex::DataDeleter{amrex::The_Cpu_Arena()});
        auto* p = reinterpret_cast<InjectorMomentum*>(inj_mom_data.get());
        for (int tid = 0; tid < nthreads; ++tid) {
            auto* q = p++;
            new (q) InjectorMomentum(
                static_cast<InjectorMomentumMaxwellian*>(nullptr), object.maxwellian);
            inj_mom_omp.push_back(q);
        }
    }
#endif
}

void InjectorMomentum::prepare (
    amrex::RealBox const& pbox,
    int moving_dir,
    int moving_sign,
    std::function<amrex::Real(amrex::Real)> const& get_zlab)
{
    if (!needPreparation()) {
        return;
    }

    object.maxwellian.prepare(pbox, moving_dir, moving_sign, get_zlab);

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    clearThreadCopies();
#endif
}

void InjectorMomentum::prepare (int li, InjectorMomentum** inj_mom)
{
    if (!needPreparation()) {
        return;
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
    if (inj_mom_data) {
        auto* my_inj_mom = inj_mom_omp[amrex::OpenMP::get_thread_num()];
        my_inj_mom->object.maxwellian.prepare(li);
        *inj_mom = my_inj_mom;
    } else
#endif
    {
        object.maxwellian.prepare(li);
#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy_async(*inj_mom, this, sizeof(InjectorMomentum));
#else
        *inj_mom = this;
#endif
    }
}

bool InjectorMomentum::distributed () const noexcept
{
    if (type != Type::maxwellian) {
        return false;
    }
    return object.maxwellian.distributed();
}

bool InjectorMomentum::bulkMomentumFromFileIsDistributed () const noexcept
{
    return type == Type::maxwellian &&
           object.maxwellian.bulkMomentumFromFileIsDistributed();
}
