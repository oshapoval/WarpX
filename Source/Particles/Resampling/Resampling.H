/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RESAMPLING_H_
#define WARPX_RESAMPLING_H_

#include "ResamplingTrigger.H"

class WarpXParIter; // forward declaration
class WarpXParticleContainer; // forward declaration

/**
 * \brief An empty base class from which specific resampling algorithms are derived.
 */
struct ResamplingAlgorithm
{
    /**
     * \brief Virtual operator() of the abstract ResamplingAlgorithm class
     */
    virtual void operator() (WarpXParIter& /*pti*/, const int /*lev*/, WarpXParticleContainer */*pc*/) const = 0;

    /**
     * \brief Virtual destructor of the abstract ResamplingAlgorithm class
     */
    virtual ~ResamplingAlgorithm () =  default;
};

/**
 * \brief This is a general class used for resampling that is instantiated as a member of
 * MultiParticleContainer. It contains a ResamplingTrigger object used to determine if resampling
 * should be done at a given timestep for a given species and a pointer to a ResamplingAlgorithm
 * object used to carry out the resampling.
 */
class Resampling
{
public:

    /**
     * \brief Default constructor of the Resampling class.
     */
    Resampling () = default;

    /**
     * \brief Constructor of the Resampling class. Reads the chosen resampling algorithm from the
     * input file.
     *
     * @param[in] species_name the name of the resampled species
     */
    Resampling (const std::string species_name);

    /**
     * \brief A method that returns true if resampling should be done for the considered species
     * at the considered timestep.
     *
     * @param[in] timestep the current timestep
     * @param[in] global_numparts the total number of particles of the considered species
     */
    bool triggered (const int timestep, const amrex::Real global_numparts) const;

    /**
     * \brief A method that uses the ResamplingAlgorithm object to perform resampling.
     *
     * @param[in] pti WarpX particle iterator of the particles to resample.
     * @param[in] lev the index of the refinement level.
     * @param[in] pc a pointer to the particle container.
     */
    void operator() (WarpXParIter& pti, const int lev, WarpXParticleContainer * const pc) const;

private:
    ResamplingTrigger m_resampling_trigger;
    std::unique_ptr<ResamplingAlgorithm> m_resampling_algorithm;
};

#endif //WARPX_RESAMPLING_H_
