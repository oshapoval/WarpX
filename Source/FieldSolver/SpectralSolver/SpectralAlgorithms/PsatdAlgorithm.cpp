#include <PsatdAlgorithm.H>
#include <WarpXConst.H>
#include <cmath>

using namespace amrex;

/* \brief Initialize coefficients for the update equation */
PsatdAlgorithm::PsatdAlgorithm(const SpectralKSpace& spectral_kspace,
                         const DistributionMapping& dm,
                         const int norder_x, const int norder_y,
                         const int norder_z, const bool nodal, const Real dt)
     // Initialize members of base class
     : SpectralBaseAlgorithm( spectral_kspace, dm,
                              norder_x, norder_y, norder_z, nodal )
{
    const BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Allocate the arrays of coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X1_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X2_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    X3_coef = SpectralRealCoefficients(ba, dm, 1, 0);


    Ch_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    Sh_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    Sh_ckdt_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    Ch_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);



    InitializeSpectralCoefficients(spectral_kspace, dm, dt);
}

/* Advance the E and B field in spectral space (stored in `f`)
 * over one time step */
void
PsatdAlgorithm::pushSpectralFields(SpectralFieldData& f) const{

    // Loop over boxes
    for (MFIter mfi(f.fields); mfi.isValid(); ++mfi){

        const Box& bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        Array4<Complex> fields = f.fields[mfi].array();
        // Extract arrays for the coefficients
        Array4<const Real> C_arr = C_coef[mfi].array();
        Array4<const Real> S_ck_arr = S_ck_coef[mfi].array();
        Array4<const Real> X1_arr = X1_coef[mfi].array();
        Array4<const Real> X2_arr = X2_coef[mfi].array();
        Array4<const Real> X3_arr = X3_coef[mfi].array();


        Array4<const Real> Ch_arr = Ch_coef[mfi].array();
        Array4<const Real> Sh_ck_arr = Sh_ck_coef[mfi].array();
        Array4<const Real> Sh_ckdt_arr = Sh_ckdt_coef[mfi].array();
        Array4<const Real> Ch_ck_arr = Ch_ck_coef[mfi].array();


        // Extract pointers for the k vectors
        const Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Loop over indices within one box
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Record old values of the fields to be updated
            using Idx = SpectralFieldIndex; //PSATDSpectralFieldIndex;
            const Complex Ex_old = fields(i,j,k,Idx::Ex);
            const Complex Ey_old = fields(i,j,k,Idx::Ey);
            const Complex Ez_old = fields(i,j,k,Idx::Ez);

            const Complex Bx_old = fields(i,j,k,Idx::Bx);
            const Complex By_old = fields(i,j,k,Idx::By);
            const Complex Bz_old = fields(i,j,k,Idx::Bz);
            // Shortcut for the values of J and rho
            const Complex Jx = fields(i,j,k,Idx::Jx);
            const Complex Jy = fields(i,j,k,Idx::Jy);
            const Complex Jz = fields(i,j,k,Idx::Jz);
            const Complex rho_old = fields(i,j,k,Idx::rho_old);
            const Complex rho_new = fields(i,j,k,Idx::rho_new);

            // const Complex Jx_old = fields(i,j,k,Idx::Jx_old);
            // const Complex Jy_old = fields(i,j,k,Idx::Jy_old);
            // const Complex Jz_old = fields(i,j,k,Idx::Jz_old);

            // k vector values, and coefficients
            const Real kx = modified_kx_arr[i];
#if (AMREX_SPACEDIM==3)
            const Real ky = modified_ky_arr[j];
            const Real kz = modified_kz_arr[k];
#else
            constexpr Real ky = 0;
            const Real kz = modified_kz_arr[j];
#endif


            const Real knorm = std::sqrt(
                std::pow(modified_kx_arr[i], 2) +
#if (AMREX_SPACEDIM==3)
                std::pow(modified_ky_arr[j], 2) +
                std::pow(modified_kz_arr[k], 2));
#else
                std::pow(modified_kz_arr[j], 2));
#endif

            const Real ck = PhysConst::c*knorm;
            constexpr Real c2 = PhysConst::c*PhysConst::c;
            constexpr Real inv_ep0 = 1./PhysConst::ep0;
            const Complex I = Complex{0,1};
            const Real C = C_arr(i,j,k);
            const Real S_ck = S_ck_arr(i,j,k);
            const Real X1 = X1_arr(i,j,k);
            const Real X2 = X2_arr(i,j,k);
            const Real X3 = X3_arr(i,j,k);

            const Real Ch = Ch_arr(i,j,k);
            const Real Sh_ck = Sh_ck_arr(i,j,k);
            const Real Sh_ckdt = Sh_ckdt_arr(i,j,k);
            const Real Ch_ck = Ch_ck_arr(i,j,k);



            //amrex::Print() <<"1"<<"------\n";
            // Update E (see WarpX online documentation: theory section)
            //PSATD via rho_old and new
            // fields(i,j,k,Idx::Ex) = C*Ex_old
            //             + S_ck*(c2*I*(ky*Bz_old - kz*By_old) - inv_ep0*Jx)
            //             - I*(X2*rho_new - X3*rho_old)*kx;
            // fields(i,j,k,Idx::Ey) = C*Ey_old
            //             + S_ck*(c2*I*(kz*Bx_old - kx*Bz_old) - inv_ep0*Jy)
            //             - I*(X2*rho_new - X3*rho_old)*ky;
            // fields(i,j,k,Idx::Ez) = C*Ez_old
            //             + S_ck*(c2*I*(kx*By_old - ky*Bx_old) - inv_ep0*Jz)
            //             - I*(X2*rho_new - X3*rho_old)*kz;
            // // Update B (see WarpX online documentation: theory section)
            // fields(i,j,k,Idx::Bx) = C*Bx_old
            //             - S_ck*I*(ky*Ez_old - kz*Ey_old)
            //             +   X1*I*(ky*Jz     - kz*Jy);
            // fields(i,j,k,Idx::By) = C*By_old
            //             - S_ck*I*(kz*Ex_old - kx*Ez_old)
            //             +   X1*I*(kz*Jx     - kx*Jz);
            // fields(i,j,k,Idx::Bz) = C*Bz_old
            //             - S_ck*I*(kx*Ey_old - ky*Ex_old)
            //             +   X1*I*(kx*Jy     - ky*Jx);
            //amrex::Print() <<"2"<<"------\n";




            // fields(i,j,k,Idx::Ex) = Ex_old
            //             + 2.*Sh_ck* ( ck*I*(ky*Bz_old - kz*By_old)  - Jx_old )
            //             + Sh_ckdt * kx * (kx*Jx_old + ky*Jy_old + kz*Jz_old);
            // fields(i,j,k,Idx::Ey) = Ey_old
            //             + 2.*Sh_ck*( ck*I*(kz*Bx_old - kx*Bz_old) - Jy_old )
            //             + Sh_ckdt * ky * (kx*Jx_old + ky*Jy_old + kz*Jz_old);
            // fields(i,j,k,Idx::Ez) = Ez_old
            //             + 2.*Sh_ck*( ck*I*(kx*By_old - ky*Bx_old) - Jz_old )
            //             + Sh_ckdt * kz * (kx*Jx_old + ky*Jy_old + kz*Jz_old);
            //
            //
            //
            // // Update B (see WarpX online documentation: theory section)
            // fields(i,j,k,Idx::Bx) = Bx_old
            //             - 2.*S_ck*I*(ky*fields(i,j,k,Idx::Ez) - kz*fields(i,j,k,Idx::Ey))
            //             + I * Ch_ck * (ky * (Jz - Jz_old) - kz*(Jy - Jy_old)) ;
            // fields(i,j,k,Idx::By) = By_old
            //             - 2.*S_ck*I*(kz*fields(i,j,k,Idx::Ex) - kx*fields(i,j,k,Idx::Ez))
            //             + I *  Ch_ck* (kz * (Jx - Jx_old) - kx*(Jz - Jz_old)) ;
            // fields(i,j,k,Idx::Bz) = Bz_old
            //             - 2.*S_ck*I*(kx*fields(i,j,k,Idx::Ey) - ky*fields(i,j,k,Idx::Ex))
            //             + I * Ch_ck * (kx * (Jy - Jy_old) - ky*(Jx - Jx_old)) ;
            fields(i,j,k,Idx::Ex) = C*Ex_old
                        + Sh_ck *I*(ky*Bz_old - kz*By_old)  - S_ck*Jx
                        + (1. - C) * kx * (kx*Ex_old + ky*Ey_old + kz*Ez_old)
                        + Sh_ckdt * kx * (kx*Jx + ky*Jy + kz*Jz);
            fields(i,j,k,Idx::Ey) = C*Ey_old
                        + Sh_ck*I*(kz*Bx_old - kx*Bz_old) - S_ck*Jy
                        + (1. - C) * ky * (kx*Ex_old + ky*Ey_old + kz*Ez_old)
                        + Sh_ckdt * ky * (kx*Jx + ky*Jy + kz*Jz);
            fields(i,j,k,Idx::Ez) = C*Ez_old
                        + Sh_ck *I*(kx*By_old - ky*Bx_old) - S_ck*Jz
                        + (1. - C) * kz * (kx*Ex_old + ky*Ey_old + kz*Ez_old)
                        + Sh_ckdt * kz * (kx*Jx + ky*Jy + kz*Jz);
            //
            // // Update B (see WarpX online documentation: theory section)
            fields(i,j,k,Idx::Bx) = C*Bx_old
                        - I*Sh_ck*(ky*Ez_old - kz*Ey_old)
                        + I * Ch_ck * (ky * Jz - kz*Jy) ;
            fields(i,j,k,Idx::By) =  C*By_old
                        - I*Sh_ck*(kz*Ex_old - kx*Ez_old)
                        + I *  Ch_ck* (kz * Jx - kx*Jz) ;
            fields(i,j,k,Idx::Bz) = C*Bz_old
                        - I*Sh_ck*(kx*Ey_old - ky*Ex_old)
                        + I * Ch_ck * (kx * Jy - ky*Jx) ;

            //amrex::Print() <<"Bz_old"<<Jx<< "||||" <<"Bx_old"<<Jz<<"------\n";
        });
    }
};

void PsatdAlgorithm::InitializeSpectralCoefficients(const SpectralKSpace& spectral_kspace,
                                    const amrex::DistributionMapping& dm,
                                    const amrex::Real dt)
{
    const BoxArray& ba = spectral_kspace.spectralspace_ba;
    // Fill them with the right values:
    // Loop over boxes and allocate the corresponding coefficients
    // for each box owned by the local MPI proc
    for (MFIter mfi(ba, dm); mfi.isValid(); ++mfi){

        const Box& bx = ba[mfi];

        // Extract pointers for the k vectors
        const Real* modified_kx = modified_kx_vec[mfi].dataPtr();
#if (AMREX_SPACEDIM==3)
        const Real* modified_ky = modified_ky_vec[mfi].dataPtr();
#endif
        const Real* modified_kz = modified_kz_vec[mfi].dataPtr();
        // Extract arrays for the coefficients
        Array4<Real> C = C_coef[mfi].array();
        Array4<Real> S_ck = S_ck_coef[mfi].array();
        Array4<Real> X1 = X1_coef[mfi].array();
        Array4<Real> X2 = X2_coef[mfi].array();
        Array4<Real> X3 = X3_coef[mfi].array();

        Array4<Real> Ch = C_coef[mfi].array(); //oshapoval
        Array4<Real> Sh_ck = Sh_ck_coef[mfi].array(); //oshapoval
        Array4<Real> Sh_ckdt = Sh_ck_coef[mfi].array(); //oshapoval
        Array4<Real> Ch_ck = Ch_ck_coef[mfi].array(); //oshapoval


        // Loop over indices within one box
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Calculate norm of vector
            const Real k_norm = std::sqrt(
                std::pow(modified_kx[i], 2) +
#if (AMREX_SPACEDIM==3)
                std::pow(modified_ky[j], 2) +
                std::pow(modified_kz[k], 2));
#else
                std::pow(modified_kz[j], 2));
#endif


            // Calculate coefficients
            constexpr Real c = PhysConst::c;
            constexpr Real ep0 = PhysConst::ep0;
            if (k_norm != 0){
                C(i,j,k) = std::cos(c*k_norm*dt);
                S_ck(i,j,k) = std::sin(c*k_norm*dt)/(c*k_norm);
                X1(i,j,k) = (1. - C(i,j,k))/(ep0 * c*c * k_norm*k_norm);
                X2(i,j,k) = (1. - S_ck(i,j,k)/dt)/(ep0 * k_norm*k_norm);
                X3(i,j,k) = (C(i,j,k) - S_ck(i,j,k)/dt)/(ep0 * k_norm*k_norm);
                //Ch(i,j,k) = std::cos(c*k_norm*dt/2.);//oshapoval
                Sh_ck(i,j,k) = std::sin(c*k_norm*dt);//std::sin(c*k_norm*dt)/(c*k_norm);//oshapoval
                Sh_ckdt(i,j,k) = std::sin(c*k_norm*dt)/(c*k_norm) - dt;//( 2.*std::sin(c*k_norm*dt/2.)/(c*k_norm) - dt );
                Ch_ck(i,j,k) = ( 1. - std::cos(c*k_norm*dt) ) / (c*k_norm);//(1. - std::cos(c*k_norm*dt/2.) ) / (c*k_norm);

            } else { // Handle k_norm = 0, by using the analytical limit
                C(i,j,k) = 1.;
                S_ck(i,j,k) = dt;
                X1(i,j,k) = 0.5 * dt * dt / ep0;
                X2(i,j,k) = c*c * dt * dt / (6.*ep0);
                X3(i,j,k) = - c*c * dt * dt / (3.*ep0);
                //Ch(i,j,k) = 1.;//oshapoval
                Sh_ck(i,j,k) = 0.0;//dt/2.;//oshapoval
                Sh_ckdt(i,j,k) = 0.;//0.;
                Ch_ck(i,j,k) = c*k_norm*dt*dt/2. ;//c*k_norm*dt*dt/8.;
            }
        });
     }
}
