#include <AvgGalileanAlgorithm.H>
#include <WarpXConst.H>
#include <cmath>
#include <complex>
#include <iomanip>

using namespace amrex;


/* \brief Initialize coefficients for the update equation */
AvgGalileanAlgorithm::AvgGalileanAlgorithm(const SpectralKSpace& spectral_kspace,
                         const DistributionMapping& dm,
                         const int norder_x, const int norder_y,
                         const int norder_z, const bool nodal,
                         const amrex::Array<amrex::Real,3>& v_galilean,
                         const Real dt)
     // Initialize members of base classinde
     : SpectralBaseAlgorithm( spectral_kspace, dm,
                              norder_x, norder_y, norder_z, nodal )
{
    const BoxArray& ba = spectral_kspace.spectralspace_ba;

    // Allocate the arrays of coefficients
    C_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S_ck_coef = SpectralRealCoefficients(ba, dm, 1, 0);

    C1_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S1_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    C3_coef = SpectralRealCoefficients(ba, dm, 1, 0);
    S3_coef = SpectralRealCoefficients(ba, dm, 1, 0);

    Psi1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Psi2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Psi3_coef = SpectralComplexCoefficients(ba, dm, 1, 0);


    X1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    X2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    X3_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    X4_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Theta2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);

    A1_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    A2_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Rhocoef_coef = SpectralComplexCoefficients(ba, dm, 1, 0);

    Rhoold_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Rhonew_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    Jcoef_coef = SpectralComplexCoefficients(ba, dm, 1, 0);

    EJmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    BJmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    ERhooldmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    ERhomult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);

    EpJmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    BpJmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    EpRhooldmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    EpRhomult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    EBpmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);
    BEpmult_coef = SpectralComplexCoefficients(ba, dm, 1, 0);



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

        //Print()<<" kx[i]"<< ' ' <<' '<<modified_kx<<' '<<'\n';




        // Extract arrays for the coefficients
        Array4<Real> C = C_coef[mfi].array();

        Array4<Real> S = S_coef[mfi].array();
        Array4<Real> S_ck = S_ck_coef[mfi].array();
        Array4<Real> C1 = C1_coef[mfi].array();
        Array4<Real> S1 = S1_coef[mfi].array();
        Array4<Real> C3 = C3_coef[mfi].array();
        Array4<Real> S3 = S3_coef[mfi].array();

        Array4<Complex> Psi1 = Psi1_coef[mfi].array();
        Array4<Complex> Psi2 = Psi2_coef[mfi].array();
        Array4<Complex> Psi3 = Psi3_coef[mfi].array();
        Array4<Complex> X1 = X1_coef[mfi].array();
        Array4<Complex> X2 = X2_coef[mfi].array();
        Array4<Complex> X3 = X3_coef[mfi].array();
        Array4<Complex> X4 = X4_coef[mfi].array();
        Array4<Complex> Theta2 = Theta2_coef[mfi].array();
        Array4<Complex> A1 = A1_coef[mfi].array();
        Array4<Complex> A2 = A2_coef[mfi].array();

        Array4<Complex> Rhocoef = Rhocoef_coef[mfi].array();

        Array4<Complex> CRhoold = Rhoold_coef[mfi].array();
        Array4<Complex> CRhonew = Rhonew_coef[mfi].array();
        Array4<Complex> Jcoef   = Jcoef_coef[mfi].array();

        Array4<Complex> EJmult = EJmult_coef[mfi].array();
        Array4<Complex> BJmult = BJmult_coef[mfi].array();
        Array4<Complex> ERhooldmult =ERhooldmult_coef[mfi].array();
        Array4<Complex> ERhomult = ERhomult_coef[mfi].array();


        Array4<Complex> EBpmult = EBpmult_coef[mfi].array();
        Array4<Complex> BEpmult = BEpmult_coef[mfi].array();
        Array4<Complex> EpJmult = EpJmult_coef[mfi].array();
        Array4<Complex> BpJmult = BpJmult_coef[mfi].array();
        Array4<Complex> EpRhooldmult =EpRhooldmult_coef[mfi].array();
        Array4<Complex> EpRhomult = EpRhomult_coef[mfi].array();


        // Extract reals (for portability on GPU)
        Real vx = v_galilean[0];
        Real vy = v_galilean[1];
        Real vz = v_galilean[2];
        Real vgal_norm = std::sqrt(vx*vx + vy*vy + vz*vz);



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
            //Print()<<" modified_k"<< ' ' << i <<' '<< j <<' '<<modified_kx[i]<< ' '<<'\n';

            // Calculate coefficients
            constexpr Real c = PhysConst::c;
            constexpr Real c2 = PhysConst::c*PhysConst::c;
            constexpr Real ep0 = PhysConst::ep0;
            const Complex I{0.,1.0};
            if (k_norm != 0){

                // Calculate dot product with galilean velocity
                const Real kv = modified_kx[i]*vx +
#if (AMREX_SPACEDIM==3)
                                 modified_ky[j]*vy +
                                 modified_kz[k]*vz;
#else
                                 modified_kz[j]*vz;
#endif
                const Real cdt = dt*c;
                const Real wdt = k_norm*cdt;
                const Real w0 = k_norm * c;

                C(i,j,k) = std::cos(wdt);
                S(i,j,k) = std::sin(wdt);

                S_ck(i,j,k) = std::sin(wdt)/(c*k_norm);

                C1(i,j,k) = std::cos(wdt/2.);
                S1(i,j,k) = std::sin(wdt/2.);
                C3(i,j,k) = std::cos(3.*wdt/2.);
                S3(i,j,k) = std::sin(3.*wdt/2.);




                const Real nu = kv/(k_norm*c);
                const Complex theta = std::exp( 0.5*I*kv*dt );
                const Complex Theta = std::exp( I*kv*dt );
                const Complex theta_star = std::exp( -I*kv*dt/2. );
                const Complex e_theta = std::exp( I*wdt );
                Theta2(i,j,k) = theta*theta;




                // std::complex<long double> Psi0[17][31][0], iu(0.l,1.l);
                // std::complex<double> Psi0_double[17][31][0];
                // long double nu_ld = (long double)nu, kv_ld =(long double)kv, dt_ld =(long double)dt  ;
                // long double c_ld = (long double)c, k_norm_ld = (long double)k_norm;
                // long double wdt_ld = c_ld*dt_ld*k_norm_ld;
                // Psi0[i][j][k] = std::exp( 0.5l*iu*kv_ld*dt_ld ) * ( (std::cos(wdt_ld/2.l) - iu*nu_ld*std::sin(wdt_ld/2.l)) - std::exp( iu*kv_ld*dt_ld )* (std::cos(1.5l*wdt_ld)- iu*nu_ld*std::sin(1.5l*wdt_ld) )) / (wdt_ld * (nu_ld*nu_ld - 1.l)) ;
                //
                //
                // Psi0_double[i][j][k] = Psi0[i][j][k];
                //Print()<<"Psi0"<< ' ' << i <<' '<< j <<' '<<Psi0[i][j][k]<<'\n';
                //Print()<<"Psi0_double"<< ' ' << i <<' '<< j <<' '<<Psi0_double[i][j][k]<<'\n';

                //Print()<<"S1"<< ' ' << std::setprecision(32)<<i <<' '<< j <<' '<<std::sin(wdt_ld/2.l)<<'\n';



                //Psi2(i,j,k) = theta*( (C1(i,j,k) - I*nu*S1(i,j,k))-Theta2(i,j,k)*(C3(i,j,k)-I*nu*S3(i,j,k)) ) / (wdt*(nu*nu-1.));



                if ( (nu != 1.) && (nu != 0.) ) {
                    const Real denom = w0*w0-kv*kv;
                    const Complex So1mT = S(i,j,k) / (1.-Theta2(i,j,k));
                    const Complex onemCo1T = (1.-C(i,j,k) )/ (1.-Theta2(i,j,k));

                    X1(i,j,k) = (1. - C(i,j,k)*Theta2(i,j,k) + I*nu*S(i,j,k)*Theta2(i,j,k)) / denom;

                    X2(i,j,k) = (1. + I*nu*Theta2(i,j,k)*So1mT + nu*nu*onemCo1T*Theta2(i,j,k)) / denom;

                    X3(i,j,k) = Theta2(i,j,k)* ( C(i,j,k) + I*nu*Theta2(i,j,k)*So1mT + nu*nu*onemCo1T )/ denom;


                    Psi1(i,j,k) = theta*( (S1(i,j,k) + I*nu*C1(i,j,k))-Theta2(i,j,k)*(S3(i,j,k)+I*nu*C3(i,j,k)) ) / (wdt*(nu*nu-1.));
                    Psi2(i,j,k) = theta*( (C1(i,j,k) - I*nu*S1(i,j,k))-Theta2(i,j,k)*(C3(i,j,k)-I*nu*S3(i,j,k)) ) / (wdt*(nu*nu-1.));
                    Psi3(i,j,k) = I * theta * (1. - Theta2(i,j,k)) / (wdt*nu);


                    BEpmult(i,j,k) = (I/c)*Psi2(i,j,k);
                    EBpmult(i,j,k) = -I*c*Psi2(i,j,k);


                    ERhomult(i,j,k) =  -I*c2*X2(i,j,k)*k_norm/ep0;
                    ERhooldmult(i,j,k) = I*c2*X3(i,j,k)*k_norm/ep0;
                    //ERhomult(i,j,k) =  -I*c2*X2(i,j,k)/ep0;
                    //ERhooldmult(i,j,k) = I*c2*X3(i,j,k)/ep0;
                    BJmult(i,j,k) = -I*k_norm*X1(i,j,k)/ep0;
                    //BJmult(i,j,k) = -I*X1(i,j,k)/ep0;
                    EJmult(i,j,k) = - (Theta2(i,j,k)*S(i,j,k))/(c*k_norm*ep0) + I*kv*X1(i,j,k)/ep0;

                    A1(i,j,k) = (Psi1(i,j,k) -1. + I*nu*Psi2(i,j,k)) / ((c*k_norm)*(c*k_norm)*(nu*nu-1.));
                    A2(i,j,k) = (Psi3(i,j,k) - Psi1(i,j,k)) / (k_norm*c*k_norm*c);


                    EpJmult(i,j,k) = Psi2(i,j,k)/(ep0*c*k_norm) + A1(i,j,k)*I*kv/ep0;

                    BpJmult(i,j,k) = -k_norm*I*A1(i,j,k)/ep0;


                    Rhocoef(i,j,k) = I*c2*k_norm / (ep0*(1.-Theta2(i,j,k)));
                    EpRhomult(i,j,k) = Rhocoef(i,j,k) * A2(i,j,k) -  Rhocoef(i,j,k)* A1(i,j,k);
                    EpRhooldmult(i,j,k) = Rhocoef(i,j,k) * (A1(i,j,k)*Theta2(i,j,k) - A2(i,j,k));

                }
                if ( nu == 0) {
                    //Print()<<"( nu == 0)"<<'\n';

                    X1(i,j,k) = (1. - C(i,j,k)) / (c2*k_norm*k_norm);
                    X2(i,j,k) = (1. - S(i,j,k)/(c*k_norm*dt)) / (c2*k_norm*k_norm);
                    X3(i,j,k) = Theta2(i,j,k)*(C(i,j,k) - S(i,j,k)/(c*k_norm*dt)) / (c2*k_norm*k_norm);

                    Psi1(i,j,k) = -(S1(i,j,k) - S3(i,j,k))/wdt;
                    Psi2(i,j,k) = -(C1(i,j,k) - C3(i,j,k))/wdt;;
                    Psi3(i,j,k) = 1.;

                    ERhomult(i,j,k) =  -I*c2*k_norm*X2(i,j,k)/ep0;
                    ERhooldmult(i,j,k) = I*c2*k_norm*X3(i,j,k)/ep0;
                    //ERhomult(i,j,k) =  -I*c2*X2(i,j,k)/ep0;
                    //ERhooldmult(i,j,k) = I*c2*X3(i,j,k)/ep0;
                    BJmult(i,j,k) = -I*k_norm*X1(i,j,k)/ep0;
                    //BJmult(i,j,k) =  -I*X1(i,j,k)/ep0;
                    EJmult(i,j,k) = - S(i,j,k)/(c*k_norm*ep0);
                    Theta2(i,j,k) = 1.;
                    BEpmult(i,j,k) = (I/c) * Psi2(i,j,k);




                    A1(i,j,k) = -(Psi1(i,j,k) -1.) / (c2*k_norm*k_norm);
                    A2(i,j,k) = (c*k_norm*dt + S1(i,j,k) - S3(i,j,k)) / (c2*k_norm*k_norm*c*k_norm*dt);
                    // Coefficients of the averaged algo
                    EpJmult(i,j,k) = Psi2(i,j,k)/(ep0*c*k_norm);;
                    BpJmult(i,j,k) = -k_norm*I*A1(i,j,k)/ep0;
                    Rhocoef(i,j,k) = -c2 / (ep0*dt*vgal_norm);
                    EpRhomult(i,j,k) = -I*c*( c2*k_norm*k_norm*dt*dt -C1(i,j,k) + C3(i,j,k)) / (ep0*c2*c*k_norm*k_norm*k_norm*dt*dt);
                    EpRhooldmult(i,j,k) = 2.*I*c*S1(i,j,k) * (c*k_norm*dt*C(i,j,k) - S(i,j,k)) / (ep0*c2*c*k_norm*k_norm*k_norm*dt*dt);
                    EBpmult(i,j,k) =-I*c*Psi2(i,j,k);

                     }
                if ( nu == 1. or nu==-1.) {
                    //Print()<<"( nu == 1. or nu==-1.) "<<'\n';
                    //Psi2(i,j,k) = 0.;
                    // X1(i,j,k) = (1. - e_theta*e_theta + 2.*I*c*k_norm*dt) / (4.*c*c*ep0*k_norm*k_norm);
                    // X2(i,j,k) = (3. - 4.*e_theta + e_theta*e_theta + 2.*I*c*k_norm*dt) / (4.*ep0*k_norm*k_norm*(1.- e_theta));
                    // X3(i,j,k) = (3. - 2./e_theta - 2.*e_theta + e_theta*e_theta - 2.*I*c*k_norm*dt) / (4.*ep0*(e_theta - 1.)*k_norm*k_norm);
                    // X4(i,j,k) = I*(-1. + e_theta*e_theta + 2.*I*c*k_norm*dt) / (4.*ep0*c*k_norm);
                }

            } else { // Handle k_norm = 0, by using the analytical limit
              //Print()<<"( k==0.) "<<'\n';
              C(i,j,k) = 1.;
              S_ck(i,j,k) = dt;
              C1(i,j,k) = 1.;
              S1(i,j,k) =  0.;
              C3(i,j,k) = 1.;
              S3(i,j,k) = 0.;

              X1(i,j,k) = dt*dt/2.;
              X2(i,j,k) = dt*dt/6.;
              X3(i,j,k) = -dt*dt/3.;
              Theta2(i,j,k) = 1.;

              Psi1(i,j,k) = 1.;
              Psi2(i,j,k) = 0.0;
              Psi3(i,j,k) = 1.;

              EJmult(i,j,k) = -dt/ep0 +  I*(vgal_norm/c)*X1(i,j,k)/ep0;
              //BJmult(i,j,k) = -I*k_norm*X1(i,j,k)/ep0;;
              BJmult(i,j,k) =-I*k_norm*X1(i,j,k)/ep0;

              // ERhomult(i,j,k) =  -I*c2*X2(i,j,k)/ep0;
              // ERhooldmult(i,j,k) = I*c2*X3(i,j,k)/ep0;
              ERhomult(i,j,k) =  -I*c2*k_norm*X2(i,j,k)/ep0;
              ERhooldmult(i,j,k) = I*c2*k_norm*X3(i,j,k)/ep0;

              A1(i,j,k) = 13.*dt*dt/24.;
              A2(i,j,k) = 13.*dt*dt/24.;
              EpJmult(i,j,k) = -dt/ep0;//+ A1(i,j,k)*I*kv/ep0;????
              BpJmult(i,j,k) = -k_norm*I*A1(i,j,k)/ep0;
              Rhocoef(i,j,k) = -c2 / (ep0*vgal_norm*dt);
              EpRhomult(i,j,k) = Rhocoef(i,j,k) * (A2(i,j,k) - A1(i,j,k));
              EpRhooldmult(i,j,k) = Rhocoef(i,j,k) * (A1(i,j,k)*Theta2(i,j,k) - A2(i,j,k));
              BEpmult(i,j,k) =(I/c)*Psi2(i,j,k);
              EBpmult(i,j,k) =-I*c*Psi2(i,j,k);



            }

        });
    }
};

/* Advance the E and B field in spectral space (stored in `f`)
 * over one time step */
void
AvgGalileanAlgorithm::pushSpectralFields(SpectralFieldData& f) const{

    // Loop over boxes
    for (MFIter mfi(f.fields); mfi.isValid(); ++mfi){

        const Box& bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        Array4<Complex> fields = f.fields[mfi].array();
        // Extract arrays for the coefficients
        Array4<const Real> C_arr = C_coef[mfi].array();
        Array4<const Real> S_arr = S_coef[mfi].array();
        Array4<const Real> S_ck_arr = S_ck_coef[mfi].array();
        Array4<const Complex> X1_arr = X1_coef[mfi].array();
        Array4<const Complex> X2_arr = X2_coef[mfi].array();
        Array4<const Complex> X3_arr = X3_coef[mfi].array();
        Array4<const Complex> X4_arr = X4_coef[mfi].array();
        Array4<const Complex> Theta2_arr = Theta2_coef[mfi].array();
        Array4<const Complex> Psi1_arr = Psi1_coef[mfi].array();
        Array4<const Complex> Psi2_arr = Psi2_coef[mfi].array();
        Array4<const Complex> Psi3_arr = Psi3_coef[mfi].array();
        Array4<const Real> C1_arr = C1_coef[mfi].array();
        Array4<const Real> S1_arr = S1_coef[mfi].array();
        Array4<const Real> C3_arr = C3_coef[mfi].array();
        Array4<const Real> S3_arr = S3_coef[mfi].array();



        Array4<const Complex> Rhocoef_arr = Rhocoef_coef[mfi].array();

        Array4<const Complex> A1_arr = A1_coef[mfi].array();
        Array4<const Complex> A2_arr = A2_coef[mfi].array();
        Array4<const Complex> Rhonew_arr = Rhonew_coef[mfi].array();
        Array4<const Complex> Rhoold_arr = Rhoold_coef[mfi].array();
        Array4<const Complex> Jcoef_arr =Jcoef_coef[mfi].array();

        Array4<const Complex> EBpmult_arr = EBpmult_coef[mfi].array();
        Array4<const Complex> BEpmult_arr = BEpmult_coef[mfi].array();
        Array4<const Complex> EJmult_arr = EJmult_coef[mfi].array();
        Array4<const Complex> BJmult_arr = BJmult_coef[mfi].array();
        Array4<const Complex> ERhooldmult_arr = ERhooldmult_coef[mfi].array();
        Array4<const Complex> ERhomult_arr = ERhomult_coef[mfi].array();

        Array4<const Complex> EpJmult_arr = EpJmult_coef[mfi].array();
        Array4<const Complex> BpJmult_arr = BpJmult_coef[mfi].array();
        Array4<const Complex> EpRhooldmult_arr = EpRhooldmult_coef[mfi].array();
        Array4<const Complex> EpRhomult_arr = EpRhomult_coef[mfi].array();


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
            using Idx = SpectralAvgFieldIndex;

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

            const Complex Ex_avg = fields(i,j,k,Idx::Ex_avg);
            const Complex Ey_avg= fields(i,j,k,Idx::Ey_avg);
            const Complex Ez_avg = fields(i,j,k,Idx::Ez_avg);
            const Complex Bx_avg = fields(i,j,k,Idx::Bx_avg);
            const Complex By_avg = fields(i,j,k,Idx::By_avg);
            const Complex Bz_avg = fields(i,j,k,Idx::Bz_avg);
            // k vector values, and coefficients
            const Real kx = modified_kx_arr[i];

#if (AMREX_SPACEDIM==3)
            const Real ky = modified_ky_arr[j];
            const Real kz = modified_kz_arr[k];
#else
            constexpr Real ky = 0;
            const Real kz = modified_kz_arr[j];
#endif

// Calculate norm of vector
//             const Real k_norm_mag = std::sqrt(
//             std::pow(kx, 2) +
// #if (AMREX_SPACEDIM==3)
//             std::pow(ky, 2) +
//             std::pow(kz, 2));
// #else
//             std::pow(kz, 2));
// #endif
//               const Real k_norm_mag = (kx == 0. && ky==0. && kz==0. ) || (kx == 0. && kz==0. ) ? 1 :  std::sqrt(
//               std::pow(kx, 2) +
// #if (AMREX_SPACEDIM==3)
//               std::pow(ky, 2) +
//               std::pow(kz, 2));
// #else
//               std::pow(kz, 2));
// #endif



            const Real k_norm_mag =
#if (AMREX_SPACEDIM==3)
            (kx == 0. && ky==0. && kz==0. ) ? 1. : std::sqrt(std::pow(kx, 2) + std::pow(ky, 2) + std::pow(kz, 2));
#else
            (kx == 0. && kz==0. ) ? 1. : std::sqrt(std::pow(kx, 2) + std::pow(kz, 2));
#endif

            constexpr Real c = PhysConst::c;
            constexpr Real c2 = PhysConst::c*PhysConst::c;
            constexpr Real inv_ep0 = 1./PhysConst::ep0;
            constexpr Complex I = Complex{0.,1.};

            const Real C = C_arr(i,j,k);
            const Real S = S_arr(i,j,k);
            const Real S_ck = S_ck_arr(i,j,k);

            const Real C1 = C1_arr(i,j,k);
            const Real C3 = C3_arr(i,j,k);
            const Real S1 = S1_arr(i,j,k);
            const Real S3 = S3_arr(i,j,k);

            const Complex X1 = X1_arr(i,j,k);
            const Complex X2 = X2_arr(i,j,k);
            const Complex X3 = X3_arr(i,j,k);
            const Complex X4 = X4_arr(i,j,k);
            const Complex T2 = Theta2_arr(i,j,k);

            const Complex Psi1 = Psi1_arr(i,j,k);
            const Complex Psi2 = Psi2_arr(i,j,k);
            const Complex Psi3 = Psi3_arr(i,j,k);
            const Complex A1 = A1_arr(i,j,k);
            const Complex A2 = A2_arr(i,j,k);
            const Complex CRhoold= Rhoold_arr(i,j,k);
            const Complex CRhonew= Rhonew_arr(i,j,k);
            const Complex Jcoef = Jcoef_arr(i,j,k);

            const Complex EJmult = EJmult_arr(i,j,k);
            const Complex BJmult = BJmult_arr(i,j,k);
            const Complex ERhooldmult = ERhooldmult_arr(i,j,k);
            const Complex ERhomult = ERhomult_arr(i,j,k);


            const Complex EpJmult = EpJmult_arr(i,j,k);
            const Complex BpJmult = BpJmult_arr(i,j,k);
            const Complex EpRhooldmult = EpRhooldmult_arr(i,j,k);
            const Complex EpRhomult = EpRhomult_arr(i,j,k);
            const Complex Rhocoef = Rhocoef_arr(i,j,k);
            const Complex BEpmult = BEpmult_arr(i,j,k);
            const Complex EBpmult = EBpmult_arr(i,j,k);

            fields(i,j,k,Idx::Bx) = T2*C*Bx_old
                        - T2*S_ck*I*(ky*Ez_old - kz*Ey_old)
                        -      BJmult*(ky*Jz     - kz*Jy)/k_norm_mag;
            fields(i,j,k,Idx::By) = T2*C*By_old
                        - T2*S_ck*I*(kz*Ex_old - kx*Ez_old)
                        -      BJmult*(kz*Jx     - kx*Jz)/k_norm_mag;
            fields(i,j,k,Idx::Bz) = T2*C*Bz_old
                        - T2*S_ck*I*(kx*Ey_old - ky*Ex_old)
                        -      BJmult*(kx*Jy     - ky*Jx)/k_norm_mag;



            fields(i,j,k,Idx::Ex) = T2*C*Ex_old
                        + T2*S_ck*c2*I*(ky*Bz_old - kz*By_old)
                        + EJmult*Jx + (ERhomult*rho_new + ERhooldmult*rho_old)*kx/k_norm_mag;
            fields(i,j,k,Idx::Ey) = T2*C*Ey_old
                        + T2*S_ck*c2*I*(kz*Bx_old - kx*Bz_old)
                        + EJmult*Jy + (ERhomult*rho_new + ERhooldmult*rho_old)*ky/k_norm_mag;
            fields(i,j,k,Idx::Ez) = T2*C*Ez_old
                        + T2*S_ck*c2*I*(kx*By_old - ky*Bx_old)
                        + EJmult*Jz + (ERhomult*rho_new + ERhooldmult*rho_old)*kz/k_norm_mag;



            // fields(i,j,k,Idx::Bx_avg) =fields(i,j,k,Idx::Bx);
            // fields(i,j,k,Idx::By_avg) =fields(i,j,k,Idx::By);
            // fields(i,j,k,Idx::Bz_avg) = fields(i,j,k,Idx::Bz);
            //
            // fields(i,j,k,Idx::Ex_avg) = fields(i,j,k,Idx::Ex);
            // fields(i,j,k,Idx::Ey_avg) = fields(i,j,k,Idx::Ey);
            // fields(i,j,k,Idx::Ez_avg) = fields(i,j,k,Idx::Ez);

            fields(i,j,k,Idx::Bx_avg) = Psi1*Bx_old
                      + BEpmult*(ky*Ez_old - kz*Ey_old)/k_norm_mag
                      - BpJmult*(ky*Jz     - kz*Jy)/k_norm_mag;

            fields(i,j,k,Idx::By_avg) = Psi1*By_old
                    + BEpmult*(kz*Ex_old - kx*Ez_old)/k_norm_mag
                    - BpJmult*(kz*Jx     - kx*Jz)/k_norm_mag;

            fields(i,j,k,Idx::Bz_avg) = Psi1*Bz_old
                    + BEpmult*(kx*Ey_old - ky*Ex_old)/k_norm_mag
                      - BpJmult*(kx*Jy     - ky*Jx)/k_norm_mag;


            fields(i,j,k,Idx::Ex_avg) = Psi1*Ex_old
                     - Psi2*c*I*(ky*Bz_old - kz*By_old)/k_norm_mag
                    + EpJmult*Jx + ( EpRhomult * rho_new +  EpRhooldmult*rho_old )*kx/k_norm_mag;

            fields(i,j,k,Idx::Ey_avg) = Psi1*Ey_old
                     - Psi2*c*I*(kz*Bx_old - kx*Bz_old)/k_norm_mag
                    + EpJmult*Jy +( EpRhomult * rho_new +  EpRhooldmult*rho_old )*ky/k_norm_mag;

            fields(i,j,k,Idx::Ez_avg) = Psi1*Ez_old
                   - Psi2*c*I*(kx*By_old - ky*Bx_old)/k_norm_mag
                   + EpJmult*Jz   + ( EpRhomult * rho_new +  EpRhooldmult*rho_old )*kz/k_norm_mag;



            fields(i,j,k,Idx::Ex_avg) = fields(i,j,k,Idx::Ex);
            fields(i,j,k,Idx::Ey_avg) = fields(i,j,k,Idx::Ey);
            fields(i,j,k,Idx::Ez_avg) = fields(i,j,k,Idx::Ez);

            fields(i,j,k,Idx::Bx_avg) = fields(i,j,k,Idx::Bx);
            fields(i,j,k,Idx::By_avg) = fields(i,j,k,Idx::By);
            fields(i,j,k,Idx::Bz_avg) = fields(i,j,k,Idx::Bz);



            //std::complex<long double> Psi0[17][31][0], iu(0.l,1.l);

            //Psi0[i][j][k] = C1;

            //theta*( (C1(i,j,k) - I*nu*S1(i,j,k))-Theta2(i,j,k)*(C3(i,j,k)-I*nu*S3(i,j,k)) ) / (wdt*(nu*nu-1.));


            //Print()<<"Ez:"<< ' ' << i << ' ' << j<< ' ' << fields(i,j,k,Idx::Ez_avg) <<' '<< fields(i,j,k,Idx::Ez) <<' '<<'\n';
            //Print()<<"EpJmult" << ' ' << i <<' '<< j <<' '<<EpJmult<<'\n';
            //
            //Print()<<"EBpmult"<< ' ' << i <<' '<< j <<' '<<EBpmult<<'\n';
            //Print()<<"BpJmult"<< ' ' << i <<' '<< j <<' '<<BpJmult<<'\n';

            //Print()<<"Psi1"<< ' ' << i <<' '<< j <<' '<<Psi1<<'\n';
            //Print()<<"Psi2"<< ' ' << i <<' '<< j <<' '<<Psi2<<'\n';
            //Print()<<" Psi3"<< ' ' << i <<' '<< j <<' '<< Psi3<<'\n';

            //Print()<<"X1"<< ' ' << i <<' '<< j <<' '<<X1<<'\n';
            //Print()<<"X2"<< ' ' << i <<' '<< j <<' '<<X2<<'\n';
            //Print()<<" X3"<< ' ' << i <<' '<< j <<' '<< X3<<'\n';

            //Print()<<"BEpmult"<< ' ' << i <<' '<< j <<' '<<BEpmult<<'\n';
            //Print()<<"EJmult"<< ' ' << i <<' '<< j <<' '<<EJmult<<'\n';

            //std::cout<<std::setprecision(24)<<' ' << i <<' '<< j <<' '<<Psi1-Psi3<<'\n';
            //Print()<<"Psi1"<< ' ' << i <<' '<< j <<' '<<Psi1<<'\n';
            //std::cout<<std::setprecision(24)<<"Psi1=" << i <<' '<< j <<' '<<Psi1<<'\n';
            //std::cout<<std::setprecision(24)<<"Psi3=" << i <<' '<< j <<' '<<Psi3<<'\n';

            //Print()<<"Psi2   " << ' ' << Psi2 << ' '<< i <<' '<< j <<' '<< kx << ' '<< kz << ' '<<'\n';
            //Print()<<"kz"<< ' ' << i <<' '<< j <<' '<<kz<<'\n';
            //Print()<<"  Psi3:"<<  ' ' << Psi3 <<' '<< i << ' '<<j <<' '<< kx<< ' ' << kz<<'\n';

            //Print()<<"  BEpmult:"<<  ' ' << BEpmult <<' '<< i << ' '<<j <<' '<< kx<< ' ' << kz<<'\n';
            // Print()<<"  A1:"<<  ' ' << A1 <<' '<< i << ' '<<j <<' '<< kx<< ' ' << kz<<'\n';

           //Print()<<"  EpRhooldmult:"<<  ' ' << EpRhooldmult<<' '<< i << ' '<<j <<' '<< kx<< ' ' << kz<<'\n';
           //Print()<<fields(i,j,k,Idx::Ez)<<  ' ' <<fields(i,j,k,Idx::Ez_avg) <<' '<< i << ' '<<j <<' '<< kx<< ' ' << kz<<'\n';
           //Print()<<"  EpRhomult:"<<  ' ' << EpRhomult<<' '<< i << ' '<<j <<' '<<'\n';



            });
    }
};
