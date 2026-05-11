#ifndef __RK4_H
#define __RK4_H

#include "../Utils/Utils.h"
#include "mkl_lapacke.h"

class RK4
{
  public:
    // Methods
    RK4(MKL_INT dim);
    ~RK4();
    // Members
    void rk4_redfield(double delta_t, 
                      MZType &Rho0, 
                      MZType &Red);

    // dw(t)_dt = Qw(t) + Rv 
    void rk4_redfield_twoops(double delta_t,
                             MZType &w0,
                             MZType &Q,
                             MZType &v0,
			                       MZType &vh2,
			                       MZType &vh,
                             MZType &R);
  private:
    void multiply_(MZType &Res, 
                   MZType &Red, 
                   MZType &Rho);
    MKL_INT dim_;
    MZType k1_;
    MZType k2_;
    MZType k3_;
    MZType k4_;
    MZType Rho0c1_;
    MZType Rho0c2_;
    MZType Rho0c3_;
    MZType Rv0_; //buffer to store Rv(t)
    MZType Rvh2_; //buffer to store Rv(t + h/2)
    MZType Rvh_; //buffer to store Rv(t + h)
};
#endif
