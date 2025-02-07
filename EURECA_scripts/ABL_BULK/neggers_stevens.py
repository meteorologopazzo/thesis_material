import numpy as np

# Define some meteorological functions from pycoare.
def qsat(t,p):
    """
    usage: es = qsat(t,p)
    Returns saturation vapor pressure es (mb) given t(C) and p(mb).
    After Buck, 1981: J.Appl.Meteor., 20, 1527-1532
    Returns ndarray float for any numeric object input.
    """
    from numpy import copy, asarray, exp

    t2 = copy(asarray(t, dtype=float))  # convert to ndarray float
    p2 = copy(asarray(p, dtype=float))
    es = 6.1121 * exp(17.502 * t2 / (240.97 + t2))
    es = es * (1.0007 + p2 * 3.46e-6)
    return es

def qsea(sst,p):
    """
    usage: qs = qsea(sst,p)
    Returns saturation specific humidity (g/kg) at sea surface
    given sst(C) and p(mb) input of any numeric type.
    Returns ndarray float for any numeric object input.
    """
    ex = qsat(sst,p) # returns ex as ndarray float
    es = ex#0.98 * ex # This is the correction for the effect of salinity, which we remove now
    qs = 622*es/(p-0.378*es) # saturation specific humidity
    return qs




# Version 1.0 released by David Romps on September 12, 2017.
# Version 1.1 vectorized lcl.R, released on May 24, 2021.
# 
# When using this code, please cite:
# 
# @article{16lcl,
#   Title   = {Exact expression for the lifting condensation level},
#   Author  = {David M. Romps},
#   Journal = {Journal of the Atmospheric Sciences},
#   Year    = {2017},
#   Month   = dec,
#   Number  = {12},
#   Pages   = {3891--3900},
#   Volume  = {74}
# }
#
# This lcl function returns the height of the lifting condensation level
# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - return_ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - return_min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.

def lcl(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):

    import math
    import scipy.special

    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v   = 2.3740e6   # J/kg
    E0s   = 0.3337e6   # J/kg
    ggr   = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K 
    rgasv = 461        # J/kg/K 
    cva   = 719        # J/kg/K
    cvv   = 1418       # J/kg/K 
    cvl   = 4119       # J/kg/K 
    cvs   = 1861       # J/kg/K 
    cpa   = cva + rgasa
    cpv   = cvv + rgasv

    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * math.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

    # The saturation vapor pressure over solid ice
    def pvstars(T):
        return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * math.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

    # Calculate pv from rh, rhl, or rhs
    rh_counter = 0
    if rh  is not None:
        rh_counter = rh_counter + 1
    if rhl is not None:
        rh_counter = rh_counter + 1
    if rhs is not None:
        rh_counter = rh_counter + 1
    if rh_counter != 1:
        print(rh_counter)
        exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
    if rh is not None:
      # The variable rh is assumed to be 
      # with respect to liquid if T > Ttrip and 
      # with respect to solid if T < Ttrip
        if T > Ttrip:
            pv = rh * pvstarl(T)
        else:
            pv = rh * pvstars(T)
        rhl = pv / pvstarl(T)
        rhs = pv / pvstars(T)
    elif rhl is not None:
        pv = rhl * pvstarl(T)
        rhs = pv / pvstars(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    elif rhs is not None:
        pv = rhs * pvstars(T)
        rhl = pv / pvstarl(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    if pv > p:
        return NA

    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    if rh == 0:
        return cpm*T/ggr
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
    cL = pv/pvstarl(T)*math.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
    aS = -(cpv-cvs)/rgasv + cpm/rgasm
    bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
    cS = pv/pvstars(T)*math.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
    lcl = cpm*T/ggr*( 1 - \
      bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
    ldl = cpm*T/ggr*( 1 - \
      bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

    # Return either lcl or ldl
    if return_ldl and return_min_lcl_ldl:
        exit('return_ldl and return_min_lcl_ldl cannot both be true')
    elif return_ldl:
        return ldl
    elif return_min_lcl_ldl:
        return min(lcl,ldl)
    else:
        return lcl
        
        
def prt_dyer74(zL):
    """
    Computation of the turbulent Prandtl number as a function of the nondimensional height z/L (stability)
    following the formulation of Dyer (1974) as in Li, Atmospheric Research (2019) 
    https://doi.org/10.1016/j.atmosres.2018.09.015
    """
    if zL>0:
        return 1
    else:
        return 1/(1-16*zL)**0.25
    
    

    
    
def neggers_et_al_2006_stevens_et_al_2002(t,y,SST,D,q_free,th_free,ps,f,U_free,V_free,we, dict_ext):
    """
    Subscript s denotes surface values, b stands for buoyancy, which is associated with the virtual potential
    temperature (theta_v), th indicates the liquid water potential temperature (theta_l) and q the total specific
    humidity (q_t).
    A bulk model of the wind speed is added starting from the Stevens et al., JCli (2002) paper
    "Entrainment, Rayleigh friction, and boundary layer winds over the tropical Pacific"
    y[3] is U, the zonal bulk wind component
    y[4] is V, the meridional bulk wind component
    V_mag is the magnitude of the bulk wind = np.sqrt(y[3]**2+y[4]**2)
    """
    # Define some constants.
    delta_z = 100.0 # [m]
    Cqs = 0.0012 # [1], surface exchange coefficient for q
    Cths = 0.0012 # [1], surface exchange coefficient for theta
    Cqc = 0.1 # [1], cloud level exchange coefficient for q
    Cthc = 0.03 # [1], cloud level exchange coefficient for theta
    Le = 2.5e6 # [J/kg], latent heat of vaporization 
    cpd  = 1004.67 # [J/K/kg], dry air specific heat at constant pressure
    Rd = 287.1 # [J/K/kg], dry air gas constant
    # air density is supposed to be constant and equal to 1...
    beta = 0.36
    gamma = 1.55
    g = 9.81 # [m/s2], acceleration due to gravity
    CD = 1/900. # drag coefficient.
    
    ### Define some forcing variables for the BOMEX example -> this will be given as input parameters.
    #V = 8.75 # [m/s], large scale horizontal wind
    #SST = 300.4 # [K], sea surface temperature
    #D = 4.3e-6 # [1/s], large scale divergence
    #q_free = 4.0e-3 # [kg/kg] free tropospheric total mixing ratio
    #th_free = 308 # [K], free tropospheric liquid virtual potential temperature
    F_advq = -1.2e-3/86400 # [kg/kg/s]
    F_advth = 0.0/86400 # [K/s]
    F_rad = -2.0/86400 # [K/s]
    #ps = 1015 # [hPa], surface pressure
    ref_p = 1000 # [hPa], reference pressure for the potential temperature: is this the surface pressure????
    q_s = qsea(SST-273.15,ps)*1e-3 # [kg/kg], surface total specific humidity (saturation value)
    th_s = SST*(ref_p/ps)**(Rd/cpd) # [K], surface liquid water potential temperature, assuming no liquid water at the surface
    
    ### Define some diagnostic variables.
    # Entrainment.
    V_mag = np.sqrt(y[3]**2+y[4]**2) # [m/s], bulk wind speed.
    q_flux_s = V_mag*Cqs*(q_s-y[1]) # total specific humidity flux at the surface
    th_flux_s = V_mag*Cths*(th_s-y[2]) # liquid water potential temperature flux at the surface
    thv_flux_s = (1+0.61*q_s)*th_flux_s + 0.61*th_s*q_flux_s # surface virtual potential temp flux, <w'theta_v'>    # Entrainment.
    delta_q = Cqc*(q_free-y[1])
    delta_th = Cthc*(th_free-y[2])
    delta_thv = delta_th + 0.61*(y[1]*delta_th + y[2]*delta_q + delta_q*delta_th)
    E = 0.2*thv_flux_s/delta_thv
    # Mass flux.
    thv0 = y[2]*(1+0.61*y[1]) # [K], ABL virtual potential temperature
    b_flux_s = g*thv_flux_s/thv0 # surface buoyancy flux
    w_star = (y[0]*b_flux_s)**(1/3) # m/s, Deardorff convective velocity scale. 5th try
#    w_star = (g*y[0]*thv_flux_s/(y[2]*(ps/ref_p)**(Rd/cpd)*(1+0.61*y[1])))**(1/3) # m/s, Deardorff convective velocity scale. 4th try
#    w_star = (g*y[0]*thv_flux_s/(y[2]*(1+0.61*y[1])))**(1/3) # m/s, Deardorff convective velocity scale. 3rd try
#    w_star = (g*y[0]*thv_flux_s/y[2])**(1/3) # m/s, Deardorff convective velocity scale. 2nd try
#    w_star = (g*y[0]*thv_flux_s/th_s)**(1/3) # m/s, Deardorff convective velocity scale. 1st try    
    T0 = y[2]*(ps/ref_p)**(Rd/cpd) # [K], air temperature at the surface, from the ABL theta value.
    T_h = T0-g/cpd*y[0] # [K], air temperature at h following a dry adiabat.
    p_h = ps*100*(1-g*y[0]/(T0*cpd))**(cpd/Rd) # [Pa], air pressure at h with p=rho*R*T; dp/dz=-rho*g; dtheta/dz=0 
    e_sat_h = qsat(T_h-273.15,p_h/100)*100 # [Pa], saturation vapor pressure at h.
    q_sat = 0.622*e_sat_h/(p_h-0.378*e_sat_h) # [kg/kg], saturation specific humidity at h.   
    sigma_q = np.sqrt(-q_flux_s*delta_q*y[0]/(w_star*delta_z))
    area_c = 0.5 + beta*np.arctan(gamma*(y[1]-q_sat)/sigma_q)
    M = area_c*w_star
    # Entrainment for the horizontal momentum.
    u_star = V_mag*np.sqrt(CD) # [m/s], Friction velocity, assuming air density equal to 1... We can correct this!
    L = - u_star**3/(0.4*b_flux_s)# [m], Monin-Obukhov length
#     print
    zL = 10/L # [1], ratio z/L, we assume z=10 m as a reference height: we are interested in the surface stability
    we_dyn = E*prt_dyer74(zL)
#     print(we_dyn)
    
    ### Define the equations to be solved.
    dh_dt = E - D*y[0] - M 
    dq_dt = (q_flux_s + E*delta_q)/y[0] + F_advq
    dth_dt = (th_flux_s + E*delta_th)/y[0] + F_advth + F_rad
    dU_dt = f*(y[4]-V_free)-y[3]*(CD*V_mag+we_dyn)/y[0]+U_free*we_dyn/y[0]
    dV_dt = -f*(y[3]-U_free)-y[4]*(CD*V_mag+we_dyn)/y[0]+V_free*we_dyn/y[0]
    
    #print(area_c)
    #print((y[1]-q_sat)*1e3)
    #print(sigma_q*1e3)
    #print(w_star)
    #print((y[1]-q_sat)/sigma_q)
    
    dict_ext['area_c_ext'].append(area_c)                 # area_c_ext
    dict_ext['w_star_ext'].append(w_star)                 # w_star_ext
    dict_ext['M_ext'].append(M)                      # M_ext
    dict_ext['E_ext'].append(E)                      # E_ext
    dict_ext['sigma_q_ext'].append(sigma_q)                # sigma_q_ext
    dict_ext['LHF_ext'].append(q_flux_s*Le)            # LHF_ext
    
    return dh_dt, dq_dt, dth_dt, dU_dt, dV_dt