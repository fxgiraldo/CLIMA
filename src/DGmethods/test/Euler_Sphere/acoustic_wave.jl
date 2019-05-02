# Standard isentropic vortex test case.  For a more complete description of
# the setup see for Example 3 of:
#
# @article{ZHOU2003159,
#   author = {Y.C. Zhou and G.W. Wei},
#   title = {High resolution conjugate filters for the simulation of flows},
#   journal = {Journal of Computational Physics},
#   volume = {189},
#   number = {1},
#   pages = {159--179},
#   year = {2003},
#   doi = {10.1016/S0021-9991(03)00206-7},
#   url = {https://doi.org/10.1016/S0021-9991(03)00206-7},
# }
#
# This version runs the isentropic vortex as a stand alone test (no dependence
# on CLIMA moist thermodynamics)

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
MPI.Initialized() || MPI.Init()
Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

#Earth Constants
const gravity = 9.80616
const earth_radius = 6.37122e6
const rgas = 287.17
const cp = 1004.67
const cv = 717.5
const p00 = 1.0e5
const ztop = 10.0e3 #10km
#Earth Constants

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")
const γ_exact = 7 // 5

const integration_testing = false

# preflux computation
@inline function preflux(Q, _...)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2), u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

# Cartesian->Spherical
@inline function cartesian_to_spherical(x,y,z,format)
    #Conversion Constants
    if format == "degrees"
        c=180/π
    elseif format == "radians"
        c=1.0
    end
    λ_max=2π*c
    λ_min=0*c
    ϕ_max=+0.5*π*c
    ϕ_min=-0.5*π*c

    #Conversion functions
    r = hypot(x,y,z)
    λ=atan(y,x)*c
    ϕ=asin(z/r)*c

    return (r, λ, ϕ)
end

# Construct Geopotential
const _nauxstate = 7
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_x, _a_y, _a_z = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
    @inbounds begin
        r=hypot(x,y,z) - earth_radius
        aux[_a_ϕ] = gravity*r
        aux[_a_x] = x
        aux[_a_y] = y
        aux[_a_z] = z
    end
end

# Construct Euler Source which is ρ*grad(ϕ)
@inline function euler_source!(S, Q, aux, t)
  @inbounds begin
      ρ, ϕx, ϕy, ϕz =  Q[_ρ], aux[_a_ϕx], aux[_a_ϕy], aux[_a_ϕz]
      S[_ρ] = 0
      S[_U] = - ρ*ϕx
      S[_V] = - ρ*ϕy
      S[_W] = - ρ*ϕz
      S[_E] = 0
  end
end

# physical flux function
euler_flux!(F, Q, aux, t) = euler_flux!(F, Q, aux, t, preflux(Q)...)

# Euler flux function
@inline function euler_flux!(F, Q, aux, t, P, u, v, w, ρinv)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
  end
end

@inline function bcstate!(QP, _, QM, auxM, nM, bctype, t,  PM, uM, vM, wM, ρMinv)
    DFloat = eltype(QM)
    γ:: DFloat = γ_exact
    @inbounds begin
        if bctype == 1 #no-flux
            #Store values at left of boundary ("-" values)
            ρM, UM, VM, WM, EM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E]
            ϕM=auxM[_a_ϕ]

            #Scalars are the same on both sides of the boundary
            ρP=ρM; PP=PM; ϕP=ϕM
            nx, ny, nz = nM[1], nM[2], nM[3]

            #reflect velocities
            uN=nx*uM + ny*vM + nz*wM
            uP=uM - 2*uN*nx
            vP=vM - 2*uN*ny
            wP=wM - 2*uN*nz

            #Construct QP state
            QP[_ρ], QP[_U], QP[_V], QP[_W] = ρP, ρP*uP, ρP*vP, ρP*wP
            QP[_E]= PP/(γ-1) + 0.5*ρP*( uP^2 + vP^2 + wP^2) + ρP*ϕP
        end
    nothing
  end
end

# initial condition
function acoustic_wave!(Q, t, x, y, z, aux, _...)
    DFloat = eltype(Q)
    γ:: DFloat = γ_exact

    #Test Case Constants
    a :: DFloat = earth_radius
    R_pert :: DFloat = a/3.0  #!Radius of perturbation
    nv :: DFloat = 1.0
    T0 :: DFloat = 300
    #Test Case Constants

    #Convert to Spherical Coords
    (r, λ, ϕ) = cartesian_to_spherical(x,y,z,"radians")
    h = r - a
    cosϕ = cos(ϕ)
    cosλ = cos(λ)
    sinλ = sin(λ)

    #Potential Temperature for an isothermal atmosphere
    θ_ref = T0*exp(gravity*h/(cp*T0))
    #Hydrostatic pressure from the def. of potential temp
    p_ref = p00*(T0/θ_ref)^(cp/rgas)
    #Density from the ideal gas law
    ρ_ref = (p00/(rgas*θ_ref))*(p_ref/p00)^(cv/cp)

    #Pressure Perturbation
    r1 = a*acos(cosϕ*cosλ)
    if (r1 < R_pert)
        f = 0.5*(1 + cos(π*r1/R_pert))
    else
        f = 0
    end

    #vertical profile
    #=
    if (nelz*nglz <= 2) #only two or fewer points in the vertical
        g = 2/(π*nv)
    else
        g = sin(nv*π*h/ztop)
    end
    =#
#    g = 2/(π*nv)
    g = sin(nv*π*h/ztop)

    dp = 100*f*g #Original
    p = p_ref + dp
    ρ = p00/(rgas*θ_ref)*(p/p00)^(cv/cp)

    #Fields
    ϕ = aux[_a_ϕ]
    ###Debug
    u, v, w = 0, 0, 0
    U = ρ*u
    V = ρ*v
    W = ρ*w
    E = p/(γ-1) + 0.5*ρ*(u^2 + v^2 + w^2) + ρ*ϕ

    #Store Initial conditions
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end

#{{{ Main
function main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)

    grid = DiscontinuousSpectralElementGrid(topl,
                                            FloatType = DFloat,
                                            DeviceArray = ArrayType,
                                            polynomialorder = N,
                                            meshwarp = Topologies.cubedshellwarp)

    # spacedisc = data needed for evaluating the right-hand side function
    numflux!(x...) = NumericalFluxes.rusanov!(x..., euler_flux!, wavespeed,
                                              preflux)
    numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., euler_flux!,
                                                              bcstate!, wavespeed,
                                                              preflux)
    spacedisc = DGBalanceLaw(grid = grid,
                             length_state_vector = _nstate,
                             inviscid_flux! = euler_flux!,
                             inviscid_numerical_flux! = numflux!,
                             inviscid_numerical_boundary_flux! = numbcflux!,
                             auxiliary_state_length = _nauxstate,
                             auxiliary_state_initialization! = auxiliary_state_initialization!,
                             source! = euler_source!)

    DGBalanceLawDiscretizations.grad_auxiliary_state!(spacedisc, _a_ϕ, (_a_ϕx, _a_ϕy, _a_ϕz))

    # This is an actual state/function that lives on the grid
    initialcondition(Q, x...) = acoustic_wave!(Q, DFloat(0), x...)
    Q = MPIStateArray(spacedisc, initialcondition)

    #Store Initial Condition as Exact Solution
    Qe = copy(Q)

    lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

    io = MPI.Comm_rank(mpicomm) == 0 ? stdout : devnull

    # Set up the information callback
    timer = [time_ns()]
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
        if s
            timer[1] = time_ns()
        else
            run_time = (time_ns() - timer[1]) * 1e-9
            (min, sec) = fldmod(run_time, 60)
            (hrs, min) = fldmod(min, 60)
            @printf(io, "----\n")
            @printf(io, "simtime =  %.16e\n", ODESolvers.gettime(lsrk))
            @printf(io, "runtime =  %03d:%02d:%05.2f (hour:min:sec)\n", hrs, min, sec)
            @printf(io, "|Mass_Final - Mass_Initial|/Mass_initial =  %.16e\n", abs(weighted_sum(Q) - weighted_sum(Qe))/ weighted_sum(Qe) )
        end
        nothing
    end

    # Set up the information callback
    cbmass = GenericCallbacks.EveryXSimulationSteps(1) do
        @printf(io, "----\n")
        @printf(io, "simtime =  %.16e\n", ODESolvers.gettime(lsrk))
        @printf(io, "|Mass_Final - Mass_Initial|/Mass_initial =  %.16e\n", abs(weighted_sum(Q) - weighted_sum(Qe))/ weighted_sum(Qe) )
        @printf(io, "Mass_initial =  %.16e\n", weighted_sum(Qe) )
        @printf(io, "Mass_Final   =  %.16e\n", weighted_sum(Q) )
        nothing
    end

    # Set up the VTK callback
    step = [0]
    mkpath("vtk")
    cbvtk = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        outprefix = @sprintf("vtk/acoustic_wave_mpirank%04d_step%04d",
                             MPI.Comm_rank(mpicomm), step[1])
        @debug "doing VTK output" outprefix
        DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames)
        step[1] += 1
        nothing
    end

    solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbmass, cbvtk))

    # Print some end of the simulation information
    if integration_testing
        mass_initial = weighted_sum(Qe)
        mass_final   = weighted_sum(Q)
        @info @sprintf """Finished
        (Mass_Final - Mass_Initial) / Mass_Initial = %.16e
        """ abs(mass_final - mass_initial) / mass_initial
    end

end
#}}} Main

#{{{ Run Program
let
    mpicomm=MPI.COMM_WORLD
    DFloat = Float64
    N=4
    ArrayType = Array
    dt=1
    timeend=100
    Nhorz = 5 #number of horizontal elements per face of cubed-sphere grid
    Nvert = 5 #number of horizontal elements per face of cubed-sphere grid
    height_min=earth_radius
    height_max=earth_radius + ztop
    Rrange=range(DFloat(height_min); length=Nhorz+1, stop=height_max)
    topl = StackedCubedSphereTopology(mpicomm,Nhorz,Rrange; bc=(1,1))
    main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end
#}}} Run Program

isinteractive() || MPI.Finalize()

#nothing
println("Done")

