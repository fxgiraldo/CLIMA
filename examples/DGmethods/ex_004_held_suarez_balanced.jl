# # Example 004: Held-Suarez on the Sphere
#
#md # !!! jupyter
#md #     This example is also available as a Jupyter notebook:
#md #     [`ex_004_held_suarez_balanced.ipynb`](@__NBVIEWER_ROOT_URL__examples/DGmethods/generated/ex_004_held_suarez_balanced.html)
#
# ## Introduction
#
# In this example we will set up and run the Held-Suarez test case from
# Held and Suarez (1994) (https://journals.ametsoc.org/doi/pdf/10.1175/1520-0477%281994%29075%3C1825%3AAPFTIO%3E2.0.CO%3B2)
# ```
#
# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref ex_004_held-suarez-plain-program).
#
# ## Commented Program
#------------------------------------------------------------------------------
#--------------------------------#
#--------------------------------#
# Can be run with:
# CPU: mpirun -n 1 julia --project=@. ex_004_held_suarez_balanced.jl
# GPU: mpirun -n 1 julia --project=/home/fxgiraldo/CLIMA/env/gpu ex_004_held_suarez_balanced.jl
#--------------------------------#
#--------------------------------#

# ### Preliminaries
# Load in modules needed for solving the problem
using MPI
using Logging
using LinearAlgebra
using Dates
using Printf
using CLIMA
using CLIMA.Topologies
using CLIMA.MPIStateArrays
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.Vtk
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using StaticArrays

# Though not required, here we are explicit about which values we read out the
# `PlanetParameters` and `MoistThermodynamics`
using CLIMA.PlanetParameters: planet_radius, R_d, cp_d, grav, cv_d, MSLP
#using CLIMA.PlanetParameters: planet_radius, grav, MSLP
using CLIMA.MoistThermodynamics: air_temperature, air_pressure, internal_energy,
                                 soundspeed_air, air_density, gas_constant_air

# Start up MPI if this has not already been done
MPI.Initialized() || MPI.Init()
#md nothing # hide

# If `CuArrays` is in the current environment we will use CUDA, otherwise we
# drop back to the CPU
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const DeviceArrayType = CuArray
else
  const DeviceArrayType = Array
end
#md nothing # hide

# Specify whether to enforce hydrostatic balance at PDE level or not
const PDE_level_hydrostatic_balance = true
# TODO: What should this value be????
const nu_artificial = 0.1

# check whether to use default VTK directory or define something else
VTKDIR = get(ENV, "CLIMA_VTK_DIR", "vtk")

# Here we setup constants to for some of the computational parameters; the
# underscore is just syntactic sugar to indicate that these are constants.

# These are parameters related to the Euler state. Here we used the conserved
# variables for the state: perturbation in density, three components of
# momentum, and perturbation to the total energy.
const _nstate = 5
const _dρ, _ρu, _ρv, _ρw, _dρe = 1:_nstate
const _statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")
#md nothing # hide

# TODO: FILL COMMENTS HERE
const _nviscstates = 6
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23 = 1:_nviscstates

const _ngradstates = 3
const _states_for_gradient_transform = (_dρ, _ρu, _ρv, _ρw)
const _gt_dρ, _gt_ρu, _gt_ρv, _gt_ρw = 1:length(_states_for_gradient_transform)

# These will be the auxiliary state which will contain the geopotential,
# gradient of the geopotential, and reference values for density and total
# energy
const _nauxstate = 6
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_ρ_ref, _a_ρe_ref = 1:_nauxstate
const _auxnames = ("ϕ", "ϕx", "ϕy", "ϕz", "ρ_ref", "ρe_ref")
#md nothing # hide

#------------------------------------------------------------------------------

# ### Definition of the physics
# Though we are interested in solving the compressible Euler equations, we also
# add some numerical viscosity to stabelize the method.
#
# We first define the Euler flux function:
function eulerflux!(F, Q, aux)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## set the actual flux
    F[1, _dρ ], F[2, _dρ ], F[3, _dρ ] = ρu          , ρv          , ρw
    if PDE_level_hydrostatic_balance
      δP = P - P_ref
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + δP, v * ρu     , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv      , v * ρv + δP, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw      , v * ρw     , w * ρw + δP
    else
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + P, v * ρu    , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv     , v * ρv + P, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw     , v * ρw    , w * ρw + P
    end
    F[1, _dρe], F[2, _dρe], F[3, _dρe] = u * (ρe + P), v * (ρe + P), w * (ρe + P)
  end
end
#md nothing # hide

function velocities!(vel, Q, aux, t)
  @inbounds begin
    # ordering should match states_for_gradient_transform
    dρ, ρu, ρv, ρw = Q[_gt_dρ], Q[_gt_ρu], Q[_gt_ρv], Q[_gt_ρw]
    ρ_ref = aux[_a_ρ_ref]

    ρ = ρ_ref + dρ
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    vel[1], vel[2], vel[3] = u, v, w
  end
end

function compute_stresses!(QV, grad_vel, _...)
  nu::eltype(QV) = nu_artificial
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]

    # strains
    ϵ11 = dudx
    ϵ22 = dvdy
    ϵ33 = dwdz
    ϵ12 = (dudy + dvdx) / 2
    ϵ13 = (dudz + dwdx) / 2
    ϵ23 = (dvdz + dwdy) / 2

    # deviatoric stresses
    QV[_τ11] = -2nu * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
    QV[_τ22] = -2nu * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
    QV[_τ33] = -2nu * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
    QV[_τ12] = -2nu * ϵ12
    QV[_τ13] = -2nu * ϵ13
    QV[_τ23] = -2nu * ϵ23
  end
end

# We use an average numerical flux for the gradient equation, which leads to a
# different divide by two when the minus side value is subtracted off
function stresses_penalty!(QV, nM, velM, _, _, velP, _, _, _)
  @inbounds begin
    n_Δvel = similar(QV, Size(3, 3))
    for j = 1:3, i = 1:3
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end

    ## "convert" from a strain penalty to a stress penalty
    compute_stresses!(QV, n_Δvel)
  end
end

# Neumann boundary condition
function stresses_boundary_penalty_neumann!(QV, nM, velM, _, _, velP, _, _, _, _)
  @inbounds begin
    n_Δvel = similar(QV, Size(3, 3))
    for j = 1:3, i = 1:3
      n_Δvel[i, j] = nM[i] * (0 - velM[j]) / 2
    end

    ## "convert" from a strain penalty to a stress penalty
    compute_stresses!(QV, n_Δvel)
  end
end

function divergenceviscousflux!(F, Q, QV, aux)
  @inbounds begin
    dρ, ρu, ρv, ρw = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw]
    ρ_ref = aux[_a_ρ_ref]

    ρ = ρ_ref + dρ
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ρτ11, ρτ22, ρτ33 = ρ * QV[_τ11], ρ * QV[_τ22], ρ * QV[_τ33]
    ρτ12 = ρτ21 = ρ * QV[_τ12]
    ρτ13 = ρτ31 = ρ * QV[_τ13]
    ρτ23 = ρτ32 = ρ * QV[_τ23]

    ## add in the viscous terms
    F[1, _ρu] += ρτ11; F[2, _ρu] += ρτ12; F[3, _ρu] += ρτ13
    F[1, _ρv] += ρτ21; F[2, _ρv] += ρτ22; F[3, _ρv] += ρτ23
    F[1, _ρw] += ρτ31; F[2, _ρw] += ρτ32; F[3, _ρw] += ρτ33

    F[1, _dρe] += u * ρτ11 + v * ρτ12 + w * ρτ13
    F[2, _dρe] += u * ρτ21 + v * ρτ22 + w * ρτ23
    F[3, _dρe] += u * ρτ31 + v * ρτ32 + w * ρτ33
  end
end

function physicalflux!(F, Q, QV, aux, t)
  eulerflux!(F, Q, aux)
  divergenceviscousflux!(F, Q, QV, aux)
end

# Define the geopotential source from the solution and auxiliary variables
function geopotential!(S, Q, aux, t)
  @inbounds begin
    ρ_ref, ϕx, ϕy, ϕz = aux[_a_ρ_ref], aux[_a_ϕx], aux[_a_ϕy], aux[_a_ϕz]
    dρ = Q[_dρ]
    S[_dρ ] = 0
    if PDE_level_hydrostatic_balance
      S[_ρu ] = -dρ * ϕx
      S[_ρv ] = -dρ * ϕy
      S[_ρw ] = -dρ * ϕz
    else
      ρ = ρ_ref + dρ
      S[_ρu ] = -ρ * ϕx
      S[_ρv ] = -ρ * ϕy
      S[_ρw ] = -ρ * ϕz
    end
    S[_dρe] = 0
  end
end
#md nothing # hide

# This defines the local wave speed from the current state (this will be needed
# to define the numerical flux)
function wavespeed(n, Q, aux, _...)
  @inbounds begin
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]

    ## get total energy and density
    ρ = ρ_ref + dρ
    e = (ρe_ref + dρe) / ρ

    ## velocity field
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the temperature
    T = air_temperature(e_int)

    abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(T)
  end
end
#md nothing # hide

# The only boundary condition needed for this test problem is the no flux
# boundary condition, the state for which is defined below. This function
# defines the plus-side (exterior) values from the minus-side (inside) values.
# This plus-side value will then be fed into the numerical flux routine in order
# to enforce the boundary condition.
function nofluxbc!(QP, QVP, _, nM, QM, QVM, auxM, _...)
  @inbounds begin
    DFloat = eltype(QM)
    ## get the minus values
    dρM, ρuM, ρvM, ρwM, dρeM = QM[_dρ], QM[_ρu], QM[_ρv], QM[_ρw], QM[_dρe]

    ## scalars are preserved
    dρP, dρeP = dρM, dρeM

    ## vectors are reflected
    nx, ny, nz = nM[1], nM[2], nM[3]

    ## reflect velocities
    mag_ρu⃗ = nx * ρuM + ny * ρvM + nz * ρwM
    ρuP = ρuM - 2mag_ρu⃗ * nx
    ρvP = ρvM - 2mag_ρu⃗ * ny
    ρwP = ρwM - 2mag_ρu⃗ * nz

    ## Construct QP state
    QP[_dρ], QP[_ρu], QP[_ρv], QP[_ρw], QP[_dρe] = dρP, ρuP, ρvP, ρwP, dρeP

    # Neumann BC
    QVP .= QVM
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Definition of the problem
# Here we define the initial condition as well as the auxiliary state (which
# contains the reference state on which the initial condition depends)

# First it is useful to have a conversion function going between Cartesian and
# spherical coordinates (defined here in terms of radians)
function cartesian_to_spherical(DFloat, x, y, z)
    r = hypot(x, y, z)
    λ = atan(y, x)
    φ = asin(z / r)
    (r, λ, φ)
end
#md nothing # hide

# FXG: reference state
# Setup the initial condition based on a N^2=0.01 uniformly stratified atmosphere with θ0=315K
function auxiliary_state_initialization!(T0, aux, x, y, z)
  @inbounds begin
    DFloat = eltype(aux)
    p0 = DFloat(MSLP)
    θ0 = DFloat(315)
    N_bv=DFloat(0.01)

    ## Convert to Spherical coordinates
    (r, λ, φ) = cartesian_to_spherical(DFloat, x, y, z)

    ## Calculate the geopotential ϕ
    h = r - DFloat(planet_radius) # height above the planet surface
    ϕ = DFloat(grav) * h

    ## Reference Potential Temperature from constant Brunt-Vaisala definition
    θ_ref = θ0 * exp( N_bv^2*h/grav )

    ## Reference Exner Pressure from hydrostatic equation
    π_ref = 1 + grav^2/(cp_d*θ0*N_bv^2)*(exp(-N_bv^2*h/grav)- 1)

    ## Calculate pressure from exner pressure definition
    P_ref = p0*(π_ref)^(cp_d/R_d)

    ## Calculate temperature
    T_ref=θ_ref*π_ref

    ## Density from the ideal gas law
    ρ_ref = air_density(T_ref, P_ref)

    ## Calculate the reference total potential energy
    e_int = internal_energy(T_ref)
    ρe_ref = e_int * ρ_ref + ρ_ref * ϕ

    ## Fill the auxiliary state array
    aux[_a_ϕ] = ϕ
    ## gradient of the geopotential will be computed numerically below
    aux[_a_ϕx] = 0
    aux[_a_ϕy] = 0
    aux[_a_ϕz] = 0
    aux[_a_ρ_ref]  = ρ_ref
    aux[_a_ρe_ref] = ρe_ref
  end
end
#md nothing # hide

# FXG: initial conditions
# Setup the initial condition based on a N^2=0.01 uniformly stratified atmosphere
function initialcondition!(domain_height, Q, x, y, z, aux, _...)
  @inbounds begin
    DFloat = eltype(Q)
    p0 = DFloat(MSLP)

    (r, λ, φ) = cartesian_to_spherical(DFloat, x, y, z)
    h = r - DFloat(planet_radius)

    ## Get the reference pressure from the previously defined reference state
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]
    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## Define the initial pressure and compute the density perturbation
    P = P_ref
    T = T_ref
    ρ = air_density(T, P)
    dρ = ρ - ρ_ref

    ## Define the initial total energy perturbation
    e_int = internal_energy(T)
    ρe = e_int * ρ + ρ * ϕ
    dρe = ρe - ρe_ref
    
    ## Store Initial conditions
#    Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe] =  dρ, 0, 0, 0, dρe
    Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe] =  0, 0, 0, 0, 0
  end
end
#md nothing # hide

# This function compute the pressure perturbation for a given state. It will be
# used only in the computation of the pressure perturbation prior to writing the
# VTK output.
function compute_δP!(δP, Q, _, aux)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ## Compute the reference pressure
    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## Compute the full states
    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    ## store the pressure perturbation
    δP[1] = P - P_ref
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initialize the DG Method
function setupDG(mpicomm, Ne_vertical, Ne_horizontal, polynomialorder,
                 ArrayType, domain_height, T0, DFloat)

  ## Create the element grid in the vertical direction
  Rrange = range(DFloat(planet_radius), length = Ne_vertical + 1,
                 stop = planet_radius + domain_height)

  ## Set up the mesh topology for the sphere
  topology = StackedCubedSphereTopology(mpicomm, Ne_horizontal, Rrange)

  ## Set up the grid for the sphere. Note that here we need to pass the
  ## `cubedshellwarp` shell `meshwarp` function so that the degrees of freedom
  ## lay on the sphere (and not just stacked cubes)
  grid = DiscontinuousSpectralElementGrid(topology;
                                          polynomialorder = polynomialorder,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          meshwarp = Topologies.cubedshellwarp)

  ## Here we use the Rusanov numerical flux which requires the physical flux and
  ## wavespeed
  numflux!(x...) = NumericalFluxes.rusanov!(x..., physicalflux!, wavespeed)

  ## We also use Rusanov to define the numerical boundary flux which also
  ## requires a definition of the state to use for the "plus" side of the
  ## boundary face (calculated here with `nofluxbc!`)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., physicalflux!,
                                                            nofluxbc!,
                                                            wavespeed)

  auxinit(x...) = auxiliary_state_initialization!(T0, x...)
  ## Define the balance law solver
  spatialdiscretization = DGBalanceLaw(grid = grid,
                                       length_state_vector = _nstate,
                                       flux! = physicalflux!,
                                       source! = geopotential!,
                                       numerical_flux! = numflux!,
                                       numerical_boundary_flux! = numbcflux!,

                                       auxiliary_state_length = _nauxstate,
                                       auxiliary_state_initialization! =
                                       auxinit,

                                       number_gradient_states = _ngradstates,
                                       states_for_gradient_transform =
                                       _states_for_gradient_transform,
                                       number_viscous_states = _nviscstates,
                                       gradient_transform! = velocities!,
                                       viscous_transform! = compute_stresses!,
                                       viscous_penalty! = stresses_penalty!,
                                       viscous_boundary_penalty! =
                                       stresses_boundary_penalty_neumann!,
                                      )

  ## Compute Gradient of Geopotential
  DGBalanceLawDiscretizations.grad_auxiliary_state!(spatialdiscretization, _a_ϕ,
                                                    (_a_ϕx, _a_ϕy, _a_ϕz))

  spatialdiscretization
end
#md nothing # hide

#FXG: run program
# ### Initializing and run the DG method
# Note that the final time and grid size are small so that CI and docs
# generation happens in a reasonable amount of time. Running the simulation to a
# final time of `33` hours allows the wave to propagate all the way around the
# sphere and back. Increasing the numeber of horizontal elements to `~30` is
# required for stable long time simulation.
let
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)

  ## parameters for defining the cubed sphere.
  Ne_vertical   = 6  # number of vertical elements (small for CI/docs reasons)
  ## Ne_vertical   = 30 # Resolution required for stable long time result
  ## cubed sphere will use Ne_horizontal * Ne_horizontal horizontal elements in
  ## each of the 6 faces
  Ne_horizontal = 3

  polynomialorder = 5

  ## top of the domain
  domain_height = 30e3

  ## Sea surfaxe temperature
  T0 = 315

  ## Floating point type to use in the calculation
  DFloat = Float64

  spatialdiscretization = setupDG(mpicomm, Ne_vertical, Ne_horizontal,
                                  polynomialorder, DeviceArrayType,
                                  domain_height, T0, DFloat)

  Q = MPIStateArray(spatialdiscretization,
                    (x...) -> initialcondition!(domain_height, x...))

  ## Since we are using explicit time stepping the acoustic wave speed will
  ## dominate our CFL restriction along with the vertical element size
  element_size = (domain_height / Ne_vertical)
  acoustic_speed = soundspeed_air(DFloat(T0))
  dt = element_size / acoustic_speed / polynomialorder^2
  dt = 1
  
  ## Adjust the time step so we exactly hit 1 hour for VTK output
  #dt = 3600 / ceil(3600 / dt)

  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  ## Uncomment line below to extend simulation time and output less frequently
  #finaltime = N * 3600 #N hours
  #finaltime = N * 86400 #N days
  finaltime = 10000
  outputtime = 1000

  @show(polynomialorder,Ne_horizontal,Ne_vertical,dt,finaltime,finaltime/dt)

  ## We will use this array for storing the pressure to write out to VTK
  δP = MPIStateArray(spatialdiscretization; nstate = 1)

  ## Define a convenience function for VTK output
  mkpath(VTKDIR)
  function do_output(vtk_step)
    ## name of the file that this MPI rank will write
    filename = @sprintf("%s/held_suarez_balanced_mpirank%04d_step%04d",
                        VTKDIR, MPI.Comm_rank(mpicomm), vtk_step)

    ## fill the `δP` array with the pressure perturbation
    DGBalanceLawDiscretizations.dof_iteration!(compute_δP!, δP,
                                               spatialdiscretization, Q)

    ## write the vtk file for this MPI rank
    writevtk(filename, Q, spatialdiscretization, _statenames, δP, ("δP",))

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
      ## name of the pvtu file
      pvtuprefix = @sprintf("held_suarez_balanced_step%04d", vtk_step)

      ## name of each of the ranks vtk files
      prefixes = ntuple(i->
                        @sprintf("%s/held_suarez_balanced_mpirank%04d_step%04d",
                                 VTKDIR, i-1, vtk_step),
                        MPI.Comm_size(mpicomm))

      ## Write out the pvtu file
      writepvtu(pvtuprefix, prefixes, (_statenames..., "δP",))

      ## write that we have written the file
      with_logger(mpi_logger) do
        @info @sprintf("Done writing VTK: %s", pvtuprefix)
      end
    end
  end

  ## Setup callback for writing VTK every hour of simulation time and dump
  #initial file
  vtk_step = 0
  do_output(vtk_step)
  cb_vtk = GenericCallbacks.EveryXSimulationSteps(floor(outputtime / dt)) do
    vtk_step += 1
    do_output(vtk_step)
    nothing
  end

  ## Setup a callback to display simulation runtime information
  starttime = Ref(now())
  cb_info = GenericCallbacks.EveryXWallTimeSeconds(1, mpicomm) do (init=false)
    if init
      starttime[] = now()
    end
    with_logger(mpi_logger) do
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     norm(Q))
    end
  end

  solve!(Q, lsrk; timeend = finaltime, callbacks = (cb_vtk, cb_info))

end
#md nothing # hide

# ### Finalizing MPI (if necessary)
Sys.iswindows() || MPI.finalize_atexit()
Sys.iswindows() && !isinteractive() && MPI.Finalize()
#md nothing # hide

#md # ## [Plain Program](@id ex_004_held_suarez_balanced-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here:
#md # [ex\_004\_held\_suarez_balanced.jl](ex_004_held_suarez_balanced.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
