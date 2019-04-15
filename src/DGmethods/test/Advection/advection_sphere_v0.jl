using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using LinearAlgebra
using Printf
MPI.Initialized() || MPI.Init()
Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

const uid, vid, wid = 1:3

#{{{ advectionflux
@inline function advectionflux!(F, Q, vel, t)
  @inbounds begin
    # get velocites out of auxiliary state array
    u, v, w = vel[uid], vel[vid], vel[wid]

    # get density from the state vector
    ρ = Q[1]

    # set the three components of the flux
    F[1, 1], F[2, 1], F[3, 1] = u * ρ, v * ρ, w * ρ
  end
end
#}}} advectionflux

#{{{ advectionspeed
# wave speed is just the dot product of velocity field and normal vector
@inline function advectionspeed(n, Q, vel, t)
  abs(dot(vel, n))
end
#}}} advectionspeed

#{{{ numericalflux
numericalflux!(F, nM, QM, auxM, QP, auxP, t) =
  NumericalFluxes.rusanov!(F, nM, QM, auxM, QP, auxP, t, advectionflux!,
                           advectionspeed)
#}}} numericalflux

#{{{ velocity initial condition
@inline function velocity_init!(vel, x, y, z)
  @inbounds begin
    DFloat = eltype(vel)
    r = hypot(x, y, z)
    λ = atan(y, x)
    ϕ = asin(z / r)
    #uλ = 2 * DFloat(π) * cos(ϕ) #Case 1 -> shear flow
    uλ = 2 * DFloat(π) * cos(ϕ) * r #Case 2 -> solid body flow

    vel[uid] = -uλ * sin(λ)
    vel[vid] =  uλ * cos(λ)
    vel[wid] =  0
  end
end
#}}} velocity initial condition

#{{{ Main
function main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
  # create the grid of DOFs and warp out to the sphere
  grid = DiscontinuousSpectralElementGrid(topl;
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp)

  #Quick Hack to spit-out Grid
  #=
  Nq = N + 1
  dim = 3
  nelem = size(grid.vgeo)[end]
  Xid = (grid.xid, grid.yid, grid.zid)
  X = ntuple(j->reshape((@view grid.vgeo[:, Xid[j], :]),
                        ntuple(j->Nq, dim)...,
                        nelem), dim)
  DGBalanceLawDiscretizations.writemesh("mesh", X...)
  =#

  # create the spatial discretization with 1 state (density) and 3 auxiliary
  # fields (u, v, w)
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = 1,
                           inviscid_flux! = advectionflux!,
                           inviscid_numericalflux! = numericalflux!,
                           auxiliary_state_length = 3,
                           auxiliary_state_initialization! = velocity_init!)
  DGBalanceLawDiscretizations.writevtk("vtk/velocity_init", spacedisc.auxstate,
                                       spacedisc, ("u", "v", "w"))

  Q = MPIStateArray(spacedisc) do Q, x, y, z, vel
    @inbounds begin
      DFloat = eltype(Q)
      r = hypot(x, y, z)
      λ = atan(y, x)
      ϕ = asin(z / r)
      Q[1] = exp(-((3λ)^2 + (3ϕ)^2))
    end
  end
  DGBalanceLawDiscretizations.writevtk("vtk/ic", Q, spacedisc, ("ρ",))

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  io = MPI.Comm_rank(mpicomm) == 0 ? stdout : devnull
  eng0 = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ (initial) =  %.16e\n", eng0)

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
          @printf(io, "||Q||₂  =  %.16e\n", norm(Q))
      end
      nothing
  end

  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
      outprefix = @sprintf("vtk/sphere_%dD_mpirank%04d_step%04d",
                             3, MPI.Comm_rank(mpicomm), step[1])
      @printf(io, "----\n")
      @printf(io, "doing VTK output =  %s\n", outprefix)
      DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, ("ρ", ))
      step[1] += 1
      nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  DGBalanceLawDiscretizations.writevtk("vtk/velocity_final", spacedisc.auxstate,
                                       spacedisc, ("u", "v", "w"))

end
#}}} Main

let
  mpicomm = MPI.COMM_WORLD
  DFloat = Float64
  N = 4 #polynomial order
  ArrayType = Array
  dt = 1e-3
  timeend = 1

  Nhorz = 5 #number of horizontal elements per face of cubed-sphere grid
  Rrange = range(DFloat(1); length=2*Nhorz+1, stop=2) #number of vertical elements

  topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange; bc = (0,0))

  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end

isinteractive() || MPI.Finalize()

nothing
