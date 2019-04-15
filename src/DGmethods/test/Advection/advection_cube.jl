using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using Printf
using LinearAlgebra
MPI.Initialized() || MPI.Init()
Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

const uid, vid, wid = 1:3

#{{{ advectionflux
@inline function advectionflux!(F,Q,aux,t)
    @inbounds begin
        u,v,w = aux[uid], aux[vid], aux[wid]
        ρ=Q[1]
        F[1,1], F[2,1], F[3,1]=u*ρ, v*ρ, w*ρ
    end
end
#}}} advectionflux

#{{{ advectionspeed
@inline function advectionspeed(n, Q, vel, t)
    abs(dot(vel,n))
end
#}}} advectionspeed

#{{{ numericalflux
numericalflux!(F, nM, QM, auxM, QP, auxP, t) = NumericalFluxes.rusanov!(F, nM, QM, auxM, QP, auxP, t, advectionflux!, advectionspeed)
#}}} numericalflux

#{{{ velocity initial condition
@inline function velocity_init!(vel, x, y, z)
    @inbounds begin
        DFloat = eltype(vel)
        vel[uid] = 2.0
        vel[vid] = 0.0
        vel[wid] = 0.0
    end
end
#}}} velocity initial condition

#{{{ Main
function main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          )

    spacedisc=DGBalanceLaw(grid = grid,
                           length_state_vector = 1,
                           inviscid_flux! = advectionflux!,
                           inviscid_numericalflux! = numericalflux!,
                           auxiliary_state_length = 3,
                           auxiliary_state_initialization! = velocity_init!)
    DGBalanceLawDiscretizations.writevtk("vtk/velocity_cube", spacedisc.auxstate, spacedisc, ("u", "v", "w"))

    Q = MPIStateArray(spacedisc) do Q, x, y, z, vel
        @inbounds begin
            DFloat = eltype(vel)
            xc, yc, zc = 0, 0, 0
            r=sqrt( (x-xc)^2 + (z-zc)^2 )
            Q[1] = exp(-5(r)^2 )
        end
    end
  DGBalanceLawDiscretizations.writevtk("vtk/ic_cube", Q, spacedisc, ("ρ",))
  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  io = MPI.Comm_rank(mpicomm) == 0 ? stdout : devnull
  eng0 = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ (initial) =  %.16e\n", eng0)

  # Set up the information callback
  timer = [time_ns()]
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(1, mpicomm) do (s=false)
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
  cbvtk = GenericCallbacks.EveryXSimulationSteps(10) do (init=false)
      outprefix = @sprintf("vtk/advection_cube_%dD_mpirank%04d_step%04d",
                             3, MPI.Comm_rank(mpicomm), step[1])
      @printf(io, "----\n")
      @printf(io, "doing VTK output =  %s\n", outprefix)
      DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, ("ρ", ))
      step[1] += 1
      nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

end
#}}} Main

#{{{ Run Program
let
    mpicomm=MPI.COMM_WORLD
    DFloat = Float64
    N=4
    ArrayType = Array
    dt=1e-2
    timeend=1.0
    Ne = (5,1,5) #number of elements
    dim = 3 #number of spatial dimensions
    brickrange = ntuple(j->range(DFloat(-1); length=Ne[j]+1, stop=1), dim)
    topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))

    main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end
#}}} Run Program

isinteractive() || MPI.Finalize()

#nothing
println("Done")
