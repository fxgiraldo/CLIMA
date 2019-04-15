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

#{{{ Cartesian->Spherical
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

    #Stay within bounds
#=    if (λ < λ_min)
        λ = λ + λ_max
    end
    if (λ > λ_max)
        λ = λ - λ_max
    end
    if (ϕ < ϕ_min)
        ϕ = ϕ + ϕ_max
    end
    if (ϕ > ϕ_max)
        ϕ = ϕ - ϕ_max
    end
    =#
    return r, λ, ϕ
end
#}}} Cartesian->Spherical

#{{{ numericalflux
numericalflux!(F, nM, QM, auxM, QP, auxP, t) = NumericalFluxes.rusanov!(F, nM, QM, auxM, QP, auxP, t, advectionflux!, advectionspeed)
#}}} numericalflux

#{{{ velocity initial condition
@inline function velocity_init!(vel, x, y, z)
    @inbounds begin
        DFloat = eltype(vel)
        (r, λ, ϕ) = cartesian_to_spherical(x,y,z,"radians")
        #w = 2 * DFloat(π) * cos(ϕ) #Case 1 -> shear flow
        w = 2 * DFloat(π) * cos(ϕ) * r #Case 2 -> solid body flow
        uλ, uϕ = w, 0
        vel[uid] = -uλ*sin(λ) - uϕ*cos(λ)*sin(ϕ)
        vel[vid] = +uλ*cos(λ) - uϕ*sin(λ)*sin(ϕ)
        vel[wid] = +uϕ*cos(ϕ)
    end
end
#}}} velocity initial condition

#{{{ Main
function main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp)

    #Define Spatial Discretization
    spacedisc=DGBalanceLaw(grid = grid,
                           length_state_vector = 1,
                           inviscid_flux! = advectionflux!,
                           inviscid_numericalflux! = numericalflux!,
                           auxiliary_state_length = 3,
                           auxiliary_state_initialization! = velocity_init!)
    DGBalanceLawDiscretizations.writevtk("vtk/velocity_sphere", spacedisc.auxstate, spacedisc, ("u", "v", "w"))

    Q = MPIStateArray(spacedisc) do Q, x, y, z, vel
        @inbounds begin
            DFloat = eltype(vel)
            (r, λ, ϕ) = cartesian_to_spherical(x,y,z,"radians")
            Q[1] = exp(-((3λ)^2 + (3ϕ)^2))
        end
    end
    DGBalanceLawDiscretizations.writevtk("vtk/ic_sphere", Q, spacedisc, ("ρ",))
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
      outprefix = @sprintf("vtk/advection_sphere_%dD_mpirank%04d_step%04d",
                             3, MPI.Comm_rank(mpicomm), step[1])
      @printf(io, "----\n")
      @printf(io, "doing VTK output =  %s\n", outprefix)
      DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, ("ρ", ))
      step[1] += 1
      nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  DGBalanceLawDiscretizations.writevtk("vtk/velocity_final_sphere", spacedisc.auxstate,
                                       spacedisc, ("u", "v", "w"))

end
#}}} Main

#{{{ Run Program
let
    mpicomm=MPI.COMM_WORLD
    DFloat = Float64
    N=4
    ArrayType = Array
    dt=1e-3
    timeend=1.0
    Nhorz = 5 #number of horizontal elements per face of cubed-sphere grid
    Nvert = 5 #number of horizontal elements per face of cubed-sphere grid
    Rrange=range(DFloat(1); length=Nhorz+1, stop=2)
    topl = StackedCubedSphereTopology(mpicomm,Nhorz,Rrange; bc=(0,0))
    main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end
#}}} Run Program

isinteractive() || MPI.Finalize()

#nothing
println("Done")
