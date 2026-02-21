## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

using Revise

using Turing
using ForwardDiff
using Distributions
using Statistics
using Random
using DataFrames
using CSV
using JLD2
using CairoMakie

Random.seed!(42)

mutable struct BasalMixingModel
    n::Int
    layer::Vector{Int}
    depth::Vector{Float64}
    thickness::Vector{Float64}
    age_k81::Vector{Float64}
    c_k81::Vector{Float64}
    age_ar40::Vector{Float64}
    c_ar40::Vector{Float64}

    time::Float64
end

function BasalMixingModel(;
    depth::Union{Vector{Float64},UnitRange{Int64}} = collect(3040.0:3053.0),
    age_k81 = zeros(length(depth)),
    c_k81 = ones(length(depth)),
    age_ar40 = zeros(length(depth)),
    c_ar40 = ones(length(depth)),
    time = 0.0
    )

    n = length(depth)
    layers = 1:n
    thickness = fill(diff(depth)[1],n)  # Uniform layer thickness
    thickness = [diff(depth)..., diff(depth)[end]]
    
    return BasalMixingModel(
        n,
        collect(layers),
        collect(depth),
        collect(thickness),
        age_k81,
        c_k81,
        age_ar40,
        c_ar40,
        time
    )
end

struct BasalMixingModelSummary1
    times::Vector{Float64}      # times of snapshots
    depth::Vector{Float64}      # depth axis
    age_k81::Array{Float64}     # [nt,nd]
    c_k81::Array{Float64}       # [nt,nd]
end

function BasalMixingModelSummary1(times::Vector{Float64},depth::Vector{Float64})
    nt = length(times)
    nd = length(depth)

    age_k81 = fill(0.0, nt, nd)
    c_k81 = fill(0.0, nt, nd)

    return BasalMixingModelSummary1(
        times, depth, age_k81, c_k81
    )
end

struct BasalMixingModelSummary2
    depths::Vector{Float64}     # depths of interest
    time::Vector{Float64}       # time axis
    age_k81::Array{Float64}     # [nd,nt]
    c_k81::Array{Float64}       # [nd,nt]
end

function BasalMixingModelSummary2(depths::Vector{Float64},time::Vector{Float64})
    nd = length(depths)
    nt = length(time)
    
    age_k81 = fill(0.0, nd, nt)
    c_k81 = fill(0.0, nd, nt)

    return BasalMixingModelSummary2(
        depths, time, age_k81, c_k81
    )
end

function concentration(R0::Float64, t::Float64; t_half::Float64 = 229.0)
    return R0 * 2.0^(-t / t_half)
end

function concentration_to_age(c::Float64, c0::Float64 = 1.0; t_half::Float64 = 229.0)
    return -t_half * log2(c / c0)
end

function decay_step(R::Float64, dt::Float64; t_half::Float64 = 229.0)
    return R * 2.0^(-dt / t_half)
end

function mixing_step(R0::Float64, R1::Float64, dz0::Float64, dz1::Float64, dt::Float64, mixing_rate::Float64)
    dV = mixing_rate * dt

    @assert dV < dz0 "Mixing volume ($dV) exceeds cell 0 thickness ($dz0) — reduce dt or mixing_rate"
    @assert dV < dz1 "Mixing volume ($dV) exceeds cell 1 thickness ($dz1) — reduce dt or mixing_rate"

    R0_new = (R0 * dz0 - dV * R0 + dV * R1) / dz0
    R1_new = (R1 * dz1 - dV * R1 + dV * R0) / dz1

    #@assert R0_new*dz0 + R1_new*dz1 == R0*dz0 + R1*dz1 "Concentration is not conserved!"
    return R0_new, R1_new
end

function RunBasalMixingModel(;t0=0.0,t1=1000.0,dt=1.0,mixing_rate_clean=0.03,mixing_rate_bottom=0.03*6,t_old=250.0)

    b = BasalMixingModel()

    # Get times to model
    time = t0:dt:t1

    # Get times and depths of interest
    times = collect(500.0:100.0:3000.0)
    depths = [3045.0, 3047.0, 3050.0]

    # Define summary objects
    b1 = BasalMixingModelSummary1(times,b.depth)
    b2 = BasalMixingModelSummary2(depths,collect(time))

    mixing_rate = fill(mixing_rate_bottom,b.n-1)
    mixing_rate[1] = mixing_rate_clean

    # Set initial values
    b.c_k81 .= 1.0

    # Loop over time and advance model
    for (k, t) in enumerate(time)

        # Save previous clean ice concentration
        c_k81_clean = b.c_k81[1]

        # Update concentration from aging
        b.c_k81 .= decay_step.(b.c_k81,dt; t_half=229.0)

        # Update concentration from mixing
        for j in 1:b.n-1
            tmp0, tmp1 = mixing_step(
                b.c_k81[j], b.c_k81[j+1], 
                b.thickness[j], b.thickness[j+1], dt, 
                mixing_rate[j]
            )
            b.c_k81[j], b.c_k81[j+1] = tmp0, tmp1
        end

        # Restore clean ice age beyond t_old time
        if t > t_old
            b.c_k81[1] = c_k81_clean
        end

        # Get ages too
        b.age_k81 .= concentration_to_age.(b.c_k81,1.0)

        # Update to current time
        b.time = t

        ## Update summary objects

        # Store time slice of current variables
        if t in b1.times
            i = findall(t .== b1.times)[1]
            b1.age_k81[i,:] = b.age_k81
            b1.c_k81[i,:] = b.c_k81
        end

        # For each depth d of interest, store the value of the variables
        # at the current time
        for (i, d) in enumerate(b2.depths)
            j = findall( abs.(d .- b.depth) .< 0.4 )[1] # Index of current depth
            b2.age_k81[i,k] = b.age_k81[j]
            b2.c_k81[i,k] = b.c_k81[j]
        end

        
    end

    return b, b1, b2
end


t = 0.0:3000.0

c = concentration.(1.0, t)

b, b1, b2 = RunBasalMixingModel(;t1=3000.0,dt=0.5)



function plot_BasalMixingModelRun(b,b1,b2)

    fig = Figure(size=(600,600))

    ## PANEL 1: Depth versus closed-system age
    ax1 = Axis(fig[1,1], limits=((200,800),(-3053,-3039)), xlabel="⁸¹K closed system age (kyr)", ylabel="Depth (m)" )
    d = collect(-3052:2:-3040)
    ax1.yticks = (d,string.(abs.(d)))
    ax1.xticks = [200,400,600,800]

    for (k, t) in enumerate(b1.times)
        lines!(ax1,b1.age_k81[k,:],-b1.depth,color=:grey50,linewidth=0.5)
        if t in [500.0,1000.0,1500.0]
            lines!(ax1,b1.age_k81[k,:],-b1.depth,color=:grey50,linewidth=1.5)
        end
    end

    ## PANEL 2: Closed-system age versus time
    ax2 = Axis(fig[1,2], limits=((0,3000),(0,900)), xlabel="Time (kyr)", ylabel="⁸¹K closed system age (kyr)" )
    ax2.xticks = [0,1000,2000,3000]
    ax2.yticks = 0:100:900

    col = ["green","purple","teal"]
    for (i,d) in enumerate(b2.depths)
        lines!(ax2,b2.time,b2.age_k81[i,:],color=col[i],linewidth=2)
    end

    return fig
end

fig = plot_BasalMixingModelRun(b,b1,b2)