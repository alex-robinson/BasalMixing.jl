## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

using Revise

using Turing
using LinearAlgebra
using Random

using Dates
using CairoMakie
using CSV
using DataFrames

Random.seed!(42)

# Include code for basal mixing model
include("BasalMixingModel.jl")

@model function basal_mixing(age_obs, depth, dt)
    L_ref       ~ Normal(1.0, 0.1)      # 1 m
    depth_scale ~ Normal(1.0, 0.1)      # 1 m
    m_clean     ~ Normal(0.03, 0.01)    # 0.03 m/kyr
    m_dirty     ~ Normal(0.18, 0.1)     # 0.18 m/kyr
    #σ ~ Exponential(10.0)
    σ = 30.0
    t_best      ~ Uniform(500.0,3000.0)

    # Run the model
    p = (L_ref=L_ref, depth_scale=depth_scale, m_clean=m_clean, m_dirty=m_dirty)
    _, _, b2, success = RunBasalMixingModel(p; depth=depth, dt=dt, store_b1=false)
    
    if !success
        # Try one more time with a smaller timestep
        _, _, b2, success = RunBasalMixingModel(p; depth=depth, dt=dt*0.5, store_b1=false)
    end

    # Extract the best-fit age profile at the optimal time

    n_obs = length(age_obs)

    if success
        # Interpolate each depth's age time series at t_best
        age_pred = [linterp(b2.time, b2.age_k81[i, :], t_best) for i in 1:n_obs]
    else
        # Assign a very high age so that Likelihood is low
        age_pred = fill(1e8, n_obs)
    end

    # Likelihood: observed k81 ages ~ Normal(predicted, σ)
    age_obs ~ MvNormal(age_pred, σ * I)
end

## SCRIPT ##

# Load datasets for comparison
(k81, ar40) = load_basalmixing_data()
age_k81 = k81[!,"age"]
        
model = basal_mixing(age_k81, depth, 0.2)

# Start with MH to verify it works, then switch to SMC for better exploration
chain = sample(model, MH(), MCMCThreads(), 10_000, 1)  # 4 chains in parallel

# Or SMC (no chain length needed — uses particle count)
#chain = @timed sample(model, SMC(1000), 1000)

## Analysis

begin
    df = DataFrame(chain)
    params = [:L_ref, :depth_scale, :m_clean, :m_dirty, :t_best]
    labels = ["L_ref", "depth_scale", "m_clean", "m_dirty", "t_best"]
    best_idx = argmax(df.logjoint)
end

begin
    fig = Figure(size=(900, 700))

    for (i, (param, label)) in enumerate(zip(params, labels))
        row, col = divrem(i-1, 2)
        ax = Axis(fig[row+1, col+1], xlabel=label, ylabel="log p")
        scatter!(ax, df[!, param], df.logjoint, alpha=0.6, markersize=6, color=:steelblue)
        # Highlight best point in red
        scatter!(ax, [df[best_idx, param]], [df.logjoint[best_idx]], 
                color=:red, markersize=12, label="MAP")
    end

    display(fig)
end

begin
    fig = Figure(size=(900, 700))

    for (i, (param, label)) in enumerate(zip(params, labels))
        row, col = divrem(i-1, 2)
        ax = Axis(fig[row+1, col+1], xlabel=label, ylabel="count")
        hist!(ax, df[!, param], bins=20, color=(:steelblue, 0.7))
        vlines!(ax, [df[best_idx, param]], color=:red, linewidth=2, label="MAP")
    end

    display(fig)
end
