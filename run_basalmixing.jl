## Preamble #############################################
cd(@__DIR__)
import Pkg; Pkg.activate(".")
#########################################################

using Revise
using CairoMakie

# Include code for basal mixing model
include("BasalMixingModel.jl")

# Test calculating concentration over time (half-life decay)
t = 0.0:3000.0
c = concentration.(1.0, t)

# Run the basal mixing model
# Return current state, summary1 and summary2

b, b1, b2 = RunBasalMixingModel(;t1=3000.0,dt=0.5)

# Plot the results

fig = plot_BasalMixingModelRun(b,b1,b2)