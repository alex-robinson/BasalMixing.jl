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

Random.seed!(42)
