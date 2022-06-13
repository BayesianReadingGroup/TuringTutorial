using CSV
using DataFrames
using Turing
using MCMCChains
using Plots
using StatsPlots

df = CSV.File("coin.csv") |> DataFrame

@model function coinflip(N_heads, N_flips, N_coins, ::Type{T} = Float64) where {T <: Real}
    p = Vector{T}(undef, N_coins)
    for i in eachindex(N_flips)
    	p[i] ~ Beta(2,2)
    end

    N_heads = Vector{T}(undef, N_coins)
    for i in eachindex(N_flips)
        N_heads[i] ~ Binomial(N_flips[i], p[i])
    end

    return N_heads
end

N_flips = df[!, :Flips]
N_heads = df[!, :Heads]
N_coins = length(unique(df[!, :ID]))

md = coinflip(N_heads, N_flips, N_coins)
chain = sample(md, NUTS(0.65), MCMCThreads(), 1000, 4)

#=
chain_prior = sample(md, Prior(), 1000)

md_predict = coinflip(missing, N_flips, N_coins)
@show md_predict() # sample N_heads from prior
chain_predict = predict(md_predict, chain)
=#
