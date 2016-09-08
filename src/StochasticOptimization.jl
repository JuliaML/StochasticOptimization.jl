__precompile__(true)

module StochasticOptimization

using Reexport
@reexport using LearnBase
@reexport using ObjectiveFunctions
@reexport using MLDataUtils
using Parameters


import LearnBase: value, learn!, update!
# import IterationManagers
# const IM = IterationManagers
# import OnlineStats: Diff, Mean, Variance, fit!, ExponentialWeight


export
    LearningStrategy,
    SGD,

    LearningRate,
    FixedLR,
    AdaptiveLR

"Holds optimizer state and parameters"
abstract LearningStrategy
abstract GradientDescent <: LearningStrategy

"Enacts a strategy to adjust the learning rate"
abstract LearningRate

# ---------------------------------------------------------------------------------

# NOTES:
#   - a strategy holds an approach and the state


# loop through batches checking for early stopping after each batch
function learn!(model, strategy, data::BatchIterator)
    pre_hook(strategy, model)
    for batch in data
        # update the params for this batch
        learn!(model, strategy, batch)

        iter_hook(strategy, model)
        finished(strategy, model) && break
    end
    post_hook(strategy, model)
end

# TODO: split into composable strategies... something like pub/sub maybe?
# A "CEO" should hold the Minimizable that we're learning as well as all the strategies that apply


# function learn!(t::Minimizable, strat::LearningStrategy, data::DataIterator)
#     # an available callback
#     pre_hook(strat, t)
#
#     dstate = start(data)
#     while !done(data, dstate) && !finished(strat, t)
#         # update the transformation with the next data point
#         (input, target), dstate = next(data, dstate)
#         transform!(t, target, input)
#         grad!(t)
#
#         # update the parameters and state
#         update!(strat, t)
#
#         # an available callback
#         iter_hook(strat, t)
#     end
#
#     # an available callback
#     post_hook(strat, t)
#     return
# end

# fallbacks don't do anything
pre_hook(strat, t) = return
iter_hook(strat, t) = return
post_hook(strat, t) = return


# ---------------------------------------------------------------------------------

function learn!(model, strategy::GradientDescent, batch::Batch)
    θ = params(model)
    ∇ = grad(model)
    ∇avg = zeros(θ)
    scalar = 1 / length(batch)
    for (input,target) in batch
        # forward and backward passes for this datapoint
        transform!(model, target, input)
        grad!(model)

        # add to the total param change for this strategy/gradient
        for (i,j) in zip(eachindex(∇avg), eachindex(∇))
            ∇avg[i] += ∇[j] * scalar
        end
    end

    # update the params using the average gradient
    update!(θ, strategy, ∇avg)
end


# ---------------------------------------------------------------------------------

# @with_kw type SGD{T, LR <: LearningRate} <: GradientDescent
#     lr::LR = FixedLR(1e-2)
#     mom::T = T(0.5) # momentum
#     niter::Int = 0
#     maxiter::Int = 100
#     n::Int
#     last_Δ::Vector{T} = zeros(T,n)
# end
# # SGD{T}(::Type{T}, n::Int) = SGD{T}(last_Δ = zeros(T,n))
# SGD{T}(::Type{T}, n::Int) = SGD{T}(n=n)
# SGD(n::Int) = SGD(Float64, n)

type SGD{T, LR <: LearningRate} <: GradientDescent
    lr::LR
    mom::T # momentum
    niter::Int
    maxiter::Int
    last_Δ::Vector{T}
end
# SGD{T}(::Type{T}, n::Int) = SGD{T}(last_Δ = zeros(T,n))
function SGD{T}(::Type{T}, n::Int;
                lr = FixedLR(1e-2),
                mom = T(0.5),
                maxiter = 100)
    SGD{T,typeof(lr)}(lr, mom, 0, maxiter, zeros(T,n))
end
SGD(n::Int; kw...) = SGD(Float64, n; kw...)

function update!(θ::AbstractVector, strat::SGD, ∇::AbstractVector)
    η = value(strat.lr)
    for (i,j,k) in zip(eachindex(θ), eachindex(∇), eachindex(strat.last_Δ))
        chg = -η * ∇[j] + strat.mom * strat.last_Δ[k]
        θ[i] += chg
        strat.last_Δ[k] = chg
    end
end

# TODO: something worthwhile
function finished(strat::SGD, t)
    strat.niter += 1
    strat.niter >= strat.maxiter
end

# ---------------------------------------------------------------------------------



include("learningrates.jl")

# ---------------------------------------------------------------------------------

# type SGDManager <: IM.IterationManager end

# @with_kw type SGDState{LR <: LearningRate, T <: Number} <: IM.IterationState
#     lr::LR = FixedLR(1e-2)
#     mom::T = T(0.5) # momentum
#     x::Vector{T}
#     ∇::Vector{T} = zeros(T, n)
# end

# ---------------------------------------------------------------------------------

# "Stochastic Gradient Descent with Momentum"
# type SGDUpdater <: ParameterUpdater
#     η::Float64 # learning rate
#     μ::Float64 # momentum
# end
# SGDUpdater(; η=0.1, μ=0.5) = SGDUpdater(η, μ)

# type SGDState <: ParameterUpdaterState
#     lastChange::Float64
# end
# SGDState() = SGDState(0.0)

# function param_change!(state::SGDState, updater::SGDUpdater, gradient::Real)
#     state.lastChange = -updater.η * gradient + updater.μ * state.lastChange
# end




# ---------------------------------------------------------------------------------


# # TODO:
# #   1) better convergence testing

# function learn!(
#         learner::AbstractLearner,
#         state::AbstractLearnerState;
#         iterations=1000,
#         ftol::Float64=1e-6
#     )
#     x = paramvec(model) # returns view of all parameters as a vector
#     ∇ = gradvec(model)  # returns view of gradient as a vector
#     f = value(model)
#     fhist = [f]

#     converged = false
#     iter = 0
#     while !converged && iter < iterations
#         iter += 1
#         f = update!(x,∇,f,model,opt)    # uodates x
#         grad!(model)                    # updates ∇
#         converged = (fhist[end]-f)/fhist[end] < ftol
#         push!(fhist,f)
#     end

#     return fhist,converged
# end


# include("learn.jl")
# include("grad_descent.jl")
# # include("stochastic_methods.jl")

end # module
