__precompile__(true)

module StochasticOptimization

using Reexport
@reexport using LearnBase
@reexport using MLDataUtils
using Parameters


import LearnBase: learn!, update!
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

"Enacts a strategy to adjust the learning rate"
abstract LearningRate

# ---------------------------------------------------------------------------------

# NOTES:
#   - a strategy holds an approach and the state


function learn!(t::Minimizable, strat::LearningStrategy, data::DataIterator)
    # an available callback
    pre_hook(strat, t)

    dstate = start(data)
    while !done(data, dstate) && !finished(strat, t)
        # update the transformation with the next data point
        (input, target), dstate = next(data, dstate)
        transform!(t, target, input)
        grad!(t)

        # update the parameters and state
        update!(strat, t)

        # an available callback
        iter_hook(strat, t)
    end

    # an available callback
    post_hook(strat, t)
    return
end

# fallbacks don't do anything
pre_hook(strat, t) = return
iter_hook(strat, t) = return
post_hook(strat, t) = return


# ---------------------------------------------------------------------------------

# TODO: split into composable strategies... something like pub/sub maybe?
# A "CEO" should hold the Minimizable that we're learning as well as all the strategies that apply

@with_kw type SGD{T, LR <: LearningRate} <: LearningStrategy
    lr::LR = FixedLR(1e-2)
    mom::T = T(0.5) # momentum
    niter::Int = 0
    maxiter::Int = 100
    last_Δ::Vector{T}
end
SGD{T}(::Type{T}, n::Int) = SGD{T}(last_Δ = zeros(T,n))

function update!(strat::SGD, t::Minimizable)
    θ = params(t)
    ∇ = grad(t)
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
