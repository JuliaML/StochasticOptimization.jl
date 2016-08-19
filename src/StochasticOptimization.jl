module StochasticOptimization

using Reexport
@reexport using LearnBase
import LearnBase: update!
using Parameters
import IterationManagers
const IM = IterationManagers
import OnlineStats: Diff, Mean, Variance, fit!, ExponentialWeight

# export
#     AbstractLearner,
#     AbstractLearnerState

# abstract AbstractLearner
# abstract AbstractLearnerState


# TODO: implement this:

# function managed_iteration!{T<:AbstractArray}(f!::Base.Callable,
#                                               mgr::IterationManager,
#                                               dest::T,
#                                               istate::IterationState{T};
#                                               by::Base.Callable=default_by)
#     pre_hook(mgr, istate)

#     while !(finished(mgr, istate))
#         f!(dest, istate.prev)
#         update!(istate, dest; by=by)
#         iter_hook(mgr, istate)
#     end

#     post_hook(mgr, istate)
#     istate
# end

# ---------------------------------------------------------------------------------

# abstract ConvergenceCheck
# IM.finished(check::ConvergenceCheck, state) = error("You should implement a finished method for this convergence check: $check $state")

# ""
# type ErrorCheck <: ConvergenceCheck
#     minerr::Float64
#     lasterr::Float64
# end
# function IM.finished(check::ErrorCheck, err::Float64)
#     diff = err - lasterr
#     check.lasterr = err
# end

# ---------------------------------------------------------------------------------

export
    AbstractLearningRate,
    FixedLR,
    AdaptiveLR

"Enacts a strategy to adjust the learning rate"
abstract AbstractLearningRate

@with_kw immutable FixedLR <: AbstractLearningRate
    lr::Float64 = 1e-2
end
update!(lr::FixedLR, err) = lr


"Adapts learning rate based on relative variance of the changes in the test error"
@with_kw type AdaptiveLR <: AbstractLearningRate
    lr::Float64 = 1e-2
    ε::Diff{Float64} = Diff()
    lookback::Int = 20
    σ²::Variance{BoundedEqualWeight} = Variance(BoundedEqualWeight(lookback))
    adjpct::Float64 = 1e-2
    cutoff::Float64 = 1e-1
end


# if the error is decreasing at a large rate relative to the variance, increase the learning rate (speed it up)
function update!(lr::AdaptiveLR, err)
    fit!(lr.ε, err)
    fit!(lr.σ², diff(lr.ε))
    μ = mean(lr.σ²)
    σ = std(lr.σ²)
    if σ > 0
        pct = lr.adjpct * (μ / σ < -lr.cutoff ? 1.0 : -1.0)
        lr.lr *= (1.0 + pct)
    end
    lr
end


# ---------------------------------------------------------------------------------

# type SGDManager <: IM.IterationManager end

@with_kw type SGDState{LR <: AbstractLearningRate, T <: Number} <: IM.IterationState
    lr::LR = FixedLR(1e-2)
    mom::T = 0.5 # momentum
    x::Vector{T}
    ∇::Vector{T} = zeros(T, n)
end

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
