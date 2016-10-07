

# ---------------------------------------------------------------------------------

# fallbacks don't do anything
pre_hook(strat::LearningStrategy, model)      = return
iter_hook(strat::LearningStrategy, model, i)     = return
post_hook(strat::LearningStrategy, model)     = return
finished(strat::LearningStrategy, model, i)      = false
learn!(model, strat::LearningStrategy, data)  = return

# ---------------------------------------------------------------------------------

# Meta-learner that can compose sub-managers of optimization components in a type-stable way.
# A sub-manager is any LearningStrategy, and may implement any subset of callbacks.
type MetaLearner{MGRS <: Tuple} <: LearningStrategy
    managers::MGRS
end

function MetaLearner(mgrs::LearningStrategy...)
    MetaLearner(mgrs)
end

pre_hook(meta::MetaLearner,  model)    = foreach(mgr -> pre_hook(mgr, model),      meta.managers)
iter_hook(meta::MetaLearner, model, i) = foreach(mgr -> iter_hook(mgr, model, i),  meta.managers)
finished(meta::MetaLearner,  model, i) = any(mgr     -> finished(mgr, model, i),   meta.managers)
post_hook(meta::MetaLearner, model)    = foreach(mgr -> post_hook(mgr, model),     meta.managers)

# This is the core iteration loop.  Loop through batches checking for early stopping after each subset
function learn!(model, meta::MetaLearner, data)
    pre_hook(meta, model)
    for (i, obs) in enumerate(data)
        # update the params for this subset
        # learn!(model, meta, subset)
        for mgr in meta.managers
            learn!(model, mgr, obs)
        end

        iter_hook(meta, model, i)
        finished(meta, model, i) && break
    end
    post_hook(meta, model)
end

# TODO: can we instead use generated functions for each MetaLearner callback so that they are ONLY called for
#   those methods which the manager explicitly implements??  We'd need to have a type-stable way
#   of checking whether that manager implements that method.

# @generated function pre_hook(meta::MetaLearner, model)
#     body = quote end
#     mgr_types = meta.parameters[1]
#     for (i,T) in enumerate(mgr_types)
#         if is_implemented(T, :pre_hook)
#             push!(body.args, :(pre_hook(meta.managers[$i], model)))
#         end
#     end
#     body
# end


# ---------------------------------------------------------------------------------

"A sub-strategy to stop the learning after a fixed number of iterations (maxiter)"
immutable MaxIter <: LearningStrategy
    maxiter::Int
end
MaxIter() = MaxIter(100)
finished(strat::MaxIter, model, i) = i >= strat.maxiter

# ---------------------------------------------------------------------------------

"Stop iterating after a pre-determined amount of time."
type TimeLimit <: LearningStrategy
    secs::Float64
    secs_end::Float64
    TimeLimit(secs::Number) = new(secs)
end
pre_hook(strat::TimeLimit, model) = (strat.secs_end = time() + strat.secs)
function finished(strat::TimeLimit, model, i)
    stop = time() >= strat.secs_end
    if stop
        info("Time's up!")
    end
    stop
end

# ---------------------------------------------------------------------------------

"A sub-strategy to stop learning when the associated function returns true."
immutable ConvergenceFunction{F<:Function} <: LearningStrategy
    f::F
end
finished(strat::ConvergenceFunction, model, i) = strat.f(model, i)

# ---------------------------------------------------------------------------------


"A sub-strategy to do something each iteration."
immutable IterFunction{F<:Function} <: LearningStrategy
    f::F
end
iter_hook(strat::IterFunction, model, i) = strat.f(model, i)

# ---------------------------------------------------------------------------------

function make_learner(args...; kw...)
    strats = []
    for (k,v) in kw
        if k == :maxiter
            push!(strats, MaxIter(v))
        elseif k == :oniter
            push!(strats, IterFunction(v))
        elseif k == :converged
            push!(strats, ConvergenceFunction(v))
        end
    end
    MetaLearner(args..., strats...)
end

# add to an existing meta
function make_learner(meta::MetaLearner, args...; kw...)
    make_learner(meta.managers..., args...; kw...)
end

# ---------------------------------------------------------------------------------

abstract StateUpdater

immutable NoUpdater <: StateUpdater end
state!(model, su::NoUpdater, obs...) = return

immutable BackpropUpdater <: StateUpdater end
function state!(model, su::BackpropUpdater, target, input)
    transform!(model, target, input)
    grad!(model)
    return
end

# ---------------------------------------------------------------------------------

"""
A sub-learner which can update model parameters using a search direction (which might be an estimate
    of the gradient), with LearningRate lr and ParamUpdater pu (SGD, Adam, etc).

The GradientLearner will update its internal state using the StateUpdater.
"""
immutable GradientLearner{LR <: LearningRate, PU <: ParamUpdater, SU <: StateUpdater} <: LearningStrategy
    lr::LR
    pu::PU
    su::SU
end

function GradientLearner(lr::LearningRate = FixedLR(1e-1),
                         pu::ParamUpdater = RMSProp(),
                         su::StateUpdater = BackpropUpdater())
    GradientLearner(lr, pu, su)
end
function GradientLearner(pu::ParamUpdater,
                         lr::LearningRate = FixedLR(1e-3),
                         su::StateUpdater = BackpropUpdater())
    GradientLearner(lr, pu, su)
end
function GradientLearner(lr::Number,
                         pu::ParamUpdater = RMSProp(),
                         su::StateUpdater = BackpropUpdater())
    GradientLearner(FixedLR(lr), pu, su)
end

pre_hook(gl::GradientLearner, model) = init(gl.pu, model)

# minibatch learning.  update with average gradient
function learn!(model, gl::GradientLearner, subset::AbstractSubset)
    θ = params(model)
    ∇ = grad(model)
    before_grad_calc(θ, gl.pu, ∇)

    ∇avg = zeros(θ)
    scalar = 1 / nobs(subset)
    for (input,target) in subset
        state!(model, gl.su, target, input)

        # add to the total param change for this gl/gradient
        for i in 1:length(∇)
            ∇avg[i] += ∇[i] * scalar
        end
    end

    # update the params using the average gradient
    lr = value(gl.lr)
    update!(θ, gl.pu, ∇avg, lr)
end

# stochastic learning.  update with a single gradient
function learn!(model, gl::GradientLearner, obs::Tuple)
    input, target = obs

    # forward and backward passes for this datapoint
    θ = params(model)
    ∇ = grad(model)
    before_grad_calc(θ, gl.pu, ∇)

    state!(model, gl.su, target, input)

    # update the params using the gradient
    lr = value(gl.lr)
    update!(θ, gl.pu, ∇, lr)
end
