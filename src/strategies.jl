

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
type MasterLearner{MGRS <: Tuple} <: LearningStrategy
    managers::MGRS
end

function MasterLearner(mgrs::LearningStrategy...)
    MasterLearner(mgrs)
end

pre_hook(master::MasterLearner,  model)    = foreach(mgr -> pre_hook(mgr, model),      master.managers)
iter_hook(master::MasterLearner, model, i) = foreach(mgr -> iter_hook(mgr, model, i),  master.managers)
finished(master::MasterLearner,  model, i) = any(mgr     -> finished(mgr, model, i),   master.managers)
post_hook(master::MasterLearner, model)    = foreach(mgr -> post_hook(mgr, model),     master.managers)

# This is the core iteration loop.  Loop through batches checking for early stopping after each subset
function learn!(model, master::MasterLearner, data)
    pre_hook(master, model)
    for (i, obs) in enumerate(data)
        # update the params for this subset
        # learn!(model, master, subset)
        for mgr in master.managers
            learn!(model, mgr, obs)
        end

        iter_hook(master, model, i)
        finished(master, model, i) && break
    end
    post_hook(master, model)
end

# TODO: can we instead use generated functions for each MasterLearner callback so that they are ONLY called for
#   those methods which the manager explicitly implements??  We'd need to have a type-stable way
#   of checking whether that manager implements that method.

# @generated function pre_hook(master::MasterLearner, model)
#     body = quote end
#     mgr_types = master.parameters[1]
#     for (i,T) in enumerate(mgr_types)
#         if is_implemented(T, :pre_hook)
#             push!(body.args, :(pre_hook(master.managers[$i], model)))
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
    MasterLearner(args..., strats...)
end

# add to an existing master
function make_learner(master::MasterLearner, args...; kw...)
    make_learner(master.managers..., args...; kw...)
end

# ---------------------------------------------------------------------------------

"""
A Stochastic Gradient Descent learner, with LearningRate lr and ParamUpdater updater (SGD, Adam, etc)
"""
immutable GradientDescent{LR <: LearningRate, PU <: ParamUpdater} <: LearningStrategy
    lr::LR
    updater::PU
end
GradientDescent(lr::LearningRate = FixedLR(1e-1), updater::ParamUpdater = RMSProp()) = GradientDescent(lr, updater)
GradientDescent(updater::ParamUpdater, lr::LearningRate = FixedLR(1e-3)) = GradientDescent(lr, updater)
GradientDescent(lr::Number, updater::ParamUpdater = RMSProp()) = GradientDescent(FixedLR(lr), updater)

pre_hook(strat::GradientDescent, model) = init(strat.updater, model)

# minibatch learning.  update with average gradient
function learn!(model, strat::GradientDescent, subset::AbstractSubset)
    θ = params(model)
    ∇ = grad(model)
    ∇avg = zeros(θ)
    scalar = 1 / length(subset)
    for (input,target) in subset
        # forward and backward passes for this datapoint
        transform!(model, target, input)
        grad!(model)

        # add to the total param change for this strat/gradient
        for (i,j) in zip(eachindex(∇avg), eachindex(∇))
            ∇avg[i] += ∇[j] * scalar
        end
    end

    # update the params using the average gradient
    lr = value(strat.lr)
    update!(θ, strat.updater, ∇avg, lr)
end

# stochastic learning.  update with a single gradient
function learn!(model, strat::GradientDescent, obs::Tuple)
    input, target = obs

    # forward and backward passes for this datapoint
    transform!(model, target, input)
    grad!(model)

    # update the params using the gradient
    lr = value(strat.lr)
    update!(params(model), strat.updater, grad(model), lr)
end
