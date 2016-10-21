

# -------------------------------------------------------------

# fallbacks don't do anything
pre_hook(strat::LearningStrategy, model)      = return
iter_hook(strat::LearningStrategy, model, i)     = return
post_hook(strat::LearningStrategy, model)     = return
finished(strat::LearningStrategy, model, i)      = false
learn!(model, strat::LearningStrategy, data)  = return

# -------------------------------------------------------------

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

# This is the core iteration loop.  Iterate through data, checking for
# early stopping after each iteration.
function learn!(model, meta::MetaLearner, data)
    pre_hook(meta, model)
    for (i, item) in enumerate(data)
        for mgr in meta.managers
            learn!(model, mgr, item)
        end

        iter_hook(meta, model, i)
        finished(meta, model, i) && break
    end
    post_hook(meta, model)
end

# we can optionally learn without input data... good for minimizing functions
function learn!(model, meta::MetaLearner)
    learn!(model, meta, infinite_obs([nothing]))
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


# -------------------------------------------------------------

"A sub-strategy to stop the learning after a fixed number of iterations (maxiter)"
immutable MaxIter <: LearningStrategy
    maxiter::Int
end
MaxIter() = MaxIter(100)
finished(strat::MaxIter, model, i) = i >= strat.maxiter

# -------------------------------------------------------------

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

# -------------------------------------------------------------

"Print out a summary of the current learning"
type ShowStatus <: LearningStrategy
    every::Int
    f::Function
end
ShowStatus(every::Int = 1) = ShowStatus(every, (model, i) -> "Iteration $i: $(params(model))")

pre_hook(strat::ShowStatus, model) = iter_hook(strat, model, 0)
function iter_hook(strat::ShowStatus, model, i)
    if mod1(i, strat.every) == strat.every
        println(strat.f(model, i))
    end
    return
end

# -------------------------------------------------------------

"A sub-strategy to stop learning when the associated function returns true."
immutable ConvergenceFunction <: LearningStrategy
    f::Function
end
finished(strat::ConvergenceFunction, model, i) = strat.f(model, i)

# -------------------------------------------------------------

"Finished when `‖f(model) - lastf‖ ≦ tol`"
type Converged <: LearningStrategy
    f::Function   # f(model)
    tol::Float64  # normdiff tolerance
    every::Int    # only check every ith iteration
    lastval::Vector{Float64}
    Converged(f::Function; tol::Number = 1e-6, every::Int = 1) = new(f, tol, every)
end
pre_hook(strat::Converged, model) = (strat.lastval = zeros(strat.f(model)); return)
function finished(strat::Converged, model, i)
    val = strat.f(model)
    if norm(val - strat.lastval) <= strat.tol
        info("Converged after $i iterations: $val")
        true
    else
        copy!(strat.lastval, val)
        false
    end
end

# -------------------------------------------------------------

"Finished when `‖f(model) - goal‖ ≦ tol`"
type ConvergedTo{V} <: LearningStrategy
    f::Function   # f(model)
    tol::Float64  # normdiff tolerance
    goal::V       # goal value
    every::Int    # only check every ith iteration
end
function ConvergedTo(f::Function, goal; tol::Number = 1e-6, every::Int = 1)
    ConvergedTo(f, tol, goal, every)
end
function finished(strat::ConvergedTo, model, i)
    val = strat.f(model)
    if norm(val - strat.goal) <= strat.tol
        info("Converged after $i iterations: $val")
        true
    else
        false
    end
end

# -------------------------------------------------------------


"A sub-strategy to do something each iteration."
immutable IterFunction <: LearningStrategy
    f::Function
    every::Int
end
IterFunction(f::Function; every::Int = 1) = IterFunction(f, every)
function iter_hook(strat::IterFunction, model, i)
    if mod1(i, strat.every) == strat.every
        strat.f(model, i)
    end
    return
end

# -------------------------------------------------------------

"Store something every ith iteration"
type Tracer{S} <: LearningStrategy
    every::Int
    f::Function
    storage::Vector{S}
end
Tracer{S}(::Type{S}, f::Function, every::Int = 1) = Tracer(every, f, S[])
function iter_hook(strat::Tracer, model, i)
    if mod1(i, strat.every) == strat.every
        push!(strat.storage, strat.f(model, i))
    end
    return
end


# -------------------------------------------------------------

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
