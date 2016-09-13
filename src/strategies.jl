
# fallbacks don't do anything
pre_hook(strat::LearningStrategy, model)      = return
iter_hook(strat::LearningStrategy, model, i::Int)     = return
post_hook(strat::LearningStrategy, model)     = return
finished(strat::LearningStrategy, model)      = false
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

pre_hook(master::MasterLearner,  model) = foreach(mgr -> pre_hook(mgr, model),  master.managers)
iter_hook(master::MasterLearner, model, i::Int) = foreach(mgr -> iter_hook(mgr, model, i), master.managers)
post_hook(master::MasterLearner, model) = foreach(mgr -> post_hook(mgr, model), master.managers)
finished(master::MasterLearner,  model) = any(mgr     -> finished(mgr, model),  master.managers)

function learn!(model, master::MasterLearner, data::AbstractSubset)
    for mgr in master.managers
        learn!(model, mgr, data)
    end
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

@with_kw type MaxIter <: LearningStrategy
    niter::Int = 1
    maxiter::Int = 100
end
MaxIter(maxiter::Int) = MaxIter(1,maxiter)

iter_hook(strat::MaxIter, model, i::Int) = (strat.niter += 1)
finished(strat::MaxIter, model) = strat.niter > strat.maxiter

# ---------------------------------------------------------------------------------

# loop through batches checking for early stopping after each subset
function learn!(model, strat::LearningStrategy, subsets::AbstractSubsets)
    pre_hook(strat, model)
    for (i, subset) in enumerate(subsets)
        # update the params for this subset
        learn!(model, strat, subset)

        iter_hook(strat, model, i)
        finished(strat, model) && break
    end
    post_hook(strat, model)
end

# ---------------------------------------------------------------------------------

immutable GradientDescent{LR <: LearningRate, PU <: ParamUpdater} <: LearningStrategy
    lr::LR
    updater::PU
end
GradientDescent(lr::LearningRate = FixedLR(1e-3), updater::ParamUpdater = SGD()) = GradientDescent(lr, updater)
GradientDescent(updater::ParamUpdater, lr::LearningRate = FixedLR(1e-3)) = GradientDescent(lr, updater)

pre_hook(strat::GradientDescent, model) = init(strat.updater, model)

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
