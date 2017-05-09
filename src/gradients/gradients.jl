
"""
An abstraction that knows how to update a model and compute a search
direction (gradient estimate).
"""
@compat abstract type SearchDirection <: LearningStrategy end

type GradientAverager <: SearchDirection
    ∇avg::Vector{Float64}
    GradientAverager() = new()
end

function init(ga::GradientAverager, model)
    ga.∇avg = zeros(length(grad(model)))
end

# for a single observation, just return ∇
function search_direction(model, ga::GradientAverager, obs)
    update!(model, obs)
    grad(model)
end

# for a minibatch, compute the average gradient
function search_direction(model, ga::GradientAverager, batch::AbstractObsIterator)
    fill!(ga.∇avg, 0.0)
    scalar = 1 / nobs(batch)
    ∇ = grad(model)
    for obs in batch
        update!(model, obs)

        # add to the total param change for this gl/gradient
        @simd for i in 1:length(∇)
            @inbounds ga.∇avg[i] += ∇[i] * scalar
        end
    end
    ga.∇avg
end

# -------------------------------------------------------------

"""
A sub-learner which can update model parameters using a search direction (which might be an estimate
    of the gradient), with LearningRate lr and ParamUpdater pu (SGD, Adam, etc).

Note: we might update the model's internal state while computing the SearchDirection.
"""
immutable GradientLearner{LR <: LearningRate, PU <: ParamUpdater, SD <: SearchDirection} <: LearningStrategy
    lr::LR
    pu::PU
    sd::SD
end

function GradientLearner(lr::LearningRate = FixedLR(1e-1),
                         pu::ParamUpdater = RMSProp())
    GradientLearner(lr, pu, GradientAverager())
end
# function GradientLearner(pu::ParamUpdater,
#                          lr::LearningRate = FixedLR(1e-3),
#                          sd::SearchDirection = GradientAverager())
#     GradientLearner(lr, pu, sd)
# end
function GradientLearner(lr::Number,
                         pu::ParamUpdater = RMSProp(),
                         sd::SearchDirection = GradientAverager())
    GradientLearner(FixedLR(lr), pu, sd)
end

function pre_hook(gl::GradientLearner, model)
    init(gl.pu, model)
    init(gl.sd, model)
end

# one iteration update
function learn!(model, gl::GradientLearner, data)
    θ = params(model)
    ∇ = grad(model)

    # # optional setup before iteration update
    # before_grad_calc(θ, gl.pu, ∇)

    # get this iterations search direction (gradient estimate)
    sd = search_direction(model, gl.sd, data)

    # update the params using the search direction
    update!(θ, gl.pu, sd, value(gl.lr))
end
