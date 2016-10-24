"""
Each learn step adds to a running sum of gradients, and doesn't update θ
until `k` samples have been collected, at which point it updates with the
average gradient over those `k` samples.
"""
type OnlineGradAvg{PU <: ParamUpdater} <: LearningStrategy
    lr::Float64
    pu::PU
    idx::Int # current step
    k::Int # number of steps to average
    ∇avg::Vector{Float64}
    # OnlineGradAvg(lr::Float64, pu::PU, k::Int) = new(lr, pu, 1, k)
end

function OnlineGradAvg(k::Int; lr::Number = 1e-2, pu::ParamUpdater = RMSProp())
    OnlineGradAvg(lr, pu, 1, k, zeros(0))
end

# initialize
@with ga function pre_hook(ga::OnlineGradAvg, model::Learnable)
    idx = 1
    init(pu, model)
    ∇avg = zeros(length(grad(model)))
end

# initialize
@with ga function pre_hook(ga::OnlineGradAvg, θ::AbstractVector)
    idx = 1
    init(pu, θ)
    ∇avg = zeros(θ)
end

function learn!(model::Learnable, ga::OnlineGradAvg)
    learn!(params(model), ga, grad(model))
end

# one iteration update... we assume θ/∇ are pre-populated
@with ga function learn!(θ, ga::OnlineGradAvg, ∇)
    # add the ∇
    @simd for i in 1:length(∇)
        @inbounds ∇avg[i] += ∇[i]
    end

    # is it time to update θ?
    if idx >= k

        # convert sums to means
        scalar = 1 / k
        for i in eachindex(∇avg)
            @inbounds ∇avg[i] *= scalar
        end

        # update the params using the search direction
        update!(θ, pu, ∇avg, lr)

        # reset
        fill!(∇avg, 0.0)
        idx = 1
    else
        idx += 1
    end
end
