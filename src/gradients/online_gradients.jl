"""
Each learn step adds to a running sum of gradients, and doesn't update θ
until `k` samples have been collected, at which point it updates with the
average gradient over those `k` samples.
"""
type OnlineGradAvg{PU <: ParamUpdater} <: LearningStrategy
    lr::Float64
    pu::PU
    i::Int # current step
    k::Int # number of steps to average
    ∇avg::Vector{Float64}
    # OnlineGradAvg(lr::Float64, pu::PU, k::Int) = new(lr, pu, 1, k)
end

function OnlineGradAvg(k::Int; lr::Number = 1e-2, pu::ParamUpdater = RMSProp())
    OnlineGradAvg(lr, pu, 1, k, zeros(0))
end

# initialize
function pre_hook(ga::OnlineGradAvg, model)
    ga.i = 1
    init(ga.pu, model)
    ga.∇avg = zeros(length(grad(model)))
end

# one iteration update... we assume θ/∇ are pre-populated
function learn!(model, ga::OnlineGradAvg, unused)
    θ = params(model)
    ∇ = grad(model)
    ∇avg = ga.∇avg

    # add the ∇
    @simd for i in 1:length(∇)
        @inbounds ∇avg[i] += ∇[i]
    end

    # is it time to update θ?
    if ga.i >= ga.k

        # convert sums to means
        scalar = 1 / ga.k
        for i in eachindex(∇avg)
            @inbounds ∇avg[i] *= scalar
        end

        # update the params using the search direction
        update!(θ, ga.pu, ∇avg, ga.lr)

        # reset
        fill!(∇avg, 0.0)
        ga.i = 1
    else
        ga.i += 1
    end
end
