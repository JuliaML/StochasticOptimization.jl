
# TODO: add citations/links for each method

"Stochastic Gradient Descent with Momentum"
type SGD{T<:Number} <: ParamUpdater
    mom::T # momentum
    last_Δ::Vector{T}
    SGD(mom::T) = new(mom)
end
SGD{T}(::Type{T}, mom = T(0.0)) = SGD{T}(T(mom))
SGD{T}(mom::T = 0.0) = SGD{T}(mom)

function init(updater::SGD, model)
    updater.last_Δ = zeros(params(model))
    return
end

function update!(θ::AbstractVector, updater::SGD, ∇::AbstractVector, lr::Number)
    for (i,j,k) in zip(eachindex(θ), eachindex(∇), eachindex(updater.last_Δ))
        chg = -lr * ∇[j] + updater.mom * updater.last_Δ[k]
        θ[i] += chg
        updater.last_Δ[k] = chg
    end
end

# -----------------------------------------------------------------------

"Adaptive Gradient"
type Adagrad{T<:Number} <: ParamUpdater
    ϵ::T
    G::Vector{T}  # sum of squared gradients
    Adagrad(ϵ::T) = new(ϵ)
end
Adagrad{T}(::Type{T}, ϵ = T(0.01)) = Adagrad{T}(T(ϵ))
Adagrad{T}(ϵ::T = 0.01) = Adagrad{T}(ϵ)

function init(updater::Adagrad, model)
    updater.G = zeros(params(model))
    return
end

function update!(θ::AbstractVector, updater::Adagrad, ∇::AbstractVector, lr::Number)
    for (i,j,k) in zip(eachindex(θ), eachindex(∇), eachindex(updater.G))
        updater.G[k] += ∇[j]^2
        Θ[i] -= lr * ∇[j] / sqrt(updater.ϵ + updater.G[k])
    end
end

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
