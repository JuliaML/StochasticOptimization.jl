
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
        θ[i] -= lr * ∇[j] / sqrt(updater.ϵ + updater.G[k])
    end
end

# -----------------------------------------------------------------------

"""
See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

Relatively parameter-free... can probably avoid changing ε and ρ
"""
type Adadelta{T<:Number} <: ParamUpdater
    ϵ::T
    ρ::T
    dmean::Vector{T} # exponential average of squared param changes
    Gmean::Vector{T} # exponential average of squared gradients
    Adadelta(ϵ::T, ρ::T) = new(ϵ, ρ)
end
Adadelta{T}(::Type{T}, ϵ = T(0.01), ρ = T(0.97)) = Adadelta{T}(T(ϵ), T(ρ))
Adadelta{T}(ϵ::T = 0.01, ρ::T = 0.97) = Adadelta{T}(ϵ, ρ)

function init(updater::Adadelta, model)
    updater.dmean = zeros(params(model))
    updater.Gmean = zeros(params(model))
    return
end

function update!{T}(θ::AbstractVector, updater::Adadelta{T}, ∇::AbstractVector, lr::Number)
    ϵ = updater.ϵ
    ρ = updater.ρ
    dm = updater.dmean
    Gm = updater.Gmean
    for i=1:length(θ)
        # average g²
        Gm[i] = ρ * Gm[i] + (one(T) - ρ) * ∇[i]^2

        # learning rate is function of prev avg dw² and current avg g²
        δ = lr * sqrt((dm[i] + ϵ) / (Gm[i] + ϵ)) * ∇[i]
        θ[i] -= δ

        # update avg dw²
        dm[i] = ρ * dm[i] + (one(T) - ρ) * δ^2
    end
end

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
