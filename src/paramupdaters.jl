
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

"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

Tracks an exponential moving average of the first and second moments of the gradient,
adjusting for zero-bias.  The defaults are those suggested in the paper.

TODO: AdaMax is similar, using the p-norm as p -> ∞
"""
type Adam{T<:Number} <: ParamUpdater
    ϵ::T  # small number so we don't divide by 0
    ρ1::T # decay for first moment (β₁ in the paper)
    ρ2::T # decay for second moment (β₂ in the paper)
    m::Vector{T} # average first moment
    v::Vector{T} # average second moment
    ρ1t::Vector{T} # β₁ᵗ from the paper... t-th power of β₁
    ρ2t::Vector{T} # β₂ᵗ from the paper... t-th power of β₂
    Adam(ϵ::T, ρ1::T, ρ2::T) = new(ϵ, ρ1, ρ2)
end
Adam{T}(::Type{T}, ϵ = T(1e-8), ρ1 = T(0.9), ρ2 = T(0.999)) = Adam{T}(T(ϵ), T(ρ1), T(ρ2))
Adam{T}(ϵ::T = 1e-8, ρ1::T = 0.9, ρ2::T = 0.999) = Adam{T}(ϵ, ρ1, ρ2)

function init(updater::Adam, model)
    n = params(model)
    updater.m = zeros(n)
    updater.v = zeros(n)
    updater.ρ1t = ones(n)
    updater.ρ2t = ones(n)
    return
end

function update!{T}(θ::AbstractVector, updater::Adam{T}, ∇::AbstractVector, lr::Number)
    ϵ = updater.ϵ
    ρ1 = updater.ρ1
    ρ2 = updater.ρ2
    m = updater.m
    v = updater.v
    ρ1t = updater.ρ1t
    ρ2t = updater.ρ2t
    for i=1:length(θ)
        m[i] = ρ1 * m[i] + (one(T) - ρ1) * ∇[i]
        v[i] = ρ2 * v[i] + (one(T) - ρ2) * ∇[i]^2
        ρ1t[i] *= ρ1
        ρ2t[i] *= ρ2
        θ[i] -= lr * m[i] * sqrt((one(T) - ρ2t[i]) / (v[i] + ϵ)) / (one(T) - ρ1t[i])
    end
end

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
