
# # this might be used for Nesterov lookahead, or similar
# before_grad_calc(θ::AbstractVector, updater::ParamUpdater, ∇::AbstractVector) = return

# TODO: add citations/links for each method
# note: good ref: http://sebastianruder.com/optimizing-gradient-descent/
# note: good ref: https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM

init(updater::ParamUpdater, model::Learnable) = init(updater, params(model))

"Stochastic Gradient Descent with Momentum"
type SGD{T<:Number} <: ParamUpdater
    mom::T              # momentum
    lastchg::Vector{T}   # most recent grad change
    (::Type{SGD{T}}){T <: Number}(mom::T) = new{T}(mom)
end
SGD{T}(::Type{T}, mom = T(0.0)) = SGD{T}(T(mom))
SGD{T}(mom::T = 0.0) = SGD{T}(mom)

function init(updater::SGD, θ::AbstractVector)
    updater.lastchg = zeros(θ)
    return
end

# # add momentum into θ, and reset lastchg
# function before_grad_calc(θ::AbstractVector, updater::SGD, ∇::AbstractVector)
#     @inbounds for (i,k) in zip(eachindex(θ), eachindex(updater.lastchg))
#         momchg = updater.mom * updater.lastchg[k]
#         updater.lastchg[k] = momchg
#         θ[i] += momchg
#     end
# end

#
function update!(θ::AbstractVector, updater::SGD, ∇::AbstractVector, lr::Number)
    @inbounds for (i,j,k) in zip(eachindex(θ), eachindex(∇), eachindex(updater.lastchg))
        chg = -lr * ∇[j] + updater.mom * updater.lastchg[k]
        θ[i] += chg
        updater.lastchg[k] = chg
    end
end

# -----------------------------------------------------------------------

"Adaptive Gradient"
type Adagrad{T<:Number} <: ParamUpdater
    ϵ::T
    G::Vector{T}  # sum of squared gradients
    (::Type{Adagrad{T}}){T <: Number}(ϵ::T) = new{T}(ϵ)
end
Adagrad{T}(::Type{T}, ϵ = 1e-2) = Adagrad{T}(T(ϵ))
Adagrad() = Adagrad(Float64)

function init(updater::Adagrad, θ::AbstractVector)
    updater.G = zeros(θ)
    return
end

function update!(θ::AbstractVector, updater::Adagrad, ∇::AbstractVector, lr::Number)
    @inbounds for (i,j,k) in zip(eachindex(θ), eachindex(∇), eachindex(updater.G))
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
    (::Type{Adadelta{T}}){T<:Number}(ϵ::T, ρ::T) = new{T}(ϵ, ρ)
end
Adadelta{T}(::Type{T}, ϵ = 1e-2, ρ = 0.97) = Adadelta{T}(T(ϵ), T(ρ))
Adadelta{T}(ϵ::T = 1e-2, ρ::T = 0.97) = Adadelta{T}(ϵ, ρ)

function init(updater::Adadelta, θ::AbstractVector)
    updater.dmean = zeros(θ)
    updater.Gmean = zeros(θ)
    return
end

function update!{T}(θ::AbstractVector, updater::Adadelta{T}, ∇::AbstractVector, lr::Number)
    ϵ = updater.ϵ
    ρ = updater.ρ
    dm = updater.dmean
    Gm = updater.Gmean
    @inbounds for i=1:length(θ)
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
    ρ₁::T # decay for first moment (β₁ in the paper)
    ρ₂::T # decay for second moment (β₂ in the paper)
    m::Vector{T} # average first moment
    v::Vector{T} # average second moment
    ρ₁ᵗ::Vector{T} # β₁ᵗ from the paper... t-th power of β₁
    ρ₂ᵗ::Vector{T} # β₂ᵗ from the paper... t-th power of β₂
    (::Type{Adam{T}}){T<:Number}(ϵ::T, ρ₁::T, ρ₂::T) = new{T}(ϵ, ρ₁, ρ₂)
end
Adam{T}(::Type{T}, ϵ = T(1e-8), ρ₁ = T(0.9), ρ₂ = T(0.999)) = Adam{T}(T(ϵ), T(ρ₁), T(ρ₂))
Adam{T}(ϵ::T = 1e-8, ρ₁::T = 0.9, ρ₂::T = 0.999) = Adam{T}(ϵ, ρ₁, ρ₂)

function init(updater::Adam, θ::AbstractVector)
    updater.m = zeros(θ)
    updater.v = zeros(θ)
    updater.ρ₁ᵗ = ones(θ)
    updater.ρ₂ᵗ = ones(θ)
    return
end

function update!{T}(θ::AbstractVector, updater::Adam{T}, ∇::AbstractVector, lr::Number)
    ϵ = updater.ϵ
    ρ₁ = updater.ρ₁
    ρ₂ = updater.ρ₂
    m = updater.m
    v = updater.v
    ρ₁ᵗ = updater.ρ₁ᵗ
    ρ₂ᵗ = updater.ρ₂ᵗ
    @inbounds for i=1:length(θ)
        m[i] = ρ₁ * m[i] + (one(T) - ρ₁) * ∇[i]
        v[i] = ρ₂ * v[i] + (one(T) - ρ₂) * ∇[i]^2
        ρ₁ᵗ[i] *= ρ₁
        ρ₂ᵗ[i] *= ρ₂
        θ[i] -= lr * m[i] * sqrt((one(T) - ρ₂ᵗ[i]) / (v[i] + ϵ)) / (one(T) - ρ₁ᵗ[i])
    end
end

# -----------------------------------------------------------------------

"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

AdaMax is similar to Adam, using the p-norm as p -> ∞
"""
type Adamax{T<:Number} <: ParamUpdater
    ρ₁::T # decay for first moment (β₁ in the paper)
    ρ₂::T # decay for second moment (β₂ in the paper)
    m::Vector{T} # average first moment
    u::Vector{T} # average second moment
    ρ₁ᵗ::Vector{T} # β₁ᵗ from the paper... t-th power of β₁
    (::Type{Adamax{T}}){T<:Number}(ρ₁::T, ρ₂::T) = new{T}(ρ₁, ρ₂)
end
Adamax{T}(::Type{T}, ρ₁ = T(0.9), ρ₂ = T(0.999)) = Adamax{T}(T(ρ₁), T(ρ₂))
Adamax{T}(ρ₁::T = 0.9, ρ₂::T = 0.999) = Adamax{T}(ρ₁, ρ₂)

function init(updater::Adamax, θ::AbstractVector)
    updater.m = zeros(θ)
    updater.u = zeros(θ)
    updater.ρ₁ᵗ = ones(θ)
    return
end

function update!{T}(θ::AbstractVector, updater::Adamax{T}, ∇::AbstractVector, lr::Number)
    ρ₁ = updater.ρ₁
    ρ₂ = updater.ρ₂
    m = updater.m
    u = updater.u
    ρ₁ᵗ = updater.ρ₁ᵗ
    @inbounds for i=1:length(θ)
        m[i] = ρ₁ * m[i] + (one(T) - ρ₁) * ∇[i]
        u[i] = max(ρ₂ * u[i], abs(∇[i]))
        ρ₁ᵗ[i] *= ρ₁
        θ[i] -= lr * m[i] / ((u[i] + T(1e-10)) * (one(T) - ρ₁ᵗ[i]))
    end
end

# -----------------------------------------------------------------------

# TODO: RMSProp: http://climin.readthedocs.io/en/latest/rmsprop.html#tieleman2012rmsprop

type RMSProp{T<:Number} <: ParamUpdater
    γ::T # exponential weight of ∇² avg
    g::Vector{T}  # the exponential mean
    (::Type{RMSProp{T}}){T<:Number}(γ::T) = new{T}(γ)
end
RMSProp{T}(::Type{T}, γ = T(0.95)) = RMSProp{T}(T(γ))
RMSProp{T}(γ::T) = RMSProp{T}(γ)
RMSProp() = RMSProp{Float64}(0.95)

function init(updater::RMSProp, θ::AbstractVector)
    updater.g = ones(θ)
    return
end

function update!{T}(θ::AbstractVector, updater::RMSProp{T}, ∇::AbstractVector, lr::Number)
    γ = updater.γ
    g = updater.g
    @inbounds for i=1:length(θ)
        g[i] = γ * g[i] + (one(T) - γ) * ∇[i]^2
        θ[i] -= lr * ∇[i] / sqrt(g[i] + 1e-10)
    end
end
