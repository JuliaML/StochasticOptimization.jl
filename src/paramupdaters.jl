
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

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
