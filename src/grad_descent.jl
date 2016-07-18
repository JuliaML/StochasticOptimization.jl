"""
    GradDescent(ρ,β)

Gradient descent with backtracking linesearch. Updates parameters
according to `x = x - ρ⋅∇x` and shrinks the step size `ρ` by a 
multiplicative factor `0 < β < 1` whenever this parameter change
increases the objective function.
"""
type GradDescent{T<:AbstractFloat} <: Optimizer
    ρ::T                # step size
    β::T                # linesearch param
    function GradDescent(ρ,β)
        0<β<1 || error("β must be in interval (0,1)")
        ρ>0 || error("ρ must be postive")
        new(ρ,β)
    end
end
GradDescent{T<:AbstractFloat}(ρ::T=1.0,β::T=0.5) = GradDescent{T}(ρ,β)

function update!(
        x::AbstractVector,
        ∇::AbstractVector,
        f::Real,
        model::Model,
        opt::GradDescent
    )
    axpy!(-opt.ρ,∇x,x)          # take gradient step
    while f_next > f
        s = opt.ρ*opt.β         # smaller stepsize
        axpy!(opt.ρ-s,∇x,x)     # backtrack
        opt.ρ = s               # update stepsize
        f_next = value(model)   # check objective value again
        opt.ρ < 1e-12 && warn("linesearch failed") && break
    end
    return f_next
end
