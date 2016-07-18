# TODO:
#   1) better convergence testing

function learn!(
        model::Model,
        opt::Optimizer;
        iterations=1000,
        ftol::Float64=1e-6
    )
    x = paramvec(model) # returns view of all parameters as a vector
    ∇ = gradvec(model)  # returns view of gradient as a vector
    f = value(model)
    fhist = [f]

    converged = false
    iter = 0
    while !converged && iter < iterations
        iter += 1
        f = update!(x,∇,f,model,opt)    # uodates x
        grad!(model)                    # updates ∇
        converged = (fhist[end]-f)/fhist[end] < ftol
        push!(fhist,f)
    end

    return fhist,converged
end
