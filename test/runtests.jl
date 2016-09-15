using StochasticOptimization
using Base.Test

using ObjectiveFunctions
# using MLDataUtils

using ValueHistories
using CatViews
using Plots; unicodeplots(show=true,leg=false)

# this is an example custom learning strategy
# which tracks the norm(true_params - estimated_params)
type NormTracer <: LearningStrategy
    θ::Vector{Float64} # true params
    normvals::Vector{Float64}
end
NormTracer(θ) = NormTracer(θ, zeros(0))
function StochasticOptimization.iter_hook(nt::NormTracer, model, i::Int)
    normw = norm(nt.θ - params(model))
    push!(nt.normvals, normw)
    # @show i, normw
end

@testset "LinReg" begin
    nin, nout = 10, 1

    # build our objective
    t = Affine(nin,nout)
    l = L2DistLoss()
    p = L1Penalty(1e-8)
    obj = objective(t, l, p)

    # create some fake data... affine transform plus noise
    # note: θ is the "true params"
    τ = 1000
    θ = randn(nout*(nin+1))
    w, b = splitview(θ, ((nout,nin),(nout,)))[1]
    inputs = randn(nin, τ)
    targets = w * inputs + repmat(b, 1, τ) + 0.1randn(nout, τ)

    # our learning strategy... SGD with a fixed learning rate
    strat = GradientDescent(FixedLR(5e-3), Adamax())

    # add norms to a trace vector
    tracer = NormTracer(θ)

    # check for convergence to the true parameter vector
    θ_converge = ConvergenceFunction((model,i) -> begin
        if mod1(i,100) == 100
            normw = norm(θ - params(model))
            @show i,normw
            if normw < 0.1
                info("Converged after $i iterations: $normw")
                return true
            end
        end
        false
    end)

    # the MasterLearner have a bunch of specialized sub-learners
    learner = make_learner(
        strat,
        tracer,
        θ_converge,
        maxiter=5000
    )

    learn!(obj, learner, MiniBatches((inputs, targets), 20))

    println()
    plot(tracer.normvals, title = "‖θₜᵣᵤₑ - θ‖²",
         xguide="Iteration")

    # scatter predicted output vs ground truth... should be diagonal line
    est_w, est_b = t.params.views
    pred = est_w * inputs + repmat(est_b, 1, τ)
    truth = w * inputs + repmat(b, 1, τ)
    @test maximum(pred - truth) < 5e-1

    println()
    plot(pred', truth', t=:scatter,
         xguide="Predicted Output",
         yguide="Actual Output")
end
