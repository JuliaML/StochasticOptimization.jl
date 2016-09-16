using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using Transformations.TestTransforms
# using MLDataUtils

using Plots; unicodeplots(show=true,leg=false)

@testset "Rosenbrock-2" begin

    n = 2
    t = rosenbrock_transform(n)
    obj = objective(t, L2DistLoss())

    # random starting values
    θ = params(t)
    startvals = 8rand(n)-4

    # build a MasterLearner to use RMSProp w/ fixed learning rate,
    # setting max iterations, a custom convergence check, and a
    # custom iteration callback to collect data to plot
    converged = (m,i) -> output_value(m)[1] < 1e-8
    maxiter = 50000

    # this problem has no input (we're learning the params only),
    # and we know the minimum is zero, so we forever pull from this
    # fixed (inputs,targets) pair
    data = zeros(0,1),zeros(1,1)

    # test the choices of ParamUpdaters
    for (T, lr) in [
                    (SGD, 1e-4),
                    (Adagrad, 1e-1),
                    (Adadelta, 1e-3),
                    (Adam, 1e-2),
                    (Adamax, 1e-3),
                    (RMSProp, 1e-3),
                    ]
        @show T,lr
        learner = make_learner(
            GradientDescent(lr, T()),
            maxiter = maxiter,
            converged = converged
        )

        # learn forever (our maxiter and converge sub-learners will stop us)
        θ[:] = startvals
        learn!(obj, learner, forever(eachobs(data)))

        @show totalcost(obj)
        @test totalcost(obj) < 1e-4
    end

    # rerun while tracking x/y
    x,y = zeros(0),zeros(0)
    learner = make_learner(
        GradientDescent(FixedLR(1e-3), RMSProp()),
        maxiter = 50000,
        converged = converged,
        oniter = (m,i) -> begin
            θ = params(m)
            push!(x, θ[1])
            push!(y, θ[2])
            if mod1(i,2000)==1
                println("Iter: $i Loss: $(output_value(m)[1]) θ: $θ")
            end
        end
    )

    # learn forever (our maxiter and converge sub-learners will stop us)
    θ[:] = startvals
    learn!(obj, learner, forever(eachobs(data)))
    @show totalcost(obj)
    @test totalcost(obj) < 1e-5

    # plot our path to solution
    plot(x,y)
end

using ValueHistories
using CatViews

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
