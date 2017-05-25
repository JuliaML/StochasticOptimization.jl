using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using Transformations.TestTransforms
using CatViews
using MLDataPattern
# using MLDataUtils

# include("tst_iteration.jl")
# Stop the tests
# error()

using Plots; unicodeplots(show=true,leg=false)

@testset "Rosenbrock-2" begin

    srand(1)
    n = 2
    t = tfunc(rosenbrock, n, rosenbrock_gradient)
    @show t
    # t = rosenbrock_transform(n)
    # obj = objective(t, NoLoss())

    # random starting values
    θ = params(t)
    startvals = 8rand(n)-4
    @show θ startvals

    # build a MetaLearner to use RMSProp w/ fixed learning rate,
    # setting max iterations, a custom convergence check, and a
    # custom iteration callback to collect data to plot
    converged = (m,i) -> totalcost(m) < 1e-6
    maxiter = 50000

    # this problem has no input (we're learning the params only),
    # and we know the minimum is zero, so we forever pull from this
    # fixed (inputs,targets) pair
    # data = zeros(0,1),zeros(1,1)

    # test the choices of ParamUpdaters
    for (T, lr) in [
                    (SGD, 5e-4),
                    (Adagrad, 1e-0),
                    (Adadelta, 1e-3),
                    (Adam, 1e-2),
                    (Adamax, 1e-2),
                    (RMSProp, 1e-3),
                    ]
        @show T,lr
        learner = make_learner(
            GradientLearner(lr, T()),
            # TimeLimit(10),
            # ShowStatus(1000),
            maxiter = maxiter,
            converged = converged
        )

        # learn forever (our maxiter and converge sub-learners will stop us)
        θ[:] = startvals
        learn!(t, learner) #, infinite_obs(data))

        tc = totalcost(t)
        @show tc
        @test 0 < tc < 1e-3
    end

    # rerun while tracking x/y
    x,y = zeros(0),zeros(0)
    learner = make_learner(
        GradientLearner(1e-1, Adamax()),
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
    learn!(t, learner)

    tc = totalcost(t)
    @show tc
    @test 0 < tc < 1e-3

    # plot our path to solution
    plot(x,y, ann=[(θ..., text("$θ", :left))])
end

@testset "LinReg" begin
    nin, nout = 10, 1

    # build our objective
    t = Affine(nin,nout)
    l = L2DistLoss()
    p = scaled(L1Penalty(), 1e-8)
    obj = objective(t, l, p)

    # create some fake data... affine transform plus noise
    # note: θ is the "true params"
    τ = 1000
    θ = randn(nout*(nin+1))
    w, b = splitview(θ, ((nout,nin),(nout,)))[1]
    inputs = randn(nin, τ)
    targets = w * inputs + repmat(b, 1, τ) + 0.1randn(nout, τ)

    # add norms to a trace vector
    # normvals = zeros(0)
    tracer = Tracer(Float64, (model,i) -> norm(θ - params(model)))

    # the MetaLearner have a bunch of specialized sub-learners
    learner = make_learner(
        GradientLearner(FixedLR(5e-3), Adamax()),
        ShowStatus(40),
        tracer,
        ConvergedTo(params, θ, tol=0.1, every=20),
        maxiter=5000,
    )


    learn!(obj, learner, RandomBatches(ObsView((inputs, targets)), size=20))

    # some summary output:

    println()
    plot(tracer.storage, title = "‖θₜᵣᵤₑ - θ‖²", xguide="Iteration")

    # scatter predicted output vs ground truth... should be diagonal line
    est_w, est_b = t.params.views
    pred = est_w * inputs + repmat(est_b, 1, τ)
    truth = w * inputs + repmat(b, 1, τ)
    @test maximum(pred - truth) < 5e-1

    println()
    plot(pred', truth', t=:scatter, xguide="Predicted Output", yguide="Actual Output")
end
