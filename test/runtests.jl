using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using Transformations.TestTransforms
using CatViews
using MLDataUtils

# @testset "Data Iteration" begin
#     n = 4
#     X = rand(2,n)
#     y = rand(n)
#
#     # nobs/getobs of arrays
#     @test nobs(X) == n
#     @test nobs(y) == n
#     @test getobs(X, 1) == X[:,1]
#     @test getobs(y, 1) == y[1]
#     @test getobs(X, 1:2) == X[:, 1:2]
#     @test getobs(y, 1:2) == y[1:2]
#
#     # construction
#     subset = eachobs(X, y)
#     @test typeof(subset) <: DataSubset{Tuple{Matrix{Float64},Vector{Float64}}}
#     @test length(subset) == n
#     @test subset.indices == 1:n
#     @test subset.source == (X,y)
#     @test nobs(subset) == n
#
#     # iterating... sort of
#     o1, o2, o3, o4 = subset
#     @test o2 == (X[:,2], y[2])
#
#     # extraction
#     subset2 = DataSubset((X, y), 1:1)
#     cx, cy = collect(subset2)
#     @test typeof(cx) <: Matrix
#     @test cx == X[:,1:1]
#     @test typeof(cy) <: Vector
#     @test cy == y[1:1]
#
#     # random obs
#     (x1,x2),yi = rand(subset)
#     @test x1 in X
#     @test x2 in X
#     @test yi in y
#
#     # random arrays
#     xs,ys = rand(subset, 2)
#     @test size(xs) == (2,2)
#     @test size(ys) == (2,)
#
#     # getindex
#     for i=1:n
#         @test subset[i] == (X[:,i], y[i])
#     end
#
#     # iteration
#     for (i,(x,yi)) in enumerate(subset)
#         @test x == X[:,i]
#         @test yi == y[i]
#     end
#
#     # shuffling
#     ss = shuffled(X,y)
#     @test length(ss.indices) == n
#
#     # test/train split
#     train, test = batches(X,y,size=0.5)
#     # @show train test
#     @test typeof(train) <: Tuple
#     @test typeof(test) <: Tuple
#     @test train[1] == view(X,:,1:2)
#     @test test[1] == view(X,:,3:4)
#     @test train[2] == view(y,1:2)
#     @test test[2] == view(y,3:4)
#     @test nobs(train) == 2
#     @test nobs(test) == 2
#
#     # minibatch split
#     bs = batches(X,y,size=2)
#     @test typeof(bs) <: DataSubsets
#     @test length(bs) == 2
#     for (x,yi) in bs
#         # @show x yi
#         @test typeof(x) <: SubArray
#         @test typeof(yi) <: SubArray
#         @test size(x) == (2,2)
#         @test length(yi) == 2
#         for (xj,yj) in eachobs(x,yi)
#             # just to make sure there's no errors in nesting...
#         end
#     end
#
#     # train/validate/test split
#     X = rand(2,10)
#     y = rand(10)
#     bs = batches(X,y,size=(0.5,0.2))
#     train, validate, test = bs
#     @test nobs(train) == 5
#     @test nobs(validate) == 2
#     @test nobs(test) == 3
#     @test bs.subsets[1].indices == 1:5
#     @test bs.subsets[2].indices == 6:7
#     @test bs.subsets[3].indices == 8:10
#
#     # kfolds
#     kf = kfolds(X, y)
#     @test typeof(kf) <: KFolds
#     @test kf.k == 5
#     @test StochasticOptimization.start_index(kf, 2) == 3
#     @test StochasticOptimization.end_index(kf, 2) == 4
#     i = 0
#     for (train, test) in kf
#         i += 1
#         # @show typeof(train) typeof(test)
#         @test typeof(train) <: Tuple
#         @test typeof(test) <: Tuple
#         @test train == getobs((X,y), setdiff(1:10, 2i-1:2i))
#         @test test == getobs((X,y), 2i-1:2i)
#         @test nobs(train) == 8
#         @test nobs(test) == 2
#     end
#     @test i == kf.k
#
#     loo = leave_one_out(y)
#     @test typeof(loo) <: KFolds
#     @test loo.k == nobs(X)
#     for (train,test) in loo
#         @test nobs(train) == 9
#         @test nobs(test) == 1
#     end
#
#     # filtering
#     newx,newy = filterobs(i -> i%2==0, X, y)
#     @test newx == X[:,2:2:10]
#     @test newy == y[2:2:10]
# end

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
    learn!(t, learner) #, infinite_batches(data))

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
    p = L1Penalty(1e-8)
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

    learn!(obj, learner, infinite_batches(inputs, targets, size=20))

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
