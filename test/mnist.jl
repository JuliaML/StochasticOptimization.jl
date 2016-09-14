module MnistTest

using Learn
import MNIST
using Plots; gr(leg=false,show=true)

function doit()

    # our data:
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # normalize
    μ = mean(train_x)
    σ = std(train_x)
    train_x = convert(Matrix{Float32}, (train_x - μ) / σ)
    test_x = convert(Matrix{Float32}, (test_x - μ) / σ)

    # TODO: this MUST be defined somewhere... where?
    function to_one_hot(y::AbstractVector)
        yint = map(yi->round(Int,yi)+1, y)
        @show size(yint), minimum(yint), maximum(yint)
        nclasses = maximum(yint)
        hot = zeros(Float32, nclasses, length(y))
        for (i,yi) in enumerate(yint)
            hot[yi,i] = 1.0
        end
        hot
    end
    train_y, test_y = map(to_one_hot, (train_y, test_y))
    @show train_y[:,1]

    # At this point we have train and test where each column of x is length-784 corresponding to pixel intensities,
    # and each column of y is length-10 corresponding to output class.

    nin, nout = 784, 10

    # build our objective
    t = Chain(Float32, Affine{Float32}(nin,nout), Activation{:softmax,Float32}(nout))
    l = CrossEntropy{Float32}(nout)
    p = L1Penalty(Float32(1e-6))
    obj = objective(t, l, p)

    ps = Float32[]
    xs = Float32[]

    early_stopping = ConvergenceFunction((model,i) -> begin
        mod1(i,50)==50 || return false
        @show i
        append!(ps, params(model))
        append!(xs, i*ones(length(params(model))))
        totloss = 0.0
        for (x,y) in DataSubset((test_x,test_y),rand(1:size(test_x,2), 500))
            totloss += transform!(model,y,x)
        end
        scatter(xs,ps,ms=1)
        @show totloss
        false
    end)

    tracer = ConvergenceFunction((model, i) -> begin
        @show i, output_value(model),output_value(model.transformation)
        @show output_value(model.transformation.ts[1])
        false
    end)

    learner = MasterLearner(
        GradientDescent(FixedLR(Float32(1e-2)), Adam(Float32)),
        MaxIter(2000),
        early_stopping
    )
    learn!(obj, learner, MiniBatches((train_x, train_y), 50))
    obj, learner
end

obj, learner = doit()

end #module
