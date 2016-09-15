module MnistTest

using Learn
import MNIST
using StatPlots; glvisualize(leg=false, size=(900,700))

function doit()

    # our data:
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # normalize
    μ = mean(train_x)
    σ = std(train_x)
    train_x = convert(Matrix{Float64}, (train_x) / σ)
    test_x = convert(Matrix{Float64}, (test_x) / σ)

    # TODO: this MUST be defined somewhere... where?
    # function to_one_hot(y::AbstractVector)
    #     yint = map(yi->round(Int,yi)+1, y)
    #     @show size(yint), minimum(yint), maximum(yint)
    #     nclasses = maximum(yint)
    #     hot = zeros(Float64, nclasses, length(y))
    #     for (i,yi) in enumerate(yint)
    #         hot[yi,i] = 1.0
    #     end
    #     hot
    # end
    # train_y, test_y = map(to_one_hot, (train_y, test_y))

    to_isone(y::AbstractVector) = (z = Array(eltype(y), 1, length(y)); map!(yi->float(yi==1.0), z, y))
    train_y, test_y = map(to_isone, (train_y, test_y))

    @show train_y[:,1]

    # At this point we have train and test where each column of x is length-784 corresponding to pixel intensities,
    # and each column of y is length-10 corresponding to output class.

    nin, nout = 784, 1
    nh = 100

    # build our objective
    t = Chain(Float64,
        Affine(Float64,nin,nh),
        Activation{:tanh,Float64}(nh),
        Affine(Float64,nh,nout),
        # Activation{:softmax,Float64}(nout)
        Activation{:logistic,Float64}(nout)
    )
    # l = CrossEntropy{Float64}(nout)
    l = CrossentropyLoss()
    p = L2Penalty(Float64(1e-6))
    obj = objective(t, l, p)

    # ps = Float64[]
    # xs = Float64[]
    # zs = Float64[]
    ni = 200
    np = length(params(t))
    indices = rand(1:np, ni)
    plt1 = plot(ni)
    # plt2 = plot(length(params(t.ts[3])))
    totl = Float64[]
    Σ₁ = Float64[]
    Σ₁ₜ = Float64[]
    Σ₂ = Float64[]
    Σ₂ₜ = Float64[]

    early_stopping = ConvergenceFunction((model,i) -> begin
        n = 50
        mod1(i,n)==n || return false
        @show i
        # append!(ps, params(model)[indices])
        # append!(xs, i*ones(ni))
        # append!(zs, 1:ni)
        θ = params(model)
        for i=1:ni
            push!(plt1[1][i], θ[indices[i]])
        end
        append!(Σ₁, output_value(t.ts[1]))
        append!(Σ₁ₜ, i*ones(length(output_value(t.ts[1]))))
        append!(Σ₂, output_value(t.ts[3]))
        append!(Σ₂ₜ, i*ones(length(output_value(t.ts[3]))))
        totloss = 0.0
        for (x,y) in DataSubset((test_x,test_y),rand(1:size(test_x,2), 50))
            totloss += transform!(model,y,x)
        end
        push!(totl, totloss)
        plot(
            # histogram2d(xs,ps,bins=(div(i,n),40)),
            plt1,
            # plt2,
            # scatter(xs,ps, m=(:black,3,0.2,stroke(0)), marker_z = zs),
            plot(totl),
            scatter(Σ₁ₜ, Σ₁, m=(:black,3,0.3,stroke(0))),
            scatter(Σ₂ₜ, Σ₂, m=(:black,5,0.5,stroke(0))),
            layout=@layout([a;b{0.2h}; c d])
        ); gui()
        @show totloss
        false
    end)

    tracer = ConvergenceFunction((model, i) -> begin
        @show i, output_value(model),output_value(model.transformation)
        @show output_value(model.transformation.ts[1])
        false
    end)

    learner = MasterLearner(
        GradientDescent(FixedLR(Float64(1e-1)), SGD(Float64)),
        MaxIter(50000),
        early_stopping
    )
    learn!(obj, learner, MiniBatches((train_x, train_y), 10))
    obj, learner
end

obj, learner = doit()

end #module
