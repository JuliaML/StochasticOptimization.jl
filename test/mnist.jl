module MnistTest

using Learn
import MNIST
using StatPlots; gr(leg=false, size=(700,700))

# ----------------------------------------------------------------------------
# TODO: add this to MLPlots??

# a helper class to track many variables at once over time
type TracePlot{T}
    n::Int
    plt::Plot{T}
end
function TracePlot(n::Int = 1; kw...)
    plt = plot(n; kw...)
    TracePlot(n,plt)
end
function add_data(tp::TracePlot, x::Number, y::AbstractVector)
    for (i,series) in enumerate(tp.plt.series_list)
        push!(series, x, y[i])
    end
end
add_data(tp::TracePlot, x::Number, y::Number) = add_data(tp, x, [y])

# ----------------------------------------------------------------------------

function doit()

    # our data:
    train = MNIST.traindata()
    test = MNIST.testdata()

    # scale pixels to [0,1]
    train[1][:] ./= 255
    test[1][:] ./= 255

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

    # to_isone(y::AbstractVector) = (z = Array(eltype(y), 1, length(y)); map!(yi->float(yi==1.0), z, y))
    # train_y, test_y = map(to_isone, (train_y, test_y))

    # keep only 0's and 1's
    train01 = filterobs(i -> train[2][i] < 1.5, train)
    test01 = filterobs(i -> test[2][i] < 1.5, test)

    # At this point we have train and test where each column of x is length-784 corresponding to pixel intensities,
    # and each column of y is length-10 corresponding to output class.

    nin, nh, nout = 784, 10, 1

    # build our objective... a neural net with nh hidden nodes,
    # tanh activation on the hidden layer, and logistic output
    t = nnet(nin, nout, [nh], :tanh, :logistic)
    obj = objective(t, L1Penalty(1e-4))


    # store 200 random weights
    θ = params(t)
    np = length(θ)
    ni = 200
    indices = rand(1:np, ni)
    wplt = TracePlot(ni, title="Param Weights")

    lossplt = TracePlot(title="Test Loss")
    outplts = [
        TracePlot(output_length(t[1]), l=0.2, m=(3,0.3,stroke(0)), title="First Affine Outputs"),
        TracePlot(output_length(t[3]), l=0.2, m=(5,0.5,stroke(0)), title="Second Affine Outputs"),
    ]

    # early_stopping = ConvergenceFunction((model,i) -> begin
    #     false
    # end)

    tracer = IterFunction((model, i) -> begin
        n = 50
        mod1(i,n)==n || return false


        # sample 50 points from the test set and compute/save the loss
        totloss = 0.0
        for (x,y) in eachobs(rand(eachobs(test01), 50))
            totloss += transform!(model,y,x)
        end
        @show i,totloss

        # add to the trace plots
        add_data(wplt, i, θ[indices])
        add_data(outplts[1], i, output_value(t[1]))
        add_data(outplts[2], i, output_value(t[3]))
        add_data(lossplt, i, totloss)

        plot(
            wplt.plt,
            lossplt.plt,
            outplts[1].plt,
            outplts[2].plt,
            layout=@layout([a;b{0.2h};c d])
        ); gui()
    end)

    # create a gradient descent learner and learn over infinite minibatches
    learner = make_learner(
        GradientDescent(1e-3, RMSProp()),
        # early_stopping,
        tracer,
        maxiter = 10000
    )
    learn!(obj, learner, infinite_batches(train01, size=10))

    obj, learner
end

obj, learner = doit()

end #module
