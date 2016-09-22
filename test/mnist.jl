module MnistTest

using Learn
import MNIST
import MLDataUtils: rescale!
using StatsBase
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
    x_train, y_train = MNIST.traindata()
    x_test, y_test = MNIST.testdata()

    μ, σ = rescale!(x_train)
    rescale!(x_test, μ, σ)

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
    train01 = filterobs(i -> y_train[i] < 1.5, x_train, y_train)
    test01 = filterobs(i -> y_test[i] < 1.5, x_test, y_test)

    # At this point we have train and test where each column of x is length-784 corresponding to pixel intensities,
    # and each column of y is length-10 corresponding to output class.

    nin, nh, nout = 784, 10, 1

    # build our objective... a neural net with nh hidden nodes,
    # tanh activation on the hidden layer, and logistic output
    t = nnet(nin, nout, [nh], :softplus, :logistic)
    obj = objective(t, L1Penalty(1e-4))
    @show obj


    # @show fit(StatsBase.Histogram, vec(params(t[1])))

    # θ = params(t[1])
    # ∇ = grad(t[1])
    # np = length(θ)

    # # store 200 random weights
    # ni = 2000
    # indices = rand(1:np, ni)
    # wplt = TracePlot(ni, title="Param Weights")
    # gplt = TracePlot(ni, title="Gradients")

    lossplt = TracePlot(title="Test Loss")
    accuracyplt = TracePlot(title="Accuracy")


    # outplts = [
    #     TracePlot(output_length(t[1]), l=0.2, m=(3,0.3,stroke(0)), title="First Affine Outputs"),
    #     TracePlot(output_length(t[3]), l=0.2, m=(5,0.5,stroke(0)), title="Second Affine Outputs"),
    # ]

    pidx = [1,3]
    pvalplts = [TracePlot(length(params(t[i])), title="param val, i=$i") for i=pidx]
    pgradplts = [TracePlot(length(params(t[i])), title="param grad, i=$i") for i=pidx]

    valinplts = [TracePlot(input_length(t[i]), title="in val") for i=1:1]
    valoutplts = [TracePlot(output_length(t[i]), title="out: $(t[i])") for i=1:length(t)]
    gradinplts = [TracePlot(input_length(t[i]), title="in grad") for i=1:1]
    gradoutplts = [TracePlot(output_length(t[i]), title="grad: $(t[i])") for i=1:length(t)]

    # early_stopping = ConvergenceFunction((model,i) -> begin
    #     false
    # end)

    tracer = IterFunction((model, i) -> begin
        n = 50
        mod1(i,n)==n || return false

        # add to the trace plots
        # add_data(wplt, i, θ[indices])
        # add_data(gplt, i, ∇[indices])

        # add param data
        for (j,k) in enumerate(pidx)
            add_data(pvalplts[j], i, params(t[k]))
            add_data(pgradplts[j], i, grad(t[k]))
        end

        # add input/output data
        for j=1:length(t)
            if j==1
                add_data(valinplts[j], i, input_value(t[j]))
                add_data(gradinplts[j], i, input_grad(t[j]))
            end
            add_data(valoutplts[j], i, output_value(t[j]))
            add_data(gradoutplts[j], i, output_grad(t[j]))
        end
        # add_data(outplts[1], i, output_value(t[1]))
        # add_data(outplts[2], i, output_value(t[3]))

        # sample 50 points from the test set and compute/save the loss
        @show i
        if mod1(i,500)==500
            totloss = 0.0
            totcorrect = 0
            totcount = 500
            for (x,y) in eachobs(rand(eachobs(test01), totcount))
                totloss += transform!(model,y,x)
                ŷ = output_value(t)[1]
                correct = (ŷ > 0.5 && y > 0.5) || (ŷ <= 0.5 && y < 0.5)
                totcorrect += correct
            end
            @show totloss, totcorrect/totcount
            add_data(lossplt, i, totloss)
            add_data(accuracyplt, i, totcorrect/totcount)
        end

        getplt(p) = p.plt

        # @show map(length, (pvalplts, pgradplts, valinplts, valoutplts, gradinplts, gradoutplts))

        plot(
            # wplt.plt,
            # gplt.plt,
            map(getplt, vcat(
                    pvalplts, pgradplts,
                    valinplts, valoutplts,
                    gradinplts, gradoutplts,
                    lossplt, accuracyplt
                ))...,
            # [pvalplts[i].plt for i=[1,3]]...,
            # [pgradplts[i].plt for i=[1,3]]...,
            # [valinplts[i].plt for i=1:4]...,
            # [valoutplts[i].plt for i=1:4]...,
            # [gradinplts[i].plt for i=1:4]...,
            # [gradoutplts[i].plt for i=1:4]...,
            # lossplt.plt,
            # accuracyplt.plt,
            # outplts[1].plt,
            # outplts[2].plt,
            size = (1400,1000),
            layout=@layout([grid(2,2); grid(2,5); grid(1,2){0.2h}])
        ); gui()
    end)

    # trace once before we start learning to see initial values
    tracer.f(obj, 0)

    # create a gradient descent learner and learn over infinite minibatches
    learner = make_learner(
        GradientDescent(1e-4, RMSProp(0.9)),
        # GradientDescent(1e-1, SGD(0.3)),
        # early_stopping,
        tracer,
        maxiter = 10000
    )
    # learn!(obj, learner, infinite_batches(getobs(train01,1:1), size=10))
    learn!(obj, learner, infinite_batches(train01, size=10))

    obj, learner
end

obj, learner = doit()

end #module
