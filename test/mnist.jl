module MnistTest

using Learn
import MNIST
using MLDataUtils
using StatsBase
using StatPlots; gr(leg=false, linealpha=0.5)

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

    # normalize the input data given μ/σ for the input training data
    μ, σ = rescale!(x_train)
    rescale!(x_test, μ, σ)

    # note: needs my update to MLDataUtils
    y_train, y_test = map(to_one_hot, (y_train, y_test))

    # to_isone(y::AbstractVector) = (z = Array(eltype(y), 1, length(y)); map!(yi->float(yi==1.0), z, y))
    # y_train, y_test = map(to_isone, (y_train, y_test))

    # # keep only 0's and 1's
    # train = filterobs(i -> y_train[i] < 1.5, x_train, y_train)
    # test = filterobs(i -> y_test[i] < 1.5, x_test, y_test)

    train = (x_train, y_train)
    test = (x_test, y_test)

    # At this point we have train and test where each column of x is length-784 corresponding to pixel intensities,
    # and each column of y is length-10 corresponding to output class.

    nin, nh, nout = 784, [10, 30], 10

    # build our objective... a neural net with nh hidden nodes,
    # tanh activation on the hidden layer, and logistic output
    t = nnet(nin, nout, nh, :softplus, :softmax)
    obj = objective(t, L1Penalty(1e-4))
    @show obj

    # parameter plots
    pidx = [1,3,5]
    pvalplts = [TracePlot(length(params(t[i])), title="params: $i") for i=pidx]
    pgradplts = [TracePlot(length(params(t[i])), title="grad: $i") for i=pidx]

    # nnet plots of values and gradients
    valinplts = [TracePlot(input_length(t[i]), title="input", yguide="Layer Value") for i=1:1]
    valoutplts = [TracePlot(output_length(t[i]), title="$(t[i])", titlepos=:left) for i=1:length(t)]
    gradinplts = [TracePlot(input_length(t[i]), title="input", yguide="Layer Grad") for i=1:1]
    gradoutplts = [TracePlot(output_length(t[i]), title="$(t[i])", titlepos=:left) for i=1:length(t)]

    # loss/accuracy plots
    lossplt = TracePlot(title="Test Loss", ylim=(0,Inf))
    accuracyplt = TracePlot(title="Accuracy", ylim=(0,1))

    # early_stopping = ConvergenceFunction((model,i) -> begin
    #     false
    # end)

    tracer = IterFunction((model, i) -> begin
        n = 100
        mod1(i,n)==n || return false

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

        # sample points from the test set and compute/save the loss
        @show i
        if mod1(i,500)==500
            totloss = 0.0
            totcorrect = 0
            totcount = 500
            for (x,y) in eachobs(rand(eachobs(test), totcount))
                totloss += transform!(model,y,x)

                # logistic version:
                # ŷ = output_value(t)[1]
                # correct = (ŷ > 0.5 && y > 0.5) || (ŷ <= 0.5 && y < 0.5)

                # softmax version:
                ŷ = output_value(t)
                chosen_idx = indmax(ŷ)
                correct = y[chosen_idx] > 0

                totcorrect += correct
            end
            @show totloss, totcorrect/totcount
            add_data(lossplt, i, totloss)
            add_data(accuracyplt, i, totcorrect/totcount)
        end

        # build a nested-grid layout for all the trace plots
        getplt(p) = p.plt
        plot(
            map(getplt, vcat(
                    pvalplts, pgradplts,
                    valinplts, valoutplts,
                    gradinplts, gradoutplts,
                    lossplt, accuracyplt
                ))...,
            size = (1400,1000),
            layout=@layout([grid(2,length(pvalplts)); grid(2,length(valoutplts)+1); grid(1,2){0.2h}])
        ); gui()
    end)

    # trace once before we start learning to see initial values
    tracer.f(obj, 0)

    # create a gradient descent learner and learn over infinite minibatches
    learner = make_learner(
        GradientDescent(1e-3, RMSProp(0.9)),
        # GradientDescent(1e-1, SGD(0.3)),
        # early_stopping,
        tracer,
        maxiter = 10000
    )
    learn!(obj, learner, infinite_batches(train, size=5))

    obj, learner
end

obj, learner = doit()

end #module
