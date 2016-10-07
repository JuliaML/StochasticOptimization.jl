module MnistTest

using Learn
import MNIST
using MLDataUtils
using StatsBase

ENV["GKS_WSTYPE"] = "x11"
using StatPlots; gr(leg=false, linealpha=0.5)

# ----------------------------------------------------------------------------
# TODO: add this to MLPlots??

# a helper class to track many variables at once over time
type TracePlot{I,T}
    indices::I
    plt::Plot{T}
end
function TracePlot(n::Int = 1; maxn::Int = 500, kw...)
    indices = if n > maxn
        # sample maxn
        shuffle(1:n)[1:maxn]
    else
        1:n
    end
    plt = plot(length(indices); kw...)
    TracePlot(indices, plt)
end
function add_data(tp::TracePlot, x::Number, y::AbstractVector)
    for (i,idx) in enumerate(tp.indices)
        push!(tp.plt.series_list[i], x, y[idx])
    end
end
add_data(tp::TracePlot, x::Number, y::Number) = add_data(tp, x, [y])


# ----------------------------------------------------------------------------

function my_test_loss(obj, testdata, totcount = 500)
    totloss = 0.0
    totcorrect = 0
    for (x,y) in eachobs(rand(eachobs(testdata), totcount))
        totloss += transform!(obj,y,x)

        # logistic version:
        # ŷ = output_value(obj.transformation)[1]
        # correct = (ŷ > 0.5 && y > 0.5) || (ŷ <= 0.5 && y < 0.5)

        # softmax version:
        ŷ = output_value(obj.transformation)
        chosen_idx = indmax(ŷ)
        correct = y[chosen_idx] > 0

        totcorrect += correct
    end
    totloss, totcorrect/totcount
end

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

    nin, nh, nout = 784, [50,50], 10

    # build our objective... a neural net with nh hidden nodes,
    # tanh activation on the hidden layer, and logistic output
    t = nnet(nin, nout, nh, :softplus, :softmax)
    # penalty = L2Penalty(1e-3)
    penalty = ElasticNetPenalty(1e-5)
    obj = objective(t, penalty)
    @show obj

    # parameter plots
    pidx = 1:2:length(t)
    pvalplts = [TracePlot(length(params(t[i])), title="$(t[i])") for i=pidx]
    ylabel!(pvalplts[1].plt, "Param Vals")
    pgradplts = [TracePlot(length(params(t[i]))) for i=pidx]
    ylabel!(pgradplts[1].plt, "Param Grads")

    # nnet plots of values and gradients
    valinplts = [TracePlot(input_length(t[i]), title="input", yguide="Layer Value") for i=1:1]
    valoutplts = [TracePlot(output_length(t[i]), title="$(t[i])", titlepos=:left) for i=1:length(t)]
    gradinplts = [TracePlot(input_length(t[i]), yguide="Layer Grad") for i=1:1]
    gradoutplts = [TracePlot(output_length(t[i])) for i=1:length(t)]

    # loss/accuracy plots
    lossplt = TracePlot(title="Test Loss", ylim=(0,Inf))
    accuracyplt = TracePlot(title="Accuracy", ylim=(0.6,1))

    doanim = false
    # anim = Animation()

    # early_stopping = ConvergenceFunction((obj,i) -> begin
    #     false
    # end)

    tracer = IterFunction((obj, i) -> begin
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
            totloss, accuracy = my_test_loss(obj, test, 500)
            @show totloss, accuracy
            add_data(lossplt, i, totloss)
            add_data(accuracyplt, i, accuracy)
        end

        # build a heatmap of the total outgoing weight from each pixel
        pixel_importance = reshape(sum(t[1].params.views[1],1), 28, 28)
        hmplt = heatmap(pixel_importance, ratio=1)

        # build a nested-grid layout for all the trace plots
        getplt(p) = p.plt
        plot(
            map(getplt, vcat(
                    pvalplts, pgradplts,
                    valinplts, valoutplts,
                    gradinplts, gradoutplts,
                    lossplt, accuracyplt
                ))...,
            hmplt,
            size = (1400,1000),
            layout=@layout([
                grid(2,length(pvalplts))
                grid(2,length(valoutplts)+1)
                grid(1,3){0.2h}
            ])
        )

        if doanim
            lastframe = 5000
            if i < lastframe
                frame(anim)
            elseif i == lastframe
                gif(anim, fps=10)
            end
        end

        gui()
        # if i>0
        #     @profile gui()
        # else
        #     gui()
        # end
    end)

    # trace once before we start learning to see initial values
    tracer.f(obj, 0)

    # create a gradient descent learner and learn over infinite minibatches
    learner = make_learner(
        # GradientLearner(5e-3, RMSProp(0.9)),
        GradientLearner(5e-2, Adadelta()),
        # GradientLearner(1e-1, SGD(0.3)),
        # early_stopping,
        tracer,
        maxiter = 50000
    )
    learn!(obj, learner, infinite_batches(train, size=5))

    obj, learner
end

obj, learner = doit()

end #module
