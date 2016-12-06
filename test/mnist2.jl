module MnistTest

# this re-exports Transformations, StochasticOptimization, PenaltyFunctions, and ObjectiveFunctions
using Learn

# my version of ML iteration.  Hopefully will be replaced with what's currently in MLDataUtils dev branch
using StochasticOptimization.Iteration

import MLDataUtils: rescale!

# for loading the data
import MNIST

# for plotting
using StatPlots, MLPlots
gr(leg=false, linealpha=0.5)

# ----------------------------------------------------------------------------

# create a one-hot matrix given class labels
# TODO: this should be added as a utility in MLDataUtils
function to_one_hot(y::AbstractVector)
    yint = map(yi->round(Int,yi)+1, y)
    nclasses = maximum(yint)
    hot = zeros(Float64, nclasses, length(y))
    for (i,yi) in enumerate(yint)
        hot[yi,i] = 1.0
    end
    hot
end

# randomly pick a subset of testdata (size = totcount) and compute the total loss
function my_test_loss(obj, testdata, totcount = 500)
    totloss = 0.0
    totcorrect = 0
    for (x,y) in each_obs(rand(each_obs(testdata), totcount))
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

    nin, nh, nout = 784, [100], 10

    # build our objective from a neural net with nh hidden nodes and cross-entropy loss
    # NOTE: cross entropy loss is assumed from the softmax output
    t = nnet(nin, nout, nh, :relu, :softmax)
    # penalty = ElasticNetPenalty(1e-5)
    penalty = L2Penalty(1e-5)
    obj = objective(t, penalty)
    @show obj

    # ---------------------------------------
    # this section is ONLY FOR PLOTTING
    # it can be skipped completely if you only care about learning the model

    # the parts of the plot
    chainplt = ChainPlot(t, maxn=100)
    lossplt = TracePlot(title="Test Loss", ylim=(0,Inf))
    accuracyplt = TracePlot(title="Accuracy", ylim=(0.6,1))
    hmplt = heatmap(rand(28,28), ratio=1)

    # put together the full plot... a ChainPlot with loss, accuracy, and the heatmap
    plot(
        chainplt.plt,
        lossplt.plt,
        accuracyplt.plt,
        hmplt,
        size = (1200,800),
        layout=@layout([a; grid(1,3){0.2h}])
    )

    doanim = false
    # anim = Animation()

    tracer = IterFunction((obj, i) -> begin
        # sample points from the test set and compute/save the loss
        @show i
        if mod1(i,500)==500
            totloss, accuracy = my_test_loss(obj, test, 200)
            @show totloss, accuracy
            push!(lossplt, i, totloss)
            push!(accuracyplt, i, accuracy)
        end

        # add transformation data
        update!(chainplt)

        # update the heatmap of the total outgoing weight from each pixel
        pixel_importance = reshape(sum(t[1].params.views[1],1), 28, 28)
        # pixel_importance = reshape(abs(input_grad(t)),28,28)
        hmplt[1][1][:z].surf[:] = pixel_importance

        # handle animation frames/output
        if doanim
            lastframe = 5000
            if i < lastframe
                frame(anim)
            elseif i == lastframe
                gif(anim, fps=10)
            end
        end

        # display the plot
        gui()
        # if i>0
        #     @profile gui()
        # else
        #     gui()
        # end
    end, every=100)

    # trace once before we start learning to see initial values
    tracer.f(obj, 0)

    # end of plotting section
    # ---------------------------------------

    # create a gradient descent learner
    learner = make_learner(
        # averages the gradient over minibatches, updating params using Adadelta method
        GradientLearner(1e-2, Adam()),

        # our custom iteration method
        tracer,

        # shorthand to add a MaxIter(50000)
        maxiter = 50000
    )

    # do the learning... average over minibatches of size 5
    learn!(obj, learner, infinite_batches(train, size=5))

    # return these in case we want to analyze afterwards
    obj, learner
end

obj, learner = doit()

end #module
