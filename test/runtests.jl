using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using MLDataUtils

using ValueHistories
using Plots; unicodeplots(show=true)

@testset "LinReg" begin

    let nin=10, nout=1
        # build our objective
        t = Affine{Float64}(nin,nout)
        l = L2DistLoss()
        p = L1Penalty(1e-8)
        obj = RegularizedObjective(t, l, p)

        # create some fake data... affine transform plus noise
        τ = 1000
        w = randn(nout, nin)
        b = randn(nout)
        inputs = randn(nin, τ)
        targets = w * inputs + repmat(b, 1, τ) + 0.1randn(nout, τ)

        # our learning strategy... SGD with a fixed learning rate
        strat = GradientDescent(FixedLR(5e-3), SGD())
        @show strat

        # trace setup
        tr = MVHistory()
        normw = normb = 0.0

        # do a bunch of epochs and trace/show in between
        for i=1:100
            # get new random batches, then learn from them
            data = batches(inputs, targets; batch_size = 100)
            learn!(obj, strat, data)

            # trace and show
            # TODO: make this a LearningStrategy which implements post_hook
            normw = norm(value(t.w) - w)
            normb = norm(value(t.b) - b)
            @trace tr i normw normb
            @show i, normw, normb
        end

        # show the trace we collected
        plot(tr)

        # scatter predicted output vs ground truth... should be diagonal line
        pred = value(t.w) * inputs + repmat(value(t.b), 1, τ)
        truth = w * inputs + repmat(b, 1, τ)
        @test maximum(pred - truth) < 1e-1
        plot(pred', truth', t=:scatter)
    end

end
