using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using MLDataUtils

using ValueHistories
using Plots; unicodeplots(show=true)

@testset "LinReg" begin

    let nin=10, nout=1
        t = Affine{Float64}(nin,nout)
        l = L2DistLoss()
        p = L1Penalty(1e-8)
        obj = RegularizedObjective(t, l, p)

        τ = 10000
        w = randn(nout, nin)
        b = randn(nout)
        inputs = randn(nin, τ)
        targets = w * inputs + repmat(b, 1, τ) + 0.1randn(nout, τ)

        strat = GradientDescent(FixedLR(5e-3), SGD())
        @show strat

        data = batches(inputs, targets; batch_size = 100)
        @show length(data)
        tr = MVHistory()
        outputs = zeros(0)
        normw = normb = 0.0
        for i=1:1000
            learn!(obj, strat, data)
            normw = norm(value(t.w) - w)
            normb = norm(value(t.b) - b)
            @trace tr i normw normb
            @show i normw normb
        end
        # @test value(t.w) ≈ w
        # @test value(t.b) ≈ b

        plot(tr)

        pred = value(t.w) * inputs + repmat(value(t.b), 1, τ)
        truth = w * inputs + repmat(b, 1, τ)
        @test norm(pred - truth) < 1e-4
        plot(pred', truth', t=:scatter)
    end

end
