using StochasticOptimization
using Base.Test

using ObjectiveFunctions
using MLDataUtils

@testset "LinReg" begin

    let nin=2, nout=3
        t = Affine{Float64}(nin,nout)
        l = L2DistLoss()
        obj = RegularizedObjective(t, l)

        τ = 100
        w = rand(nout, nin)
        b = rand(nout)
        inputs = randn(nin, τ)
        targets = w * inputs + b + randn(nout, τ)

        strat = SGD(length(params(obj)))
        @show strat

        learn!(strat, obj)
        @test value(t.w) ≈ w
        @test value(t.b) ≈ b
    end

end
