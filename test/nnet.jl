module NnetTest

using Learn
# using StatPlots; glvisualize(leg=false, show=true)

function doit()

    nin = 10
    nout = 1
    τ = 1000
    w1 = randn(nout, nin)
    w2 = randn(nout, nin)
    # f(x) = tanh(w1*x + w2*sin(20x) + 0.05randn(nout))[1] > 0.0 ? 1.0 : 0.0
    f(x) = tanh(10w1*x)[1] > 0.0 ? 1.0 : 0.0
    inputs = randn(nin,τ)
    targets = zeros(nout,τ)
    for (i,x) in enumerate(DataSubset(inputs, 1:nobs(inputs)))
        targets[i] = f(x)
    end
    # scatter(inputs', targets', layout=nin, m=(3,0.2,stroke(0)), smooth=true)

    # build our model
    nh = 10
    chain = Chain(
        Affine(nin,nh),
        Activation(:relu,nh),
        Affine(nh,nout),
        Activation(:logistic,nout)
    )
    obj = objective(chain, CrossentropyLoss(), L1Penalty(1e-6))

    learner = make_learner(
        GradientLearner(FixedLR(1e-4), SGD()),
        maxiter = 10000,
        oniter = (model,i) -> begin
            @show i
        end
    )
    learn!(obj, learner, MiniBatches((inputs, targets), 5))

    # eval
    for (i,(input,target)) in enumerate(eachobs((inputs,targets)))
        @show i,transform!(chain,input),target
    end

    # store for inspection
    obj,learner
end
obj,learner = doit()
end #module
