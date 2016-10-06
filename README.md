# StochasticOptimization

[![Build Status](https://travis-ci.org/JuliaML/StochasticOptimization.jl.svg?branch=master)](https://travis-ci.org/JuliaML/StochasticOptimization.jl)

Utilizing the JuliaML ecosystem, StochasticOptimization is a framework for iteration-based optimizers.  Below is a complete example, from creating transformations, losses, penalties, and the combined objective function, to building custom sub-learners for the optimization, to constructing and running a stochastic gradient descent learner.

```julia
using StochasticOptimization
using ObjectiveFunctions
using CatViews
nin, nout = 10, 1

# Build our objective. Note this is LASSO regression.
t = Affine(nin,nout)
l = L2DistLoss()
p = L1Penalty(1e-8)
obj = RegularizedObjective(t, l, p)

# Create some fake data... affine transform plus noise
τ = 1000
w = randn(nout, nin)
b = randn(nout)
inputs = randn(nin, τ)
noise = 0.1rand(nout, τ)
targets = w * inputs + repmat(b, 1, τ) + noise

# Our core learning strategy... uses Adamax with a fixed learning rate
strat = GradientDescent(FixedLR(5e-3), Adamax())

# Create a view of w and b which looks like a single vector
θ = CatView(w,b)

# Check for convergence to the true parameter vector.
# This is an example of a custom convergence check.
θ_converge = ConvergenceFunction((model,i) -> begin
    if mod1(i,100) == 100
        normw = norm(θ - params(model))
        @show i,normw
        if normw < 0.1
            info("Converged after $i iterations: $normw")
            return true
        end
    end
    false
end)

# The MasterLearner has a bunch of specialized sub-learners.
learner = MasterLearner(
    strat,
    MaxIter(5000),
    θ_converge
)

# Note: Each sub-learner might only implement a subset of the iteration API:
#   pre_hook(learner, model)
#   learn!(model, learner, data)
#   iter_hook(learner, model, i)
#   finished(learner, model, i)
#   post_hook(learner, model)

# Everything is set up... learn the parameters
learn!(obj, learner, MiniBatches((inputs, targets), 20))
```
