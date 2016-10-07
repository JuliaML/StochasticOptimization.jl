# StochasticOptimization

[![Build Status](https://travis-ci.org/JuliaML/StochasticOptimization.jl.svg?branch=master)](https://travis-ci.org/JuliaML/StochasticOptimization.jl)

Utilizing the JuliaML ecosystem, StochasticOptimization is a framework for iteration-based optimizers.  Below is a complete example, from creating transformations, losses, penalties, and the combined objective function, to building custom sub-learners for the optimization, to constructing and running a stochastic gradient descent learner.

```julia
using StochasticOptimization
using ObjectiveFunctions
using CatViews

# Build our objective. Note this is LASSO regression.
# The objective method constucts a RegularizedObjective composed
#   of a Transformation, a Loss, and an optional Penalty.
nin, nout = 10, 1
obj = objective(
    Affine(nin,nout),
    L2DistLoss(),
    L1Penalty(1e-8)
)

# Create some fake data... affine transform plus noise
τ = 1000
w = randn(nout, nin)
b = randn(nout)
inputs = randn(nin, τ)
noise = 0.1rand(nout, τ)
targets = w * inputs + repmat(b, 1, τ) + noise

# Create a view of w and b which looks like a single vector
θ = CatView(w,b)

# The MetaLearner has a bunch of specialized sub-learners.
# Our core learning strategy is Adamax with a fixed learning rate.
# The `maxiter` and `converged` keywords will add `MaxIter`
#   and `ConvergenceFunction` sub-learners to the MetaLearner.
learner = make_learner(
    GradientLearner(5e-3, Adamax()),
    maxiter = 5000,
    converged = (model,i) -> begin
        if mod1(i,100) == 100
            if norm(θ - params(model)) < 0.1
                info("Converged after $i iterations")
                return true
            end
        end
        false
    end
)

# Everything is set up... learn the parameters by iterating through
#   random minibatches forever until convergence, or until the max iterations.
learn!(obj, learner, infinite_batches(inputs, targets, size=20))
```

With any luck, you'll see something like:

```
INFO: Converged after 800 iterations
```

### Notes:

Each sub-learner might only implement a subset of the iteration API:
- `pre_hook(learner, model)`
- `learn!(model, learner, data)`
- `iter_hook(learner, model, i)`
- `finished(learner, model, i)`
- `post_hook(learner, model)`
