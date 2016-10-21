__precompile__(true)

module StochasticOptimization

using Reexport
@reexport using LearnBase

import LearnBase: value, learn!, update!

export
    LearningStrategy,
    MetaLearner,
    make_learner,

    MaxIter,
    TimeLimit,
    ConvergenceFunction,
    IterFunction,
    ShowStatus,
    Tracer,
    Converged,
    ConvergedTo,

    GradientLearner,
    OnlineGradAvg,
    SearchDirection,
    GradientAverager,

    pre_hook,
    iter_hook,
    post_hook,
    finished,

    LearningRate,
    FixedLR,
    # AdaptiveLR,

    ParamUpdater,
    SGD,
    Adagrad,
    Adadelta,
    Adam,
    Adamax,
    RMSProp

include("iteration.jl")
using .Iteration

"Enacts a strategy to adjust the learning rate"
abstract LearningRate
include("learningrates.jl")

"An algorithm to update paramaters using a gradient (i.e. SGD, Adam, Adagrad, etc)"
abstract ParamUpdater
include("paramupdaters.jl")

"Holds optimizer state and parameters"
abstract LearningStrategy
include("strategies.jl")
include("gradients/gradients.jl")
include("gradients/online_gradients.jl")

# ---------------------------------------------------------------------------------

end # module
