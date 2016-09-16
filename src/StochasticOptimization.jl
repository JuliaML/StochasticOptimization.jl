__precompile__(true)

module StochasticOptimization

using Reexport
@reexport using LearnBase
@reexport using ObjectiveFunctions
using Parameters

import LearnBase: value, learn!, update!

export
    LearningStrategy,
    MasterLearner,
    MaxIter,
    ConvergenceFunction,
    IterFunction,
    GradientDescent,
    make_learner,

    forever,
    eachobs,

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

include("datasubsets.jl")

"Enacts a strategy to adjust the learning rate"
abstract LearningRate
include("learningrates.jl")

"An algorithm to update paramaters using a gradient (i.e. SGD, Adam, Adagrad, etc)"
abstract ParamUpdater
include("paramupdaters.jl")

"Holds optimizer state and parameters"
abstract LearningStrategy
include("strategies.jl")

# ---------------------------------------------------------------------------------

end # module
