__precompile__(true)

module StochasticOptimization

using Reexport
@reexport using LearnBase
@reexport using ObjectiveFunctions
# @reexport using MLDataUtils
using Parameters

import LearnBase: value, learn!, update!
# import OnlineStats: Diff, Mean, Variance, fit!, ExponentialWeight

export
    LearningStrategy,
    MasterLearner,
    MaxIter,
    GradientDescent,

    pre_hook,
    iter_hook,
    post_hook,
    finished,

    LearningRate,
    FixedLR,
    # AdaptiveLR,

    ParamUpdater,
    SGD

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
