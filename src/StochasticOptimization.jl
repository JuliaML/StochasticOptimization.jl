__precompile__(true)

module StochasticOptimization

using Compat: @compat
using Reexport
using LearnBase
import LearnBase: value, learn!, update!

@reexport using LearningStrategies
import LearningStrategies: pre_hook, iter_hook, finished, post_hook

export
    # LearningStrategy,
    # MetaLearner,
    # make_learner,
    #
    # MaxIter,
    # TimeLimit,
    # ConvergenceFunction,
    # IterFunction,
    # ShowStatus,
    # Tracer,
    # Converged,
    # ConvergedTo,

    GradientLearner,
    OnlineGradAvg,
    SearchDirection,
    GradientAverager,

    # pre_hook,
    # iter_hook,
    # post_hook,
    # finished,

    LearningRate,
    FixedLR,
    # AdaptiveLR,

    ParamUpdater,
    SGD,
    Adagrad,
    Adadelta,
    Adam,
    Adamax,
    RMSProp,

    @with

include("utils.jl")

# include("iteration.jl")
# using .Iteration

"Enacts a strategy to adjust the learning rate"
@compat abstract type LearningRate end
include("learningrates.jl")

"An algorithm to update paramaters using a gradient (i.e. SGD, Adam, Adagrad, etc)"
@compat abstract type ParamUpdater end
include("paramupdaters.jl")
include("gradients/gradients.jl")
include("gradients/online_gradients.jl")


# ---------------------------------------------------------------------------------

end # module
