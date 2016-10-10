
# TODO: maybe we should get rid of the LearningRate abstraction, make it a number,
#   and then allow sub-learner(s) in the GradientLearner to update the learning rate?

immutable FixedLR <: LearningRate
    lr::Float64
end
value(lr::FixedLR) = lr.lr
update!(lr::FixedLR, err) = lr

# --------------------------------------------------------------------------

# "Adapts learning rate based on relative variance of the changes in the test error"
# @with_kw type AdaptiveLR <: LearningRate
#     lr::Float64 = 1e-2
#     ε::Diff{Float64} = Diff()
#     lookback::Int = 20
#     σ²::Variance{BoundedEqualWeight} = Variance(BoundedEqualWeight(lookback))
#     adjpct::Float64 = 1e-2
#     cutoff::Float64 = 1e-1
# end
#
#
# # if the error is decreasing at a large rate relative to the variance, increase the learning rate (speed it up)
# function update!(lr::AdaptiveLR, err)
#     fit!(lr.ε, err)
#     fit!(lr.σ², diff(lr.ε))
#     μ = mean(lr.σ²)
#     σ = std(lr.σ²)
#     if σ > 0
#         pct = lr.adjpct * (μ / σ < -lr.cutoff ? 1.0 : -1.0)
#         lr.lr *= (1.0 + pct)
#     end
#     lr
# end
