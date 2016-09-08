nfeatures, ntargets = 2, 3
τ = 100
X = rand(nfeatures, τ) # feature matrix: nfeatures x nobs
Y = rand(ntargets, τ) # target matrix: ntargets x nobs
model = Affine{Float64}(nfeatures, ntargets) # some learnable Transformation?
strategy = SGD(model) # the spec and state of a learning algorithm


# loop through batches checking for early stopping after each batch
function learn!(model, strategy, data::BatchIterator)
    pre_hook(model, strategy)
    for batch in data
        # update the params for this batch
        learn!(model, strategy, batch)

        iter_hook(model, strategy)
        finished(model, strategy) && break
    end
    post_hook(model, strategy)
end

function learn!(model, strategy, batch::Batch)
    θ = params(model)
    ∇ = grad(model)
    ∇avg = zeros(θ)
    scalar = 1 / length(batch)
    for (input,target) in batch
        # forward and backward passes for this datapoint
        transform!(model, target, input)
        grad!(model)

        # add to the total param change for this strategy/gradient
        for (i,j) in zip(eachindex(∇avg), eachindex(∇))
            ∇avg[i] += ∇[j] * scalar
        end
    end

    # update the params using the average gradient
    update!(θ, strategy, ∇avg)
end

# NOTE: a strategy would have to implement update! and finished... the rest is optional


# do a kfolds training

scores = []
for (train, validate) in kfolds(X, Y, k = 5, batchsize = 20)
    # here train and validate are type BatchIterator
    learn!(model, strategy, train)

    # add the score (accuracy? loss?)
    push!(scores, score(model, validate))
end
info("Accuracy: $(mean(scores)) (+/- $(std(scores)))")
