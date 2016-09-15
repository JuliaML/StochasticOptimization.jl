
# NOTE: this all belongs in MLDataUtils.  It was originated as a PR to that package,
#   but I'm instead experimenting with the concepts here

export
    AbstractSubset,
    AbstractSubsets,
    DataSubset,
    MiniBatches

abstract AbstractSubset
abstract AbstractSubsets

# ----------------------------------------------------------------------------
# standard definitions of nobs and getobs for use with DataSubset

@generated function LearnBase.nobs(A::AbstractArray)
    T, N = A.parameters
    :(size(A, $N))
end

# apply a view to the last dimension
@generated function LearnBase.getobs(A::AbstractArray, idx)
    T, N = A.parameters
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

# add support for arbitrary tuples
LearnBase.nobs{T<:Tuple}(tup::T) = nobs(tup[1])
LearnBase.getobs{T<:Tuple}(tup::T, idx) = map(a -> getobs(a, idx), tup)

# specialized for empty tuples
LearnBase.nobs(tup::Tuple{}) = 0
LearnBase.getobs(tup::Tuple{}) = ()

# ----------------------------------------------------------------------------

"Lazy subsetting of source data, tracking the indices of observations in the source data."
immutable DataSubset{S,I<:AbstractVector{Int}} <: AbstractSubset
    source::S
    indices::I
end
DataSubset(subset::DataSubset, indices) = DataSubset(subset.source, subset.indices[indices])

Base.start(subset::DataSubset) = 1
Base.done(subset::DataSubset, i) = i > length(subset.indices)
Base.next(subset::DataSubset, i) = (getobs(subset.source, subset.indices[i]), i+1)
Base.length(subset::DataSubset) = length(subset.indices)

# ----------------------------------------------------------------------------

default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)

"An infinite-length iterator, producing a minibatch for source data (with replacement) at each iteration."
immutable MiniBatches{S} <: AbstractSubsets
    source::S
    batch_size::Int
end
MiniBatches(source) = MiniBatches(source, default_batch_size(source))

Base.start(mb::MiniBatches) = nothing
Base.done(mb::MiniBatches, i) = false
Base.next(mb::MiniBatches, i) = (DataSubset(mb.source, rand(1:nobs(mb.source), mb.batch_size)), nothing)


# ----------------------------------------------------------------------------

# convenience to iterate through source data
function eachobs(source)
    DataSubset(source, 1:nobs(source))
end
