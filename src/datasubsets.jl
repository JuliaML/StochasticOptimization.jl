
module Iteration

# NOTE: this all belongs in MLDataUtils.  It was originated as a PR to that package,
#   but I'm instead experimenting with the concepts here

# Here's the API, as copied from https://github.com/JuliaML/MLDataUtils.jl/issues/3#issuecomment-247786693

# X = rand(2,100) # input matrix
# y = rand(100)  # target vector
#
# for x in eachobs(X)
#     # x = view(X,:,i)
# end
#
# for yi in eachobs(y)
#     # yi = y[i]
# end
#
# ### iterate one observation at a time: obs = (x, yi)
#
# for obs in eachobs(X,y)
#     # obs = (view(X,:,i), y[i])
# end
#
# for obs in eachobs(shuffled(X,y))
#     # obs = (view(X,:,i), y[i])  where i comes from a shuffled indices
# end
#
# for obs in sample_forever(X,y)
#     # infinitely sample one observation
# end
#
# ### iterate one batch/partition at a time: batch = (batch_x, batch_y)
#
# # one pass through the data in chunks of 10
# for batch in batches(X, y, size=10)
#     # batch = (view(X, :, i:i+10), view(y, i:i+10))
#     # we could also do "for (x,yi) in ..."
# end
#
# for batch in batches(shuffled(X,y), size=10)
#     # same, but the indices are shuffled first
# end
#
# # a float for size would imply a fractional split.
# # (I'm pretty sure this notation works if batches returns a length-2 iterator)
# train, test = batches(X, y, size = 0.7)
#
# # this would work too, since train and test are tuples of views
# (train_x, train_y), (test_x, test_y) = batches(X, y, size = 0.7)
#
# # a tuple or vector of floats could give more than 2 batches:
# train, validate, test = batches(X, y, size = (0.6, 0.2))
#
# for batch in partition_forever(X,y, size=10)
#     # infinitely sample a random batch
# end
#
# ### iterators over partition-iterators
#
# for (train, test) in kfolds(X, y, k=10)
#     # train = (train_x, train_y)
# end
#
# for (train, test) in leave_one_out(X, y)
#     ...
# end
#
# ### utils
#
# # lazy filter on index
# newX, newy = filterobs(i -> y[i] < 2, X, y)

# TODO: do we even need AbstractSubset/AbstractSubsets?

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

import Iterators: repeatedly, repeated
import LearnBase: nobs, getobs

export
    DataIterator,
        ObsIterator,        # each iteration is an observation T
            EachObs,
            DataSubset,
            InfiniteObs,
        BatchIterator,      # each iteration is a ObsIterator{T}
            EachBatch,
            Batches,
            InfiniteBatches,
        BatchesIterator,    # each iteration is a BatchIterator{T}
            KFolds,

    eachobs,
    shuffled,
    infinite_obs,
    batches,
    infinite_batches,
    repeatedly,
    repeated,
    kfolds,
    leave_one_out,
    filterobs

abstract DataIterator{T} <: AbstractVector{T}
    abstract ObsIterator{T} <: DataIterator{T}
    abstract BatchIterator{T} <: DataIterator{T}
    abstract BatchesIterator{T} <: DataIterator{T}

# ----------------------------------------------------------------------------
# standard definitions of nobs and getobs for use with DataSubset

@generated function nobs(A::AbstractArray)
    T, N = A.parameters
    :(size(A, $N))
end

# apply a view to the last dimension
@generated function getobs(A::AbstractArray, idx)
    T, N = A.parameters
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    else
        :(view(A,  $(fill(:(:),N-1)...), idx))
    end
end

# add support for arbitrary tuples
nobs{T<:Tuple}(tup::T) = nobs(tup[1])
getobs{T<:Tuple}(tup::T, idx) = map(a -> getobs(a, idx), tup)

# specialized for empty tuples
nobs(tup::Tuple{}) = 0
getobs(tup::Tuple{}) = ()


default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)

# ----------------------------------------------------------------------------

immutable EachObs{S,T} <: ObsIterator{T}
    source::S
end

Base.length(itr::EachObs) = nobs(itr.source)
Base.start(itr::EachObs) =

# ----------------------------------------------------------------------------

"Lazy subsetting of source data, tracking the indices of observations in the source data."
immutable DataSubset{S,I<:AbstractVector{Int}} <: AbstractSubset
    source::S
    indices::I
end
DataSubset(source) = DataSubset(source, 1:nobs(source))
DataSubset(subset::DataSubset, indices = 1:nobs(subset)) = DataSubset(subset.source, subset.indices[indices])

Base.get(subset::DataSubset) = getobs(subset.source, subset.indices)

Base.length(subset::DataSubset) = length(subset.indices)
Base.size(subset::DataSubset) = size(subset.indices)
nobs(subset::DataSubset) = length(subset.indices)

Base.getindex(subset::DataSubset, idx) = getobs(subset.source, subset.indices[idx])
getobs(subset::DataSubset, idx) = getobs(subset.source, subset.indices[idx])

Base.start(subset::DataSubset) = 1
Base.done(subset::DataSubset, i) = i > length(subset.indices)
Base.next(subset::DataSubset, i) = (getobs(subset.source, subset.indices[i]), i+1)

Base.rand(subset::DataSubset, args...) = getobs(subset.source, rand(subset.indices, args...))

Base.collect(subset::DataSubset) = collect(getobs(subset.source, subset.indices))
Base.collect{S<:Tuple}(subset::DataSubset{S}) = map(collect, getobs(subset.source, subset.indices))

# ----------------------------------------------------------------------------

# call with a tuple for more than one arg
for f in [:eachobs, :shuffled, :infinite_obs]
    @eval $f(s_1, s_2, s_rest...) = $f((s_1, s_2, s_rest...))
end

"""
Iterate over source data.

```julia
for (x,y) in eachobs(X,Y)
    ...
end
```
"""
eachobs(source) = DataSubset(source)

"""
Iterate over shuffled (randomized) source data.  This is non-copy and non-mutating (only the indices are shuffled).

```julia
for (x,y) in shuffled(X,Y)
    ...
end
```
"""
shuffled(source) = DataSubset(source, shuffle(1:nobs(source)))

"""
Infinitely return a random observation.

```julia
for (x,y) in infinite_obs(X,Y)
    ...
end
```
"""
infinite_obs(source) = repeatedly(() -> rand(DataSubset(source)))

"""
Non-copy, non-mutating filter over observation indices.

```julia
for (x, yi) in filterobs(i -> y[i] < 2, X, y)
    ...
end
```
"""
filterobs(f::Function, subset::DataSubset) = getobs(subset.source, filter(f, subset.indices))
filterobs(f::Function, source) = getobs(source, filter(f, 1:nobs(source)))
filterobs(f::Function, s_1, s_2, s_rest...) = filterobs(f, (s_1, s_2, s_rest...))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

"An explicit wrapper around an vector of DataSubset"
immutable DataSubsets{D<:DataSubset} <: AbstractSubsets
    subsets::Vector{D}
end

Base.getindex(subsets::DataSubsets, i::Int) = get(subsets.subsets[i])
Base.start(subsets::DataSubsets) = 1
Base.done(subsets::DataSubsets, i) = i > length(subsets.subsets)
Base.next(subsets::DataSubsets, i) = (get(subsets.subsets[i]), i+1)
Base.length(subsets::DataSubsets) = length(subsets.subsets)
Base.size(subsets::DataSubsets) = size(subsets.subsets)
Base.rand(subsets::DataSubsets) = rand(subsets.subsets)
Base.collect(subsets::DataSubsets) = map(collect, subsets.subsets)

# ----------------------------------------------------------------------------

"""
Split the data apart, either by specifying a size or giving a percentage split point.

```julia
# split into training and test sets, 60%/40% respectively
train, test = batches(X, Y, size = 0.6)

# split into equal-sized minibatches of 10 observations each
for batch in batches(X, Y, size = 10)
    ...
end

# Tips:
#   - Iterators can be nested
#   - Observations can be extracted immediately
for (x,y) in batches(shuffled(X, Y), size = 10)
    ...
end
```
"""
function batches(subset::DataSubset; size = default_batch_size(subset.source))
    n = nobs(subset)
    T = typeof(size)
    idx_list = if T <: AbstractFloat
        # partition into 2 sets
        n1 = clamp(round(Int, size*n), 1, n)
        [1:n1, n1+1:n]
    elseif (T <: NTuple || T <: AbstractVector) && eltype(T) <: AbstractFloat
        nleft = n
        lst = []
        for (i,sz) in enumerate(size)
            ni = clamp(round(Int, sz*n), 0, nleft)
            push!(lst, n-nleft+1:n-nleft+ni)
            nleft -= ni
        end
        push!(lst, n-nleft+1:n)
        lst
    elseif T <: Integer
        offset = 0
        lst = []
        while offset < n
            sz = clamp(n - offset, 1, size)
            push!(lst, offset+1:offset+sz)
            offset += sz
        end
        lst
    end
    # @show idx_list
    subsets = typeof(subset)[DataSubset(subset, idx) for idx in idx_list]
    DataSubsets(subsets)
end

"""
Sample a random minibatch (with replacement) repeatedly forever.

```julia
for batch in infinite_batches(X, Y, size = 10)
    ...
end
```
"""
function infinite_batches(subset::DataSubset; size = default_batch_size(subset.source))
    repeatedly(() -> DataSubset(subset.source, rand(1:nobs(subset.source), size)))
end


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Iterators of DataSubsets

immutable KFolds{S}
    subset::DataSubset{S}
    k::Int
end

fold_count(kf::KFolds) = div(nobs(kf.subset), kf.k)
start_index(kf::KFolds, i::Int) = clamp((i-1) * fold_count(kf) + 1, 1, nobs(kf.subset))
end_index(kf::KFolds, i::Int) = clamp(i * fold_count(kf), 1, nobs(kf.subset))

function Base.getindex(kf::KFolds, idx)
    test_idx = start_index(kf,idx):end_index(kf,idx)
    train_idx = setdiff(1:nobs(kf.subset), test_idx)
    # @show train_idx, test_idx
    kf.subset[train_idx], kf.subset[test_idx]
end
Base.start(kf::KFolds) = 1
Base.done(kf::KFolds, i) = i > kf.k
Base.next(kf::KFolds, i) = (kf[i], i+1)
Base.length(kf::KFolds) = kf.k
Base.size(kf::KFolds) = (kf.k,)


"""
K-Folds validation.  Iterate over k pairs of (train,test) splits, where each test set has approximately nobs/k observations.

```julia
for (train, test) in kfolds(X, y, k=10)
    ...
end
```
"""
kfolds(subset::DataSubset; k::Int = 5) = KFolds(subset, k)


"""
Leave-one-out validation.  K-Folds where k == nobs.

```julia
for (train, test) in leave_one_out(X, y)
    ...
end
```
"""
leave_one_out(subset::DataSubset) = KFolds(subset, nobs(subset))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# generic method calls for anything that's not a DataSubset
for f in [:batches, :infinite_batches, :kfolds, :leave_one_out]
    @eval begin
        $f(source; kw...) = $f(DataSubset(source); kw...)
        $f(source...; kw...) = $f(DataSubset(source); kw...)
    end
end

end # module Iteration
