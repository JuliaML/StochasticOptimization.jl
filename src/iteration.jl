
module Iteration

# NOTE: this all belongs in MLDataUtils.  It was originated as a PR to that package,
#   but I'm instead experimenting with the concepts here

# Here's the API, as copied from https://github.com/JuliaML/MLDataUtils.jl/issues/3#issuecomment-247786693

# X = rand(2,100) # input matrix
# y = rand(100)  # target vector
#
# for x in each_obs(X)
#     # x = view(X,:,i)
# end
#
# for yi in each_obs(y)
#     # yi = y[i]
# end
#
# ### iterate one observation at a time: obs = (x, yi)
#
# for obs in each_obs(X,y)
#     # obs = (view(X,:,i), y[i])
# end
#
# for obs in each_obs(shuffled_obs(X,y))
#     # obs = (view(X,:,i), y[i])  where i comes from a shuffled_obs indices
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
# for batch in batches(shuffled_obs(X,y), size=10)
#     # same, but the indices are shuffled_obs first
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
# newX, newy = filtered_obs(i -> y[i] < 2, X, y)

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
            SubsetObs,
            InfiniteObs,
        BatchIterator,      # each iteration is a ObsIterator{T}
            EachBatch,
            Batches,
            InfiniteBatches,
        BatchesIterator,    # each iteration is a BatchIterator{T}
            KFolds,

    each_obs,     # in-order
    subset_obs,   # like SubArray
    shuffled_obs, # subset_obs(source, shuffle(1:nobs(source)))
    infinite_obs, #
    filtered_obs,
    random_obs,
    each_batch,
    batches,
    infinite_batches,
    split_obs,
    repeatedly,
    repeated,
    kfolds,
    leave_one_out

abstract DataIterator{T} <: AbstractVector{T}
    abstract ObsIterator{T} <: DataIterator{T}
    abstract BatchIterator{T} <: DataIterator{T}
    abstract BatchesIterator{T} <: DataIterator{T}

# ----------------------------------------------------------------------------
# standard definitions of nobs and getobs for use with SubsetObs

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
nobs(tup::Tuple) = nobs(tup[1])
getobs(tup::Tuple, idx) = map(a -> getobs(a, idx), tup)
Base.collect(tup::Tuple) = map(collect, tup)

# specialized for empty tuples
nobs(tup::Tuple{}) = 0
getobs(tup::Tuple{}) = ()

obstype(source) = typeof(getobs(source, 1))
all_indices(source) = 1:nobs(source)

Base.size(itr::DataIterator) = (length(itr),)

# define the degenerate case so we can do infinite_obs(nothing) and get nothing forever
nobs(::Void) = 1
getobs(::Void, idx) = nothing

# ----------------------------------------------------------------------------
# ObsIterator
# ----------------------------------------------------------------------------

# default implementations
nobs(itr::ObsIterator) = nobs(itr.source)
getobs(itr::ObsIterator, idx) = getobs(itr.source, all_indices(itr)[idx])

Base.rand(itr::ObsIterator, dims::Integer...) = getobs(itr.source, rand(all_indices(itr), dims...))
Base.length(itr::ObsIterator) = length(all_indices(itr))
Base.getindex(itr::ObsIterator, idx) = getobs(itr, idx)

# views
Base.get(itr::ObsIterator) = getobs(itr.source, all_indices(itr))

# copies
Base.collect(itr::ObsIterator) = collect(get(itr))

# ----------------------------------------------------------------------------

"""
Iterate over source data in-order.

```julia
for (x,y) in each_obs(X,Y)
    ...
end
```
"""
immutable EachObs{S,T} <: ObsIterator{T}
    source::S
end
EachObs{S}(source::S) = EachObs{S, obstype(source)}(source)
const each_obs = EachObs

Base.start(itr::EachObs) = 1
Base.done(itr::EachObs, i) = i > length(itr)
Base.next(itr::EachObs, i) = (getobs(itr.source, i), i+1)

# ----------------------------------------------------------------------------

"Lazy subsetting of source data, tracking the indices of observations in the source data."
immutable SubsetObs{S,T,I<:AbstractVector{Int}} <: ObsIterator{T}
    source::S
    indices::I
end
SubsetObs(source) = SubsetObs(source, 1:nobs(source))
function SubsetObs{S,I}(source::S, indices::I)
    SubsetObs{S,obstype(source),I}(source, indices)
end
# TODO: allow lazily composing subsets
# SubsetObs(itr::SubsetObs, indices = 1:nobs(itr)) = SubsetObs(itr.source, itr.indices[indices])
const subset_obs = SubsetObs

nobs(itr::SubsetObs) = length(itr.indices)
all_indices(itr::SubsetObs) = itr.indices

Base.start(itr::SubsetObs) = 1
Base.done(itr::SubsetObs, i) = i > length(itr.indices)
Base.next(itr::SubsetObs, i) = (getobs(itr.source, itr.indices[i]), i+1)

# ----------------------------------------------------------------------------

"""
Infinitely return a random observation.

```julia
for (x,y) in infinite_obs(X,Y)
    ...
end
```
"""
immutable InfiniteObs{S,T} <: ObsIterator{T}
    source::S
end
InfiniteObs{S}(source::S) = InfiniteObs{S, obstype(source)}(source)
const infinite_obs = InfiniteObs

Base.length(itr::InfiniteObs) = Inf
Base.start(itr::InfiniteObs) = 1
Base.done(itr::InfiniteObs, i) = false
Base.next(itr::InfiniteObs, i) = (rand(itr), i+1)

# ----------------------------------------------------------------------------

# call with a tuple for more than one arg
for f in [:each_obs, :subset_obs, :infinite_obs, :shuffled_obs, :filtered_obs]
    @eval $f(s_1, s_2, s_rest...) = $f((s_1, s_2, s_rest...))
end


"""
Iterate over shuffled_obs (randomized) source data.  This is non-copy and non-mutating (only the indices are shuffled_obs).

```julia
for (x,y) in shuffled_obs(X,Y)
    ...
end
```
"""
shuffled_obs(source) = SubsetObs(source, shuffle(1:nobs(source)))

"""
Non-copy, non-mutating filter over observation indices.

```julia
for (x, yi) in filtered_obs(i -> y[i] < 2, X, y)
    ...
end
```
"""
filtered_obs(f::Function, itr::SubsetObs) = SubsetObs(itr.source, filter(f, itr.indices))
filtered_obs(f::Function, source) = SubsetObs(source, filter(f, 1:nobs(source)))
filtered_obs(f::Function, s_1, s_2, s_rest...) = filtered_obs(f, (s_1, s_2, s_rest...))

# ----------------------------------------------------------------------------
# BatchIterator
# ----------------------------------------------------------------------------

function default_batch_size(source)
    clamp(div(nobs(source), 5), 1, 100)
end

function default_batch_indices(source)
    batchsize = default_batch_size(source)
    n = nobs(source)
    offset = 0
    lst = []
    while offset < n
        sz = clamp(n - offset, 1, batchsize)
        push!(lst, offset+1:offset+sz)
        offset += sz
    end
    lst
end

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1)
    num_observations = nobs(source)::Int
    @assert num_observations > 0
    size  <= num_observations || throw(BoundsError(source,size))
    count <= num_observations || throw(BoundsError(source,count))
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_batch_size(source)::Int
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. uses all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # try to use both (usually to use a subset of the observations)
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(DimensionMismatch("Specified number of partitions is not possible with specified size"))
    end

    # check if the settings will result in all data points being used
    unused = num_observations % size
    if unused > 0
        info("The specified values for size and/or count will result in $unused unused data points")
    end
    size::Int, count::Int
end

# ----------------------------------------------------------------------------

immutable EachBatch{S,T} <: BatchIterator{T}
    source::S
    batchsize::Int
    batchcount::Int
end
function EachBatch{S}(source::S; size=-1, count=-1)
    batchsize, batchcount = _compute_batch_settings(source, size, count)
    EachBatch{S, SubsetObs{S,SubsetObs{S,obstype(source)}}}(source, batchsize, batchcount)
end
const each_batch = EachBatch

Base.start(itr::EachBatch) = 1:itr.batchsize
Base.done(itr::EachBatch, i) = maximum(i) > nobs(itr.source) #i > itr.batchcount || (i * itr.batchsize)-1 > nobs(itr.source)
Base.next(itr::EachBatch, i) = (subset_obs(itr.source, i), i+itr.batchsize)
Base.getindex(itr::EachBatch, i::Integer) = subset_obs(itr.source, (1:itr.batchsize)+(i-1))
Base.length(itr::EachBatch) = itr.batchcount

# ----------------------------------------------------------------------------

immutable Batches{S,T,I} <: BatchIterator{T}
    source::S
    batch_indices::I
end
# Batches(source; indices = default_batch_indices(source)) = Batches(source, indices)
function Batches{S,I}(source::S, indices::I)
    Batches{S, SubsetObs{S,obstype(source)}, I}(source, indices)
end
batches(source; indices = default_batch_indices(source)) = Batches(source, indices)
# const batches = Batches

Base.start(itr::Batches) = 1
Base.done(itr::Batches, i) = done(itr.batch_indices, i)
function Base.next(itr::Batches, i)
    v, nexti = next(itr.batch_indices,i)
    subset_obs(itr.source, v), nexti
end
Base.getindex(itr::Batches, i::Integer) = subset_obs(itr.source, itr.batch_indices[i])
Base.length(itr::Batches) = length(itr.batch_indices)

# ----------------------------------------------------------------------------

immutable InfiniteBatches{S,T} <: BatchIterator{T}
    source::S
    batchsize::Int
end
function InfiniteBatches{S}(source::S; size = default_batch_size(source))
    InfiniteBatches{S, SubsetObs{S,obstype(source)}}(source, size)
end
const infinite_batches = InfiniteBatches

Base.length(itr::InfiniteBatches) = Inf
Base.start(itr::InfiniteBatches) = 1
Base.done(itr::InfiniteBatches, i) = false
Base.next(itr::InfiniteBatches, i::Integer) = (itr[1], 1)
Base.getindex(itr::InfiniteBatches, i) = subset_obs(itr.source, rand(all_indices(itr.source), itr.batchsize))

# ----------------------------------------------------------------------------


# "An explicit wrapper around an vector of SubsetObs"
# immutable DataSubsets{D<:SubsetObs} <: AbstractSubsets
#     subsets::Vector{D}
# end
#
# Base.getindex(subsets::DataSubsets, i::Int) = get(subsets.subsets[i])
# Base.start(subsets::DataSubsets) = 1
# Base.done(subsets::DataSubsets, i) = i > length(subsets.subsets)
# Base.next(subsets::DataSubsets, i) = (get(subsets.subsets[i]), i+1)
# Base.length(subsets::DataSubsets) = length(subsets.subsets)
# Base.size(subsets::DataSubsets) = size(subsets.subsets)
# Base.rand(subsets::DataSubsets) = rand(subsets.subsets)
# Base.collect(subsets::DataSubsets) = map(collect, subsets.subsets)
#
# # ----------------------------------------------------------------------------
#
# default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)
#
# """
# Split the data apart, either by specifying a size or giving a percentage split point.
#
# ```julia
# # split into training and test sets, 60%/40% respectively
# train, test = batches(X, Y, size = 0.6)
#
# # split into equal-sized minibatches of 10 observations each
# for batch in batches(X, Y, size = 10)
#     ...
# end
#
# # Tips:
# #   - Iterators can be nested
# #   - Observations can be extracted immediately
# for (x,y) in (shuffled_obs(X, Y), size = 10)
#     ...
# end
# ```
# """

function split_obs(source; at = 0.7)
    n = nobs(source)
    T = typeof(at)
    idx_list = if T <: AbstractFloat
        # partition into 2 sets
        n1 = clamp(round(Int, at*n), 1, n)
        [1:n1, n1+1:n]
    elseif (T <: NTuple || T <: AbstractVector) && eltype(T) <: AbstractFloat
        nleft = n
        lst = []
        for (i,sz) in enumerate(at)
            ni = clamp(round(Int, sz*n), 0, nleft)
            push!(lst, n-nleft+1:n-nleft+ni)
            nleft -= ni
        end
        push!(lst, n-nleft+1:n)
        lst
    else
        throw(ArgumentError("Expecting a float or tuple/vector of floats for `at` in split_obs.  Got: $T"))
    # elseif T <: Integer
    #     offset = 0
    #     lst = []
    #     while offset < n
    #         sz = clamp(n - offset, 1, size)
    #         push!(lst, offset+1:offset+sz)
    #         offset += sz
    #     end
    #     lst
    end
    batches(source, indices = idx_list)
    # @show idx_list
    # subsets = typeof(itr)[SubsetObs(itr, idx) for idx in idx_list]
    # DataSubsets(subsets)
end

# """
# Sample a random minibatch (with replacement) repeatedly forever.
#
# ```julia
# for batch in infinite_batches(X, Y, size = 10)
#     ...
# end
# ```
# """
# function infinite_batches(itr::SubsetObs; size = default_batch_size(itr.source))
#     repeatedly(() -> SubsetObs(itr.source, rand(1:nobs(itr.source), size)))
# end


# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------
# # Iterators of DataSubsets
#
# immutable KFolds{S}
#     itr::SubsetObs{S}
#     k::Int
# end
#
# fold_count(kf::KFolds) = div(nobs(kf.itr), kf.k)
# start_index(kf::KFolds, i::Int) = clamp((i-1) * fold_count(kf) + 1, 1, nobs(kf.itr))
# end_index(kf::KFolds, i::Int) = clamp(i * fold_count(kf), 1, nobs(kf.itr))
#
# function Base.getindex(kf::KFolds, idx)
#     test_idx = start_index(kf,idx):end_index(kf,idx)
#     train_idx = setdiff(1:nobs(kf.itr), test_idx)
#     # @show train_idx, test_idx
#     kf.itr[train_idx], kf.itr[test_idx]
# end
# Base.start(kf::KFolds) = 1
# Base.done(kf::KFolds, i) = i > kf.k
# Base.next(kf::KFolds, i) = (kf[i], i+1)
# Base.length(kf::KFolds) = kf.k
# Base.size(kf::KFolds) = (kf.k,)
#
#
# """
# K-Folds validation.  Iterate over k pairs of (train,test) splits, where each test set has approximately nobs/k observations.
#
# ```julia
# for (train, test) in kfolds(X, y, k=10)
#     ...
# end
# ```
# """
# kfolds(itr::SubsetObs; k::Int = 5) = KFolds(itr, k)
#
#
# """
# Leave-one-out validation.  K-Folds where k == nobs.
#
# ```julia
# for (train, test) in leave_one_out(X, y)
#     ...
# end
# ```
# """
# leave_one_out(itr::SubsetObs) = KFolds(itr, nobs(itr))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# generic method calls for anything that's not a SubsetObs
for f in [:each_batch, :batches, :infinite_batches, :split_obs] #, :kfolds, :leave_one_out]
    @eval begin
        # $f(source; kw...) = $f(source; kw...)
        $f(source...; kw...) = $f(source; kw...)
    end
end

end # module Iteration
