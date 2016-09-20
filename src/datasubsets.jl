
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
# for batch in splitdata(X, y, size=10)
#     # batch = (view(X, :, i:i+10), view(y, i:i+10))
#     # we could also do "for (x,yi) in ..."
# end
#
# for batch in splitdata(shuffled(X,y), size=10)
#     # same, but the indices are shuffled first
# end
#
# # a float for size would imply a fractional split.
# # (I'm pretty sure this notation works if splitdata returns a length-2 iterator)
# train, test = splitdata(X, y, size = 0.7)
#
# # this would work too, since train and test are tuples of views
# (train_x, train_y), (test_x, test_y) = splitdata(X, y, size = 0.7)
#
# # a tuple or vector of floats could give more than 2 splitdata:
# train, validate, test = splitdata(X, y, size = (0.6, 0.2))
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
    AbstractSubset,
    AbstractSubsets,
    DataSubset,
    DataSubsets,
    # RandomMiniBatches,

    eachobs,
    shuffled,
    splitdata,
    infinite_obs,
    infinite_batches,
    repeatedly,
    repeated

abstract AbstractSubset
abstract AbstractSubsets

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


# ----------------------------------------------------------------------------

"Lazy subsetting of source data, tracking the indices of observations in the source data."
immutable DataSubset{S,I<:AbstractVector{Int}} <: AbstractSubset
    source::S
    indices::I
end
DataSubset(source) = DataSubset(source, 1:nobs(source))
DataSubset(subset::DataSubset, indices = 1:nobs(subset)) = DataSubset(subset.source, subset.indices[indices])

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

# "Infinite sampler of the source data"
# immutable InfiniteSampler{S} <: AbstractSubset
#     source::S
#     idxfunc::Function  # call this to generate idx... method should take source as input
# end
#
# Base.rand(infsamp::InfiniteSampler) = DataSubset(infsamp.source, infsamp.idxfunc(infsamp.source))
# Base.getindex(infsamp::InfiniteSampler, idx::Integer) = rand(infsamp)
# Base.getindex(infsamp::InfiniteSampler, idx::AbstractArray) = rand(infsamp, size(idx)...)
# Base.done(subsets::InfiniteSampler)

# ----------------------------------------------------------------------------

"An explicit wrapper around an vector of DataSubset"
immutable DataSubsets{D<:DataSubset} <: AbstractSubsets
    subsets::Vector{D}
end

Base.getindex(subsets::DataSubsets, idx) = subsets.subsets[idx]
Base.start(subsets::DataSubsets) = start(subsets.subsets)
Base.done(subsets::DataSubsets, i) = done(subsets.subsets)
Base.next(subsets::DataSubsets, i) = next(subsets.subsets)
Base.length(subsets::DataSubsets) = length(subsets.subsets)
Base.size(subsets::DataSubsets) = size(subsets.subsets)
Base.rand(subsets::DataSubsets) = rand(subsets.subsets)
Base.collect(subsets::DataSubsets) = map(collect, subsets.subsets)

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

default_batch_size(source) = clamp(div(nobs(source), 5), 1, 100)

# "An infinite-length iterator, producing a minibatch for source data (with replacement) at each iteration."
# immutable RandomMiniBatches{S} <: AbstractSubsets
#     source::S
#     batch_size::Int
# end
# RandomMiniBatches(source) = RandomMiniBatches(source, default_batch_size(source))
#
# Base.start(mb::RandomMiniBatches) = nothing
# Base.done(mb::RandomMiniBatches, i) = false
# Base.next(mb::RandomMiniBatches, i) = (DataSubset(mb.source, rand(1:nobs(mb.source), mb.batch_size)), nothing)

# ----------------------------------------------------------------------------

# "just keep giving object `o` at each iteration"
# type InfiniteGenerator{T} <: AbstractSubsets
#     o::T
# end
# Base.start(ep::InfiniteGenerator) = nothing
# Base.done(ep::InfiniteGenerator, i) = false
# Base.next(ep::InfiniteGenerator, i) = ep.o, nothing
#
# forever(o) = InfiniteGenerator(o)

# ----------------------------------------------------------------------------
# convenience API

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
Split the data apart, either by specifying a size or giving a percentage split point.

```julia
# split into training and test sets, 60%/40% respectively
train, test = splitdata(X, Y, size = 0.6)

# split into equal-sized minibatches of 10 observations each
for batch in splitdata(X, Y, size = 10)
    ...
end

# Tips:
#   - Iterators can be nested
#   - Observations can be extracted immediately
for (x,y) in splitdata(shuffled(X, Y), size = 10)
    ...
end
```
"""
function splitdata(subset::DataSubset; size = default_batch_size(subset.source))
    n = nobs(subset)
    idx_list = if typeof(size) <: AbstractFloat
        # partition into 2 sets
        n1 = clamp(round(Int, size*n), 1, n)
        [1:n1, n1+1:n]
    elseif typeof(size) <: Integer
        offset = 0
        lst = []
        while offset < n
            sz = clamp(n - offset, 1, size)
            push!(lst, offset+1:offset+sz)
            offset += sz
        end
        lst
    end
    @show idx_list
    DataSubsets(map(idx -> DataSubset(subset, idx), idx_list))
end
splitdata(source...; kw...) = splitdata(DataSubset(source...); kw...)

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
infinite_batches(source; kw...) = infinite_batches(DataSubset(source); kw...)
infinite_batches(source...; kw...) = infinite_batches(DataSubset(source); kw...)
