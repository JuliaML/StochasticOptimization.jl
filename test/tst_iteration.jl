
using StochasticOptimization.Iteration

@testset "Data Iteration" begin
    n = 4
    X = rand(2,n)
    y = rand(n)

    # nobs/getobs of arrays
    @test nobs(X) == n
    @test nobs(y) == n
    @test getobs(X, 1) == X[:,1]
    @test getobs(y, 1) == y[1]
    @test getobs(X, 1:2) == X[:, 1:2]
    @test getobs(y, 1:2) == y[1:2]

    # construction
    subset = eachobs(X, y)
    @test typeof(subset) <: DataSubset{Tuple{Matrix{Float64},Vector{Float64}}}
    @test length(subset) == n
    @test subset.indices == 1:n
    @test subset.source == (X,y)
    @test nobs(subset) == n

    # iterating... sort of
    o1, o2, o3, o4 = subset
    @test o2 == (X[:,2], y[2])

    # extraction
    subset2 = DataSubset((X, y), 1:1)
    cx, cy = collect(subset2)
    @test typeof(cx) <: Matrix
    @test cx == X[:,1:1]
    @test typeof(cy) <: Vector
    @test cy == y[1:1]

    # random obs
    (x1,x2),yi = rand(subset)
    @test x1 in X
    @test x2 in X
    @test yi in y

    # random arrays
    xs,ys = rand(subset, 2)
    @test size(xs) == (2,2)
    @test size(ys) == (2,)

    # getindex
    for i=1:n
        @test subset[i] == (X[:,i], y[i])
    end

    # iteration
    for (i,(x,yi)) in enumerate(subset)
        @test x == X[:,i]
        @test yi == y[i]
    end

    # shuffling
    ss = shuffled(X,y)
    @test length(ss.indices) == n

    # test/train split
    train, test = batches(X,y,size=0.5)
    # @show train test
    @test typeof(train) <: Tuple
    @test typeof(test) <: Tuple
    @test train[1] == view(X,:,1:2)
    @test test[1] == view(X,:,3:4)
    @test train[2] == view(y,1:2)
    @test test[2] == view(y,3:4)
    @test nobs(train) == 2
    @test nobs(test) == 2

    # minibatch split
    bs = batches(X,y,size=2)
    @test typeof(bs) <: DataSubsets
    @test length(bs) == 2
    for (x,yi) in bs
        # @show x yi
        @test typeof(x) <: SubArray
        @test typeof(yi) <: SubArray
        @test size(x) == (2,2)
        @test length(yi) == 2
        for (xj,yj) in eachobs(x,yi)
            # just to make sure there's no errors in nesting...
        end
    end

    # train/validate/test split
    X = rand(2,10)
    y = rand(10)
    bs = batches(X,y,size=(0.5,0.2))
    train, validate, test = bs
    @test nobs(train) == 5
    @test nobs(validate) == 2
    @test nobs(test) == 3
    @test bs.subsets[1].indices == 1:5
    @test bs.subsets[2].indices == 6:7
    @test bs.subsets[3].indices == 8:10

    # kfolds
    kf = kfolds(X, y)
    @test typeof(kf) <: KFolds
    @test kf.k == 5
    @test StochasticOptimization.start_index(kf, 2) == 3
    @test StochasticOptimization.end_index(kf, 2) == 4
    i = 0
    for (train, test) in kf
        i += 1
        # @show typeof(train) typeof(test)
        @test typeof(train) <: Tuple
        @test typeof(test) <: Tuple
        @test train == getobs((X,y), setdiff(1:10, 2i-1:2i))
        @test test == getobs((X,y), 2i-1:2i)
        @test nobs(train) == 8
        @test nobs(test) == 2
    end
    @test i == kf.k

    loo = leave_one_out(y)
    @test typeof(loo) <: KFolds
    @test loo.k == nobs(X)
    for (train,test) in loo
        @test nobs(train) == 9
        @test nobs(test) == 1
    end

    # filtering
    newx,newy = filterobs(i -> i%2==0, X, y)
    @test newx == X[:,2:2:10]
    @test newy == y[2:2:10]
end
