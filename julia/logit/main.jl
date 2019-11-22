include("logit.jl")
include("libsvm_parser.jl")
include("lbfgsb.jl")
include("scaling.jl")

using Optim, LineSearches

#= DATASET = "/media/sf_D_DRIVE/dev/Knitro-Learn/examples/data/colon-cancer.bz2" =#
DATASET = "/media/sf_D_DRIVE/dev/Knitro-Learn/examples/data/covtype.libsvm.binary.bz2"
#= DATASET = "/media/sf_D_DRIVE/dev/Knitro-Learn/examples/SUSY.bz2" =#
if true
    res = @time data_parser(DATASET)
    X = to_matrices(res)
    scale!(NormalScaler(), X)
    y = copy(res.labels)
    y[y .== 2] .= -1
    data = LogitData(X, y)
    f, g! = callback_builder(data)
end

algo = BFGS(linesearch=LineSearches.HagerZhang())
options = Optim.Options(iterations = 1000, show_trace=true)
Optim.optimize(f, g!, zeros(dim(data)), algo, options)
