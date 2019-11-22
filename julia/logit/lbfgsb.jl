using LBFGSB


include("logit.jl")

function callback_builder(dat::LogitData)
    eval_f = x -> loss(x, dat)
    eval_g = (g, x) -> ∇loss(g, x, dat)
    return (eval_f, eval_g)
end

function bfit(dat::LogitData)
    f, g! = callback_builder(dat)
    optimizer = L_BFGS_B(2048, 17)
    n = dim(dat)  # the dimension of the problem
    x = fill(Cdouble(0e0), n)  # the initial guess
    # set up bounds
    bounds = zeros(3, n)
    #= for i = 1:n =#
    #=     bounds[1,i] = 0  # represents the type of bounds imposed on the variables: =#
    #=                      #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound =#
    #=     #1= bounds[2,i] = isodd(i) ? 1e0 : -1e2  #  the lower bound on x, of length n. =1# =#
    #=     #1= bounds[3,i] = 1e2  #  the upper bound on x, of length n. =1# =#
    #= end =#

    fout, xout = optimizer(f, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=1, maxfun=15000, maxiter=15000)
end
