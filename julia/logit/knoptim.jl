using KNITRO

include("logit.jl")

function callbackEvalF(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    dat = userParams[:data]
    evalResult.obj[1] = loss(x, dat)
    return 0
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, userParams)
    ω = evalRequest.x
    dat = userParams[:data]
    ∇loss(evalResult.objGrad, ω, dat)
    return 0
end

function callbackEvalH!(kc, cb, evalRequest, evalResult, userParams)
    h = evalRequest.x
    vec = evalRequest.vec
    dat = userParams[:data]
    hessvec_loss(evalResult.hessVec, h, vec, dat)
    return 0
end

knfit(X, y; hv=false) = knfit(LogitData(X, y), hv=hv)
function knfit(dat::LogitData; hv=false)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, dim(dat))
    KNITRO.KN_set_var_primal_init_values(kc, zeros(dim(dat)))
    cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)
    KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)
    if hv
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callbackEvalH!)
    end

    KNITRO.KN_set_param(kc, "outlev", 2)
    KNITRO.KN_set_param(kc, "algorithm", 1)
    KNITRO.KN_set_param(kc, "hessopt", 6)
    KNITRO.KN_set_param(kc, "linesearch", 1)
    #= KNITRO.KN_set_param(kc, "derivcheck", 2) =#
    #= KNITRO.KN_set_param(kc, "derivcheck_type", 2) =#
    if hv
        KNITRO.KN_set_param(kc, "algorithm", 2)
        KNITRO.KN_set_param(kc, "hessopt", 5)
    end

    KNITRO.KN_set_cb_user_params(kc, cb, dat)

    nStatus = @time KNITRO.KN_solve(kc)
    w_opt = KNITRO.get_solution(kc)
    KNITRO.KN_free(kc)
    return w_opt
end
