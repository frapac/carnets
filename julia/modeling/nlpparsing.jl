using ModelingToolkit

is_leaf(ex::Operation) = isempty(ex.args)
depth(ex::ModelingToolkit.Constant) = 1
function depth(ex::Operation)
    if is_leaf(ex)
        return 1
    else
        dep = 0
        for op in ex.args
            dep = max(dep, depth(op))
        end
        return 1 + dep
    end
end

is_linear(ex::ModelingToolkit.Constant) = true
function is_linear(ex::Operation)
    dep = depth(ex)
    if dep == 1
        return true
    elseif dep == 2 && ex.op === (*)
        return any(ModelingToolkit.is_constant.(ex.args))
    elseif ex.op === (+)
        return all(is_linear.(ex.args))
    end
    return false
end

linear_structure!(ex::ModelingToolkit.Constant) = nothing
function linear_structure!(ex::Operation)
    dep = depth(ex)
    if dep == 1
        return
    end
    #= elseif ex.op !== (+) =#
    #=     return =#
    #= end =#

    linear_structure!.(ex.args)
    idx = findall(is_linear, ex.args)
    println(idx)
    filter!(is_linear, ex.args)

    # Process tree.
    #= filter!(is_linear, op.args) =#
    return
end


function f_test(x)
    return 2*(2*x + log(x))
end


@variables x
ex = f_test(x)
depth(ex)
