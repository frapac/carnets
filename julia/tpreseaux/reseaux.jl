# TP RESEAU D'EAU
# ajouter matrice sparse
# ajouter descente nesterov
# ajouter plot graph

using LinearAlgebra

include("data.jl")


################################################################################
# PRIMAL PROBLEM
function fprimal(qc)
    q = q0 + B*qc
    z = r .* abs.(q) .* q
    # valeur du critère
    F = q'*z / 3 + pr'*Ar*q
    return F
end

function gprimal!(storage, qc)
    q = q0 + B*qc
    z = r .* abs.(q) .* q
    storage[:] = B' * (z+(Ar'*pr))
end

function hprimal!(storage, qc)
    q = q0 + B*qc
    z = r .* abs.(q) .* q
    storage[:, :] = 2 * B' * diagm(0 => vec(r .* abs.(q))) * B
end

function primaloracle(qc)
    q = q0 + B*qc
    z = r .* abs.(q) .* q
    # valeur du critère
    F = q'*z / 3 + pr'*Ar*q
    G = B' * (z+(Ar'*pr))
    H = 2 * B' * diagm(0 => vec(r .* abs.(q))) * B
    return F, G, H
end


################################################################################
# DUAL PROBLEM
function dualoracle(pd)
    # Valeur des pressions, des pertes de charge et des debits
    p = [ pr ; pd ];
    z = - (Ar' * pr + Ad'*pd);
    q = z ./ sqrt.(r.*abs.(z));

    # Valeur du critere
    F = ((q'*z)*(2/3)) + (pd'*fd);
    # Valeur du gradient
    G = fd - (Ad*q);
    # Valeur du hessien
    H = 0.5 * Ad * diagm(0 => ones(n) ./ (r .* abs.(q))) * Ad';
    return F, G, H
end


################################################################################
function wolfe(oracle, x, F, G, D, alpha)

    imax = 1000;

    # coefficients de la recherche linéaire
    omega1 = 0.1;
    omega2 = 0.9;
    taui   = 0.5;
    taue   = 2.0;
    # Indicateur de succes
    ok = 0;

    # Tolerance sur x
    dltx = 0.00000001;
    # initialization intervalle
    alphamin = 0.0;
    alphamax = Inf;

    # Initialisation du pas
    alphan = alpha;

    # Initialisation des variables
    xn = copy(x)
    xp = copy(x)
    Gn = zeros(Float64, length(x))
    # Terme des conditions de Wolfe
    prodir = G' * D;
    # Boucle de recherche lineaire

    i = -1;
    while i <= imax
        i = i + 1;
        # memorisation des variables
        xp[:] = xn;

        #  mise a jour des variables
        xn[:] = x + (alphan*D);

        Fn, Gn[:], _ = oracle(xn);

        #  valeurs des conditions de Wolfe
        condA = (Fn - F - (omega1*alphan*prodir))
        condW = ((Gn'*D) - (omega2*prodir))

        #  test des conditions de Wolfe
        if condA > 0.
            alphamax = alphan;
            alphan = (taui*alphamin) + (1-taui)*alphamax;
        else
            if condW >= 0
                ok = 1;
                break
            else
                alphamin = alphan;
                if alphamax == Inf
                    alphan = taue * alphamin;
                else
                    alphan = (taui*alphamin) + (1-taui)*alphamax;
                end
            end
        end
        #  test d'indistingabilite
        if norm(xn-xp) < dltx
            ok = 2;
            break
        end

    end

    return alphan
end


function cauchy(oracle, x, F, G, D, alpha)
    fcauchy(alpha) = fprimal(x + (alpha[1]*D))
    function gcauchy!(storage, alpha)
        gprimal!(G, x + (alpha[1]*D))
        storage[:] = D'*G;
    end
    res = optimize(fcauchy, gcauchy!, [alpha], g_tol=1e-10)
    return res.minimizer[1]
end


################################################################################
"Gradient descent algorithm."
function gradient_F(oracle, xini;
                    iter=5000, alphai=.0005, tol=1e-6, rl=true)
    nx = length(xini)
    logG = Float64[]
    logP = Float64[]
    cout = Float64[]

    tic = time()

    x = copy(xini);
    kstar = iter;

    F = Inf
    G = zeros(nx)

    for k = 1:iter
        F, G[:], _ = oracle(x)

        if norm(G) <= tol
            kstar = k
            break
        end

        D = -G
        alpha = (rl) ?  wolfe(oracle, x, F, G, D, alphai) : alphai
        x[:] = x + alpha * D

        push!(logG, log10(norm(G)))
        push!(logP, log10(alpha))
        push!(cout, F)

    end

    fopt = F
    xopt = x
    gopt = G

    exectime = time() - tic
    println("Number iterations: ", kstar)
    println("CPU time: ", exectime)
    println("Critere optimal: ", fopt)
    println("Norme gradient: ", norm(G))
    return logG, logP, cout
end




################################################################################
function polakribiere(oracle, xini;
                      iter=5000, alphai=.0005, tol=1e-6, rl=true)

    x = copy(xini);
    nx = length(x)
    logG = Float64[];
    logP = Float64[];
    cout = Float64[];

    tic = time();

    F = Inf
    G  = zeros(Float64, nx)
    Gk = zeros(Float64, nx)
    D  = zeros(Float64, nx)

    kstar = iter;
    for k = 1:iter
        #    - valeur du critere et du gradient
        F, G[:], _ = oracle(x);
        #    - test de convergence
        if norm(G) <= tol
            kstar = k;
            break
        end
        #    - evolution de la norme du gradient
        push!(logG, log10(norm(G)))
        #    - evolution du critere
        push!(cout, F)
        #    - direction de descente
        if k == 1
            D[:] = -G;
        else
            bet = (G'*(G-Gk)) / (Gk'*Gk);
            D[:] = -G + (bet*D);
        end
        #    - test de la direction de descente
        coe = (D' * G)[1]
        if coe >= 0.
            D[:] = -G;
        end
        #    - memorisation du gradient
        Gk[:] = G[:];
        #    - valeur initiale du pas
        alpha  = alphai;
        #    - recherche lineaire par la regle de Wolfe
        alphan = wolfe(oracle, x, F, G, D, alpha);
        #    - evolution du pas de gradient
        push!(logP, log10(alphan))

        x[:] = x + (alphan*D);
    end


    fopt = F;
    xopt = x;
    gopt = G;

    tcpu = time() - tic


    println("Number iterations: ", kstar)
    println("CPU time: ", tcpu)
    println("Critere optimal: ", fopt)
    println("Norme gradient: ", norm(G))
    return logG, logP, cout
end



################################################################################
function BFGS(oracle, xini;
              iter=5000, alphai=.0005, tol=1e-6, rl=true)
    x = copy(xini);
    nx = length(x)
    logG = Float64[];
    logP = Float64[];
    cout = Float64[];

    tic = time();

    I = Diagonal(ones(nx));
    W = I;

    F = Inf
    G  = zeros(Float64, nx)
    Gk = zeros(Float64, nx)
    D  = zeros(Float64, nx)
    xk = zeros(Float64, nx)

    # Boucle sur les iterations

    kstar = iter;
    for k = 1:iter
        #    - valeur du critere et du gradient
        F, G, _ = oracle(x);
        #    - test de convergence
        if norm(G) <= tol
            kstar = k;
            break
        end
        #    - evolution de la norme du gradient
        push!(logG, log10(norm(G)))
        #    - evolution du critere
        push!(cout, F)
        #    - direction de descente
        if k == 1
            D = - G;
        else
            dx = x - xk;
            dG = G - Gk;
            cf = dG' * dx;
            W = ((I-((dx*dG')/cf))*W*(I-((dG*dx')/cf))) + ((dx*dx')/cf);
            D = - W * G;
        end

        if isnan(norm(D))
            break
        end
        #    - test de la direction de descente
        coe = (D' * G)
        if coe >= 0.
            println("Direction de montée --- itération ", k)
        end
        #    - memorisation des variables et du gradient
        xk[:] = x;
        Gk[:] = G;

        #    - valeur initiale du pas
        alpha  = alphai;
        #    - recherche lineaire par la regle de Wolfe
        alphan = wolfe(oracle, x, F, G, D, alpha);

        #    - evolution du pas de gradient
        push!(logP, log10(alphan))
        #    - mise a jour des variables
        x = x + (alphan*D);
    end

    fopt = F;
    xopt = x;
    gopt = G;

    tcpu = time() - tic

    println("Number iterations: ", kstar)
    println("CPU time: ", tcpu)
    println("Critere optimal: ", fopt)
    println("Norme gradient: ", norm(G))
    return logG, logP, cout
end



################################################################################
function newton(oracle, xini;
                iter=5000, alphai=.0005, tol=1e-6, rl=true)
    x = copy(xini);
    nx = length(x)
    logG = [];
    logP = [];
    cout = [];

    tic = time();

    F = Inf
    G  = zeros(Float64, nx)
    Gk = zeros(Float64, nx)
    D  = zeros(Float64, nx)
    xk = zeros(Float64, nx)

    # Boucle sur les iterations

    kstar = iter;
    for k = 1:iter
        #    - valeur du critere et du gradient
        F, G, H = oracle(x);
        #    - test de convergence
        if norm(G) <= tol
            kstar = k;
            break
        end
        #    - evolution de la norme du gradient
        push!(logG, log10(norm(G)))
        #    - evolution du critere
        push!(cout, F)
        #    - direction de descente
        D = - H \ G
        #    - test de la direction de descente
        coe = (D' * G)[1]
        if coe >= 0.
            println("Direction de montée --- itération ", k)
        end

        #    - valeur initiale du pas
        alpha  = alphai;
        #    - recherche lineaire par la regle de Wolfe
        alphan = wolfe(oracle, x, F, G, D, alpha);

        #    - evolution du pas de gradient
        push!(logP, log10(alphan))
        #    - mise a jour des variables
        x = x + (alphan*D);
    end

    fopt = F;
    xopt = x;
    gopt = G;

    tcpu = time() - tic

    println("Number iterations: ", kstar)
    println("CPU time: ", tcpu)
    println("Critere optimal: ", fopt)
    println("Norme gradient: ", norm(G))
    return logG, logP, cout
end

xini = .1 * rand(9)
println("#"^60)
println("PRIMAL")
println("Gradient descent")
f1, f2, f3 = gradient_F(primaloracle, xini, alphai=5e-4, rl=false)
println()
println("Gradient descent + Wolfe linesearch")
f1, f2, f3 = gradient_F(primaloracle, xini, alphai=1., rl=true)
println()
println("Polak Ribiere + Wolfe linesearch")
f1, f2, f3 = polakribiere(primaloracle, xini, alphai=1.)
println()
println("BFGS + Wolfe linesearch")
f1, f2, f3 = BFGS(primaloracle, xini, iter=100, alphai=1.)
println()
println("Newton + Wolfe linesearch")
f1, f2, f3 = newton(primaloracle, xini, iter=100, alphai=1)

if false
    xini = 100 .+ (10 .* rand(md));
    println("#"^60)
    println("DUAL")
    println("Gradient descent")
    f1, f2, f3 = gradient_F(dualoracle, xini, alphai=5e-4, rl=false)
    println()
    println("Gradient descent + Wolfe linesearch")
    f1, f2, f3 = gradient_F(dualoracle, xini, alphai=1., rl=true)
    println()
    println("Polak Ribiere + Wolfe linesearch")
    f1, f2, f3 = polakribiere(dualoracle, xini, alphai=1.)
    println()
    println("BFGS + Wolfe linesearch")
    f1, f2, f3 = BFGS(dualoracle, xini, iter=100, alphai=1.)
    println()
    println("Newton + Wolfe linesearch")
    f1, f2, f3 = newton(dualoracle, xini, iter=100, alphai=1)
end
