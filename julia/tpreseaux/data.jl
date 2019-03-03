
n = 22
m = 16
mr = 3
md = m - mr

orig = [ 1 2 3 4 5 6 7 8 8 9 10 11 13 1 2 4 5 7 8 14 2 10];
dest = [ 4 16 15 5 6 10 16 9 12 10 11 14 15 16 6 8 9 11 13 15 4 13];

absn = [11 18 38 4 8 15 26 4 10 19 26 7 21 33 33 16];
ordn = [28 21 8 21 17 17 26 9 13 13 18 4 9 18 12 24];


r = Float64[100, 10,1000, 100, 100, 10, 1000, 100, 1000, 100,
            1000, 1000, 1000, 10, 10, 100, 100, 1000, 100, 1000, 100, 10];
pr = Float64[105, 104, 110];
fd = Float64[0.08, -1.30, 0.13, 0.09, 0.16, 0.14, 0.12, 0.07, 0.17, 0.11,
             0.25, 0.01, 0.13];


################################################################################
A = zeros(m, n);

for l = 1:n
    A[orig[l], l] = +1
    A[dest[l], l] = -1
end

Ar = A[1:mr, :]
Ad = A[mr+1:m, :];

AdT = Ad[:, 1:md]
AdI = inv(AdT);

AdC = Ad[:, md+1:n];

# Here I stands for identity matrix (former eye function)
B = [-AdI * AdC ; I];

# Vecteur des débits admissibles
q0 = [AdI * fd; zeros(n-md)];


function hydrauliqueP(qc)
    # débit des arcs
    q = q0 + B*qc
    # pertes de charge des arcs
    z = r .* abs.(q) .* q
    # flux aux noeuds
    f = [Ar*q ; fd]
    # pressions aux noeuds
    temp = (Ar'*pr) + z
    p = [pr; -AdI'*temp[1:md]]
    return q, z, f, p
end

function hydrauliqueD(pd)
    # pressions aux noeuds
    p = [pr ; pd]
    # pertes de charge aux arcs
    z = -A' * p;
    # débit des arcs
    q = z ./ sqrt(r .* abs.(z))
    # flux aux noeuds
    f = [Ar*q; fd]
    return q, z, f, p
end

# vérification des équations d'équilibre d'un réseau de
# distribution d'eau
function verification(q, z, f, p)
    # écarts maximaux sur les lois de Kirchhoff
    tol1 = max(abs.(A*q-f))
    tol2 = max(abs.(A'*p+z))
    println("Vérification des équations d'équilibre du réseau")
    println("sur les débits: ", tol1, "\t sur les pressions: ", tol2)
end
