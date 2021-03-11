using MatrixMarket

using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER

using Test
using SparseArrays


# Reference QR
function csrlsvqr!(A::CUSPARSE.CuSparseMatrixCSR{Float64},
                   b::CUDA.CuVector{Float64},
                   x::CUDA.CuVector{Float64},
                   tol::Float64,
                   reorder::Cint,
                   inda::Char)
    n = size(A,1)
    desca = CUSPARSE.CuMatrixDescriptor(
        CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL,
        CUSPARSE.CUSPARSE_FILL_MODE_LOWER,
        CUSPARSE.CUSPARSE_DIAG_TYPE_NON_UNIT, inda
    )
    singularity = Ref{Cint}(1)
    CUSOLVER.cusolverSpDcsrlsvqr(CUSOLVER.sparse_handle(), n, A.nnz, desca, A.nzVal, A.rowPtr, A.colVal, b, tol, reorder, x, singularity)

    if singularity[] != -1
        throw(SingularException(singularity[]))
    end

    return x
end

function batch_qr!(A::CUSPARSE.CuSparseMatrixCSR{Float64},
                  b::CUDA.CuVector{Float64},
                  x::CUDA.CuVector{Float64},
                  tol::Float64,
                  batchsize::Cint,
                  inda::Char)
    m, n = size(A)

    nz = nnz(A)
    # WORKAROUND
    # Batch QR uses different left-hand side A_i and solves
    # the systems A_i x_i = b_i.
    # Here, we want to solve A x_i = b_i, for a single LHS.
    # We duplicate the matrix A to create virtually the matrix A_1, ..., A_batchsize
    # WARNING: could be harmful for GPU's memory
    bnzval = CUDA.zeros(Float64, nnz(A) * batchsize)
    for i in 1:batchsize
        bnzval[1+(i-1)*nz:i*nz] .= A.nzVal
    end

    desca = CUSPARSE.CuMatrixDescriptor(
        CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL,
        CUSPARSE.CUSPARSE_FILL_MODE_LOWER,
        CUSPARSE.CUSPARSE_DIAG_TYPE_NON_UNIT, inda)

    info = Ref{CUSOLVER.csrqrInfo_t}()
    CUSOLVER.cusolverSpCreateCsrqrInfo(info)

    # Step 1: symbolic analysis
    # cusolverSpXcsrqrAnalysisBatched
    CUSOLVER.cusolverSpXcsrqrAnalysisBatched(
        CUSOLVER.sparse_handle(),
        m, m, nnz(A),
        desca,
        A.rowPtr, A.colVal, info[],
    )

    # Step 2: prepare working space
    # cusolverSpDcsrqrBufferInfoBatched
    internalDataInBytes = Ref{Csize_t}(0)
    workspaceInBytes = Ref{Csize_t}(0)

    CUSOLVER.cusolverSpDcsrqrBufferInfoBatched(
        CUSOLVER.sparse_handle(),
        m, m, A.nnz, desca,
        bnzval, A.rowPtr, A.colVal,
        batchsize, info[],
        internalDataInBytes, workspaceInBytes,
    )

    n_bytes = workspaceInBytes[] |> Int
    println("Run batch QR with ", n_bytes, " bytes")
    buffer_qr = CUDA.alloc(n_bytes)

    # Step 3: numerical factorization
    # cusolverSpDcsrqrsvBatched
    CUSOLVER.cusolverSpDcsrqrsvBatched(
        CUSOLVER.sparse_handle(), m, m, A.nnz, desca,
        bnzval, A.rowPtr, A.colVal,
        b, x, batchsize, info[], buffer_qr,
    )

    return x
end

function test_batch(J, batch)
    n, m = size(J)
    gJ = CuSparseMatrixCSR(J)
    b = rand(batch * n)
    gb = CuVector{Float64}(b)
    x = CUDA.zeros(Float64, batch * m)

    CUDA.@time batch_qr!(gJ, gb, x, 1e-8, Cint(batch), 'O')

    # Test that first RHS is the same as the one computed
    # using UMFPACK
    res = Array(x[1:n])
    @test isapprox(res, J \ b[1:n])
end

J = mmread("case300.txt")
n_batch = 100
test_batch(J, n_batch)

