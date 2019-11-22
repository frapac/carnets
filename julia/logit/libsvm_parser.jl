
import GZip, CodecBzip2

const SEPARATOR = ":"

struct LibSVMData{T}
    # Label to predict
    labels::Vector{T}
    # Features (as sparse matrix)
    index_rows::Vector{Int64}
    index_cols::Vector{Int64}
    coefs::Vector{T}
end
LibSVMData{T}() where T = LibSVMData(T[], Int64[], Int64[], T[])
ndata(data::LibSVMData) = length(data.labels)
nfeatures(data::LibSVMData) = maximum(data.index_cols)

function to_matrices(data::LibSVMData{T}) where T
    n, m = ndata(data), nfeatures(data)
    X = zeros(T, n, m)
    for (i, j, val) in zip(data.index_rows, data.index_cols, data.coefs)
        @inbounds X[i, j] = val
    end
    return X
end

function gzip_open(f::Function, filename::String, mode::String)
    if endswith(filename, ".gz")
        return GZip.open(f, filename, mode)
    elseif endswith(filename, ".bz2")
        return Base.open(f, CodecBzip2.Bzip2DecompressorStream, filename, mode)
    else
        return open(f, filename, mode)
    end
end

function data_parser(filename::String, T::Type=Float64)
    dataset = LibSVMData{T}()
    gzip_open(filename, "r") do io
         _fetch_dataset(io, dataset)
    end
    return dataset
end

function _fetch_dataset(io::IO, data::LibSVMData{T}) where T
    nline = 0
    while !eof(io)
        nline += 1
        line = strip(readline(io))

        values_ = split(line, " ", keepempty=false)

        push!(data.labels, parse(T, values_[1]))

        for val in values_[2:end]
            tmp = split(val, SEPARATOR, limit=2)
            ncol = parse(Int64, tmp[1])
            push!(data.index_rows, nline)
            push!(data.index_cols, ncol)
            push!(data.coefs, parse(T, tmp[2]))
        end
    end
end
