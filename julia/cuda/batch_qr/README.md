# Batch QR with CUSPARSE

Tested with CUDA 11.2.0 and V100 GPU.

Require Julia, installed with the CUDA.jl wrapper.

## Installation

Launch Julia:
```shell
julia --project

```
Then download all dependencies via
```julia
julia> ] instantiate

```

If CUDA.jl is not installed, this command will downlad a CUDA artifact.


## Test

Test on case300:

```julia
julia> include("batch_qr.jl")
```
