using JSOSolvers, NLPModels

mutable struct LogReg <: AbstractNLPModel
  meta::NLPModelMeta
  counters::Counters
  data::LogitData
end

function LogReg(X::Array{T, 2}, y::Vector{T}) where T
  n, d = size(X)
  meta = NLPModelMeta(d, x0=zeros(T, d),
                      name="Logit")

  return LogReg(meta, Counters(), LogitData(X, y))
end

function NLPModels.obj(nlp :: LogReg, x :: AbstractVector)
  tmp = loss(x, nlp.data)
  println(tmp)
  return tmp
end

function NLPModels.grad!(nlp :: LogReg, x :: AbstractVector, gx :: AbstractVector)
  ∇loss(gx, x, nlp.data)
  return gx
end

function NLPModels.hprod!(nlp :: LogReg, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight=1.0)
  hessvec_loss(Hv, x, v, nlp.data)
  return Hv
end

