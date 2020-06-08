using Arpack
using LightGraphs
using EzXML
using GraphIO
using DataFrames
using LinearAlgebra
using CSV

DUMP = "all_graph.xml"

reader = GraphIO.GraphML.GraphMLFormat()
g = loadgraph(DUMP, reader)

# Get first connected component
sv = connected_components(g)[1]
sg, _ = induced_subgraph(g, sv)

df = DataFrame(CSV.File("metadata.txt", delim=';', header=0))

#= all_degrees = degree(sg) =#
#= deg_max = maximum(all_degrees) =#
#= out = zeros(Int, deg_max) =#
#= for i in 1:deg_max =#
#=     out[i] = length(findall(all_degrees .== i)) =#
#= end =#
#
function spectralclustering(inverse_laplacian, ncluster::Int64)
    l, v = eigvecs(inverse_laplacian, nv=ncluster+1)
    u = v[:, 2:ncluster+1]  #nbclusters first eigen vectors
    #Clustering of the rows of U among nbclusters clusters
    class =  Clustering.kmeans(u', ncluster)
    return class.assignments
end

#= using Compose, Cairo, Colors =#
#= graph = SimpleGraph(log1p.(coauthor_graph)) =#

#= lastname(n) = split(n,",")[1] =#
#= gplot(sg, nodelabelsize=0.1, layout=(args...)->spring_layout(args...; C=20)) =#
