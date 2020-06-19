# Source code of the blog post:
# https://frapac.github.io/2020/06/1_graph_analysis/
using LightGraphs
using EzXML
using GraphIO
using DataFrames
using LinearAlgebra
using Statistics
using UnicodePlots
using Printf
using CSV

#+
GRAPH_SRC = "coauthors.graphml"

# Import graph into LightGraphs
reader = GraphIO.GraphML.GraphMLFormat()
g = loadgraph(GRAPH_SRC, reader)

# Load metadata
df = DataFrame(CSV.File("metadata.txt", delim=';', header=0))

#+
# Utils function
function parse_name(name)
    # First try
    regs = [r"\'(.*?)\'", r"\"(.*?)\""]
    for reg in regs
        m = match(reg, name)
        if m !== nothing
            # Remove matching ''
            return m.match[2:end-1]
        end
    end
end

function get_name(id::Int)
    raw_name = df[id, 3]
    return parse_name(raw_name)
end


#+
is_connected(g)
connected_graphs = connected_components(g)

#+
number_nodes = [length(s) for s in connected_graphs]
# Sort the results
sort!(number_nodes, rev=true)
number_nodes[1:10]
print("Average number of nodes: ", mean(number_nodes[2:end]))

#+

# Get first connected component
sv = connected_components(g)[1]
sg, _ = induced_subgraph(g, sv)

deg = degree(sg)
diam = diameter(sg)
println("Nodes: ", nv(sg))
println("Edges: ", ne(sg))
println("Max degree: ", maximum(deg))
println("Average degree: ", mean(deg))
println("Diameter: ", diam)
println("Density: ", density(sg))

# Output degree distribution
histogram(deg, nbins=30, xscale=log10)


deg_distrib = degree_histogram(sg)
deg_keys = keys(deg_distrib) |> collect
deg_vals = values(deg_distrib) |> collect

r = scatterplot(log10.(deg_keys), log10.(deg_vals))
xlabel!(r, "log degree")
ylabel!(r, "log n")

#+
# Average distance
avg_distance = zeros(Float64, nv(sg))
for i in eachindex(avg_distance)
    res = bellman_ford_shortest_paths(sg, i)
    avg_distance[i] = mean(res.dists)
end
mean(avg_distance)

#+
author_min = findmin(avg_distance)[2]
original_index = sv[author_min]
name = get_name(original_index)

#+

function rank(sg::Graph, metrics, n=10)
    m = metrics(sg)
    classement = sortperm(m, rev=true)
    # From biggest to lowest
    head = classement[1:n]
    @printf("* %-4s   (%7s) %-30s\n", "#", "Score", "Name")
    for i in eachindex(head)
        name = parse_name(df[sv[head[i]], 3])
        @printf("* %-4d * (%.1e) %-35s %s \n", i, m[head[i]], name, "*")
    end
end

rank(sg, closeness_centrality)
rank(sg, betweenness_centrality)
rank(sg, eigenvector_centrality)
rank(sg, pagerank)

function dump_ranking(file, metrics)
    m = metrics(sg)
    classement = sortperm(m, rev=true)
    open(file, "w") do io
        @printf(io, "* %-4s   (%7s) %-30s\n", "#", "Score", "Name")
        for i in eachindex(classement)
            name = parse_name(df[sv[classement[i]], 3])
            @printf(io, "* %-4d * (%.1e) %-35s %s \n", i, m[classement[i]], name, "*")
        end
    end
end
rankings = [pagerank,
            degree_centrality,
            eigenvector_centrality,
            closeness_centrality,
            betweenness_centrality,
           ]
for ranking in rankings
    sym = Symbol(ranking)
    dump_ranking("results/$sym.txt", ranking)
end

#+
# Core / periphery

core_periph = core_periphery_deg(sg)
core_index = sv[findall(core_periph .== 1)]
core_names = get_name.(core_index)

ncols = 3
nlines = div(length(core_names), ncols)
for i in 1:nlines
    offset = (i-1) * ncols
    for j in 1:ncols
        @printf("* %-25s", core_names[offset + j])
    end
    @printf("\n")
end
offset = nlines * ncols
remain = length(core_names) % ncols
for i in 1:remain
    @printf("* %-25s", core_names[offset + i])
end
@printf("\n")

