using LinearAlgebra

mutable struct SiteOperatorCache

    site_count::Int64
    operators::Dict{Symbol, Array{Array{Float64,2}, 1}}

    function SiteOperatorCache(site_count::Int64)
        cache = new(site_count)
        cache.operators = Dict{Symbol, Array{Array{Float64,2}, 1}}()
        return cache
    end

end

function add_operator(cache::SiteOperatorCache, basis_operator::Symbol)

    id = I(4)
    id_jw = Diagonal([1, -1, 1 ,-1])

    cache.operators[basis_operator] = []

    for site_index in 1:cache.site_count

        site_operator = eval(basis_operator)

        for i in 1 : site_index - 1
            site_operator = kron(id_jw, site_operator)
        end

        for i in site_index + 1 : cache.site_count
            site_operator = kron(site_operator, id)
        end

        push!(cache.operators[basis_operator], site_operator)

    end

end

function get_operator(cache::SiteOperatorCache, basis_operator::Symbol, site_index)
    return cache.operators[basis_operator][site_index]
end



cache = SiteOperatorCache(2)

op1 = [0.0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]

add_operator(cache, :op1)


