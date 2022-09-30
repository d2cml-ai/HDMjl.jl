] activate .

using HDMjl, Random, Distributions

n = 250
p = 40
px = 10


X = randn((n, p))
beta = vcat(repeat([2.], px), zeros(p - px))
intercept = 1
P = exp.(intercept .+ X * beta) ./ (1 .+ exp.(intercept .+ X * beta))
# y = Int64[]

x = rand(Normal(), (n, p))

y = rand(Binomial(), n)

logit_out = rlassologit(x, y, intercept = true)

exit()

function suma(x)
    z1 = print("z")
    n = Dict("x" => x, "z" => z1)
    z1
    return z1;
end

function suma.actio(X)
    return x + 1
end


suma(x, y) = x + y

mutable struct lasologit
    x::Int
    y::Int

    fsit::Function = x -> x + y

end

lasologit(12, 3)


mutable struct BitNumber
    val::Int
    bit_end::UInt
    bit_start::UInt
    width::UInt
    the_bits::Function
    BitNumber(v,e,s,w,bits_func) = begin
        ret = new( (a=63-e+s; v=v<<a ; v>>a) ,e,s,w)
        ret.the_bits = (bit_end,bit_start) -> bits_func(ret,bit_end,bit_start)
        ret
    end
end

mutable struct sumas
    x::Int64
    y::Int64
    # add_new::Function

    suma(x, y) = x + y
end

Base.@kwdef struct Model1
    p::Float64 = 2.0
    n::Int64 = 4
    # f::Function
    f::Function = (p, n) -> begin
    n + p + 1
    end
    # function f(x, y)
    #     x + y
    # end

end

typeof(Model1().f(12, 1))


mutable struct si1
    x::Real = 12
    y::Real = 8
    si1(x, y) = x + y

end