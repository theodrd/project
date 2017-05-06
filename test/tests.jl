using Distributions
using Dierckx
using Optim
using Plots
using env_comp_MFG

# Initialization
sigma = 0.02
max_E = 20
step_E = 0.5 
step_k = 0.25 
k_max = 5 
step_time = 1
T = 25
a = 2
b_cost = 8
c = 2
tax = 0.1
rate = 0.05
rho = 0.02
demand = 8
elas = 1.4
max_price = 0.5

# Discretization and dimension of the grid
dim_E     = try 
    Integer(1 + max_E / step_E)
catch
    println("max_E must be a multiple of step_E. Dimension is evaluated as its entire part.")
    Integer(floor(1 + max_E / step_E))
end
dim_t     = try 
    Integer(1 + T / step_time)
catch
    println("T must be a multiple of step_time. Dimension is evaluated as its entire part.")
    Integer(floor(1 + T / step_time))
end
dim_k     = try 
    Integer(k_max / step_k)
catch
    println("T must be a multiple of step_time. Dimension is evaluated as its entire part.")
    Integer(floor(k_max / step_k))
end

m0 = zeros(dim_E, dim_k,)

# parameter initialization
for k in 1:dim_k
    for i in 1:dim_E
        m0[i,k] = max((i-Integer(floor(dim_E/12)))*(Integer(ceil(dim_E/2)-i)),0) # technology is uniformly distributed
    end
end

m0    = m0 / sum(m0)


p_init        = 8 * ones(1,dim_t-1)
N          = Integer(floor((3 * sigma* max_E * sqrt(step_time)/step_E)))
NormConv = zeros(dim_E,1+2*N)

# Construct the variable NormConv for convolutions
for E in 2:dim_E
    N = Integer(floor((3 * sigma* (E-1) * sqrt(step_time))))

    if sigma == 0
        NormConv[E, N+1] = 1
    else
        NormConv[E, N+1] = pdf((Normal(0,sigma*(E-1) * step_E * sqrt(step_time))), 0)
    end

    for j=1:N
        NormConv[E,N+1-j] = pdf(Normal(0, sigma *(E-1)* step_E* sqrt(step_time)), j*step_E)
        NormConv[E,N+1+j] = pdf(Normal(0, sigma *(E-1)* step_E* sqrt(step_time)), j*step_E)
    end

    NormConv[E,:] = NormConv[E,:] / sum(NormConv[E,:])
end


@testset "Project test" begin

    @testset "firm's problem" begin
        # verify that the function indeed update the different values
        V, q, i, m, Tot_prod_iter, Tot_inv_iter, p_iter = env_comp_MFG.fun_firm_prob(p_init, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, a, b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
        @test p_iter != p_init

    end

    @testset "iterations" begin
        # check the find_fp_Vp function
        V, q, i, m, Tot_prod, Tot_inv, p, iter = env_comp_MFG.find_fp_Vp(p_init, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, a,
                            b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
        @test m[:,:,1] != m[:,:,5]
        @test m[:,:,5] != m[:,:,10]
    end


end
