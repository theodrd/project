# This code is my version of a MFG. It was inspired by Guéant, O., Lasry, J. M., & Lions, P. L. (2010),
# Mean field games and oil production


module env_comp_MFG
   

    using Distributions
    using Dierckx
    using Optim
    using Plots
    
    

    """
    Solves the stochastic control problem of the firms and computes the new price.

    #### Fields
        - `p_guess`    : price vector
        - `m0`         : initial firms distribution 
        - `step_time`  : unit step for T grid 
        - `T`          : final time
        - `step_E`     : unit step for E grid
        - `max_E`      : max permit to emit given by gov
        - `step_k`     : unit step for k grid
        - `k_max`      : max mitigating technology 
        - `sigma`      : intensity of the volatility of the emission induced by production 
        - `a`          : cost parameter of producing
        - `b_cost`     : cost parameter of producing 
        - `rate`       : interest rate 
        - `rho`        : growth rate of the economy 
        - `demand`     : global wealth 
        - `tax`        : tax on pollution stock 
        - `c`          : investment cost 
        - `elas`       : demand parameter from the CES demand function 
        - `max_price`  : This is not a maximum price per se but make the demand collapse and facilitate the 
                         numerical computations 
        - `NormConv`   : normal kernel for convolution

    #### Returns
        -`V`        : the value function
        -`q`        : production policy
        -`i`        : investemnt policy
        -`m`        : evolution of the distribution
        -`Tot_prod` : total production at each t
        -`Tot_inv`  : total investment at each t
        -`p`        : updated price        

    """
    function fun_firm_prob(p_guess, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, 
        a, b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
        
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
                        println("k_max must be a multiple of step_k. Dimension is evaluated as its entire part.")
                        Integer(floor(k_max / step_k))
                    end
    
        
        # Define a grid we will use later for interpolation
        E_grid = 0:step_E:max_E
        k_grid = step_k:step_k:k_max

        # create array to stock the values
        V         = zeros(dim_E, dim_k, dim_t)
        q         = zeros(dim_E, dim_k, dim_t)
        i         = zeros(dim_E, dim_k, dim_t)
        Tot_prod  = zeros(1, dim_t)
        Tot_inv   = zeros(1, dim_t)
        p         = zeros(1, dim_t - 1)
        m         = zeros(dim_E, dim_k, dim_t) 
        
        # Optimal control problem
        for t in (dim_t-1):-1:1

            # We compute the convolution product of u and the gaussian to replace the Laplace operator
            # following Guéant, O., Lasry, J. M., & Lions, P. L. (2010). Mean field games and oil production
            Vconv = zeros(dim_E, dim_k,)

            for k in 1:dim_k

                Vconv_int = zeros(dim_E,)

                for E in 2:dim_E

                    N = Integer(floor((3 * sigma * (E-1) * sqrt(step_time))))
                    Vconv_int[E, 1] = V[E, k, t+1] * NormConv[E,N+1]

                    for j in 1:N
                        if (E + j <= dim_E)
                            Vconv_int[E,1] = Vconv_int[E,1] + V[max(E-j,1), k, t+1] * NormConv[E,N+1+j] + V[E+j, k, t+1] * NormConv[E,N+1-j]
                        else # Neumann condition in E_max
                            Vconv_int[E,1] = Vconv_int[E,1] + V[max(E-j,1), k, t+1] * NormConv[E,N+1+j] + V[2 * dim_E - (E+j), k, t+1] * NormConv[E, N+1-j]
                        end
                    end

                    Vconv[:, k] = Vconv_int[:,1]

                end
            end
            
            Vconv_f = Dierckx.Spline2D(E_grid, k_grid, Vconv)

            for E in 2:dim_E
                for k in 1:(dim_k)
                    # Optimization at date t, i.e., where to go at time t + 1
                    min_Vconv_f(x::Vector) = - (((p_guess[t] - a) * x[1]
                            - b_cost / (2 * step_time) * (x[1]^2) - c/(2 * step_time)*x[2]^2 + tax * E
                            + (1 - rate * step_time) * Vconv_f(E*step_E - k*x[1], k*step_k - x[2])))
                    function min_Vconv_f_gradient!(x::Vector, storage::Vector)
                        storage[1] = max(3, min((p_guess[t] - a) - b_cost / (step_time) * x[1] - k * (1 - rate * step_time) * (Vconv_f(max(E*step_E - k*x[1] + step_E, max_E), k*step_k - x[2]) 
                            - Vconv_f(min(E*step_E - k*x[1] - step_E, step_E), k*step_k - x[2]))/(2 * step_E), -3))
                        storage[2] = max(3, min(- c/step_time * x[2] - (1 - rate * step_time) * (Vconv_f(E*step_E - k*x[1], max(k*step_k - x[2] + step_k, k_max)) 
                            - Vconv_f(E*step_E - k*x[1], min(k*step_k - x[2] - step_k, step_k))), -3))
                    end

                    controls = optimize(DifferentiableFunction(min_Vconv_f, min_Vconv_f_gradient!), [E*step_E / k /2,k*step_k/2], [0.0,0.0], [E*step_E / k, k*step_k], Fminbox(); optimizer = GradientDescent)
                    q[E, k, t] = controls.minimizer[1]
                    i[E, k, t] = controls.minimizer[2]
                    V[E,k,t]   = - controls.minimum

                end
            end

        end


        # Initialization of m
        m[:,:,1] = m0

        # Transport equation for m
        for t in 1:(dim_t-1)

            # Need to change the integration procedure and interpolate m
            Tot_prod[1,t] = sum(q[:,:,t] .* m[:,:,t])
            Tot_inv[1,t]  = sum(i[:,:,t] .* m[:,:,t])

            # non-random part of the transport equation obtained using the optimal control q
            for k in 1:dim_k
                for E in 2:dim_E
                    cible_E                      = E - k * step_k *q[E, k, t] / step_E
                    cible_k                      = k - i[E, k, t]/step_k
                    arr_inf_E                    = Integer(floor(cible_E)) # floor in case production brings you out of the grid
                    arr_sup_E                    = min(Integer(arr_inf_E + 1), dim_E)
                    arr_inf_k                    = Integer(floor(cible_k)) # floor in case production brings you out of the grid
                    arr_sup_k                    = min(Integer(arr_inf_k + 1), dim_k)
                    # correct on the grid for the approximation of q by a grid point: linear approximation
                    m[max(arr_inf_E, 1), k, t+1] = m[max(arr_inf_E,1), k, t+1] + (arr_sup_E - cible_E) * m[E, k, t]
                    m[arr_sup_E, k, t+1]         = m[arr_sup_E, k, t+1] + (cible_E - arr_inf_E) * m[E, k, t]
                    m[E, max(arr_inf_k, 1), t+1] = m[E, max(arr_inf_k,1), t+1] + (arr_sup_k - cible_k) * m[E, k, t]
                    m[E, arr_sup_k, t+1]         = m[E, arr_sup_k, t+1] + (cible_k - arr_inf_k) * m[E, k, t]
                end

            end


            # We now apply the convolution instead of the Laplace operator to genenerate randomness
            # following Guéant, O., Lasry, J. M., & Lions, P. L. (2010). Mean field games and oil production
            mconv = zeros(dim_E, dim_k, 1)

            for k in 1:dim_k

                mconv_int = zeros(dim_E,)

                for E in 2:dim_E
                    N               = Integer(floor((3 * sigma* (E-1) * sqrt(step_time))))
                    mconv_int[E, 1] = mconv_int[E, 1] + m[E, k, t] * NormConv[E,N+1]

                    for j in 1:N
                        if (E+j <= dim_E)
                            mconv_int[E-j, 1] = mconv_int[E-j, 1] + m[E, k, t+1] * NormConv[E, N+1-j]
                            mconv_int[E+j, 1] = mconv_int[E+j, 1] + m[E, k, t+1] * NormConv[E, N+1+j]
                        else # Neumann condition in E_max
                            mconv_int[E-j, 1] = mconv[E-j,1] + m[E, k, t+1] * NormConv[E,N+1-j]
                            mconv_int[2 * dim_E - (E + j), 1] = mconv[2 * (dim_E)-(E+j), 1] + m[E, k, t+1] * NormConv[E, N+1+j]
                        end 
                    end
                end
                
                
                mconv[:,k] = mconv_int

            end
            

            m[:, :, t+1] = mconv[:, :, 1]

            # Determination of the price with the max_price parameter
            p[1,t] = ((Tot_prod[1,t] + max_price) / (demand * exp(rho * (t - 1) * step_time)))^(-1/elas)

        end

        return V, q, i, m, Tot_prod, Tot_inv, p

    end



    """
    Finds the equilibrium p and hence the solution of the problem: (V,q,i,m,p)

    #### Fields
        - `p_guess`    : price vector
        - `m0`         : initial firms distribution 
        - `step_time`  : unit step for T grid 
        - `T`          : final time
        - `step_E`     : unit step for E grid
        - `max_E`      : max permit to emit given by gov
        - `step_k`     : unit step for k grid
        - `k_max`      : max mitigating technology 
        - `sigma`      : intensity of the volatility of the emission induced by production 
        - `a`          : cost parameter of producing
        - `b_cost`     : cost parameter of producing 
        - `rate`       : interest rate 
        - `rho`        : growth rate of the economy 
        - `demand`     : global wealth 
        - `tax`        : tax on pollution stock 
        - `c`          : investment cost 
        - `elas`       : demand parameter from the CES demand function 
        - `max_price`  : This is not a maximum price per se but make the demand collapse and facilitate the 
                         numerical computations 
        - `NormConv`   : normal kernel for convolution

    #### Returns
        -`V`        : equilibrium value function
        -`q`        : equilibrium production policy
        -`i`        : equilibrium investemnt policy
        -`m`        : evolution of the equilibrium distribution
        -`Tot_prod` : total production at each t
        -`Tot_inv`  : total investment at each t
        -`p`        : solution price 
        -`iter`     : the number of iterations

    """
    function find_fp_Vp(p_guess, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, a,
                            b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
        
        
        nb_iter   = 90 # max iterations
    
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
                        println("k_max must be a multiple of step_k. Dimension is evaluated as its entire part.")
                        Integer(floor(k_max / step_k))
                    end
    
    
        # create array to stock the values
        V                = zeros(dim_E, dim_k, dim_t)
        q                = zeros(dim_E, dim_k, dim_t)
        i                = zeros(dim_E, dim_k, dim_t)
        Tot_prod         = zeros(nb_iter, dim_t)
        Tot_inv          = zeros(nb_iter, dim_t)
        p                = zeros(nb_iter, dim_t -1)
        p_guess_un       = zeros(1, dim_t - 1)
        pg               = zeros(1, dim_t -1)
        m                = zeros(dim_E, dim_k, dim_t)
        
        # initialization
        p_guess_un = p_guess
    
        V, q, i, m, Tot_prod_iter, Tot_inv_iter, p_iter = fun_firm_prob(p_guess, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, 
                                            a, b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
        
        Tot_prod[1, :] = Tot_prod_iter
        Tot_inv[1,:]   = Tot_inv_iter
        p[1, :]  = p_iter
        pg       = p_guess_un
        
        for i in 1:(dim_t-1)
            # update the price slowly
            p_guess_un[i] = p_guess_un[i] + max(-0.2*p_guess_un[i], min(0.2*p_guess_un[i],(p_iter[i] - p_guess_un[i])/3))
        end
        
        iter     = 2
        erreur    = 1000
        nv_erreur = maximum(abs((p_iter - pg)./pg))
        # Print the error between two steps.
        println(nv_erreur)
        
        # Start to update
        tol = 0.02
        
        println("Start loop")
    
        while ((nv_erreur > tol) && (iter <= nb_iter)) && (erreur > nv_erreur)
        
            println("iter:", iter)
            
            V, q, i, m, Tot_prod_iter, Tot_inv_iter, p_iter = fun_firm_prob(p_guess_un, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, 
                                                a, b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
            Tot_prod[iter,:] = Tot_prod_iter
            Tot_inv[iter,:]  = Tot_inv_iter
            p[iter,:]        = p_iter
            pg               = p_guess_un
            
            for i in 1:(dim_t - 1)
                # update the price slowly
                p_guess_un[i] = p_guess_un[i] + max(-0.2*p_guess_un[i], min(0.2*p_guess_un[i],(p_iter[i] - p_guess_un[i])/3))
            end
            
            erreur = nv_erreur
            iter = iter +1
            nv_erreur = maximum(abs((p_iter -pg)./pg))
            # Print the error between two steps.
            println("nv_erreur:", nv_erreur)
            
        end
        
        if (nv_erreur >= erreur)
            iter = max(iter-2,1)
        else
            iter = iter - 1
        end    
    
        return V, q, i, m, Tot_prod[iter,:], Tot_inv[iter,:], p[iter,:], iter
    
    end 

    """
    Constructs the normal kernel, the initial price, and the initial distribution for a given set of parameters
    and returns the solution for these parameters.

    #### Fields
        - `step_time`  : unit step for T grid 
        - `T`          : final time
        - `step_E`     : unit step for E grid
        - `max_E`      : max permit to emit given by gov
        - `step_k`     : unit step for k grid
        - `k_max`      : max mitigating technology 
        - `sigma`      : intensity of the volatility of the emission induced by production 
        - `a`          : cost parameter of producing
        - `b_cost`     : cost parameter of producing 
        - `rate`       : interest rate 
        - `rho`        : growth rate of the economy 
        - `demand`     : global wealth 
        - `tax`        : tax on pollution stock 
        - `c`          : investment cost 
        - `elas`       : demand parameter from the CES demand function 
        - `max_price`  : This is not a maximum price per se but make the demand collapse and facilitate the 
                         numerical computations

    #### Returns
        -`V`        : equilibrium value function
        -`q`        : equilibrium production policy
        -`i`        : equilibrium investemnt policy
        -`m`        : evolution of the equilibrium distribution
        -`Tot_prod` : total production at each t
        -`Tot_inv`  : total investment at each t
        -`p`        : solution price

    """
    function run_MFG(; sigma = 0.02, max_E = 20, step_E = 0.5, step_k = 0.25, k_max = 5, step_time = 1, T = 25, 
                    a = 2, b_cost = 8, c = 2, tax = 0.1, rate = 0.05, rho = 0.02, demand = 8, elas = 1.4,
                    max_price = 0.5)

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
    
        V, q, i, m, Tot_prod, Tot_inv, p, iter = find_fp_Vp(p_init, m0, step_time, T, step_E, max_E, step_k, k_max, sigma, a,
                            b_cost, rate, rho, demand, tax, c, elas, max_price, NormConv)
    
        return V, q, i, m, Tot_prod, Tot_inv, p
    
    end

    """
    Plots the solutions. Takes no input and return plots.
    """
    function plot_sol()
    
        # Initialization of the mean field games
        V, q, i, m, Tot_prod, Tot_inv, p                = env_comp_MFG.run_MFG()
        V_b, q_b, i_b, m_b, Tot_prod_b, Tot_inv_b, p_b  = env_comp_MFG.run_MFG(max_E = 30)
        V_t, q_t, i_t, m_t, Tot_prod_t, Tot_inv_t, p_t  = env_comp_MFG.run_MFG(max_E = 10, step_E = 0.25)
        
        dim_t = 21
        step_T = 1
        
        plot1 = Plots.plot(1:step_T:(dim_t-1), Tot_prod[1:(dim_t-1)], title="Total production", label="E = 20", xlabel = "time", ylabel = "Tot_prod")
        Plots.plot!(1:step_T:(dim_t-1), Tot_prod_b[1:(dim_t-1)], label = "E = 30")
        Plots.plot!(1:step_T:(dim_t-1), Tot_prod_t[1:(dim_t-1)], label = "E = 10")
    
        plot2 = Plots.plot(1:step_T:(dim_t-1), Tot_inv[1:(dim_t-1)], title="Total investment", label="E = 20", xlabel = "time", ylabel = "Tot_inv")
        Plots.plot!(1:step_T:(dim_t-1), Tot_inv_b[1:(dim_t-1)], label = "E = 30")
        Plots.plot!(1:step_T:(dim_t-1), Tot_inv_t[1:(dim_t-1)], label = "E = 10")
        
        plot3 = Plots.plot(1:step_T:(dim_t-1), p[1:(dim_t-1)], title="Price", label="E = 20", xlabel = "time", ylabel = "p")
        Plots.plot!(1:step_T:(dim_t-1), p_b[1:(dim_t-1)], label = "E = 30")
        Plots.plot!(1:step_T:(dim_t-1), p_t[1:(dim_t-1)], label = "E = 10")
    
        pyplot(leg=false, ticks=nothing)

        step_E = 0.5
        step_k = 0.25
        max_E = 20
        k_max = 5


        E_grid    = 0:step_E:max_E
        k_grid    = step_k:step_k:k_max
        m_dist_2  = Dierckx.Spline2D(E_grid, k_grid, m[:,:,2])
        f_2(x,y)  = m_dist_2(x,y)
        m_dist_15 = Dierckx.Spline2D(E_grid, k_grid, m[:,:,15])
        f_15(x,y) = m_dist_15(x,y)

        x    = 0:step_E/2:max_E
        y    = step_k:step_k/2:k_max

        plot4_1 = Plots.plot(x, y, f_2, st = [:surface], title = "Firm distribution at time 2")
        plot4_2 = Plots.plot(x, y, f_2, st = [:contourf], title = "Firm distribution at time 2")
        plot4_3 = Plots.plot(x, y, f_15, st = [:surface], title = "Firm distribution at time 15")
        plot4_4 = Plots.plot(x, y, f_15, st = [:contourf], title = "Firm distribution at time 15")
        plot4   = Plots.plot(plot4_1, plot4_2, plot4_3, plot4_4, layout = 4)
            
        return plot1, plot2, plot3, plot4
    
    end
    
    
end