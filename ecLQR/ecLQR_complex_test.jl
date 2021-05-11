
include("ecLQR.jl")
using Plots
using MATLAB

function random_test_ecLQR_laine()
    nj = 1  # number of joint 
    nx = 13*(nj+1);
    nu = 6+nj+5*nj;
    ncxu = 5*nj;

    x0 = zeros(nx)
    xN = Vector{Float64}(1:nx)

    traj_time = 1
    dt = 0.01
    N::Int = traj_time / dt

    mat"""
        ss = rss($nx,2,$nu);
        $cA = ss.A;
        $cB = ss.B;
    """
    A = I(nx) +dt*cA
    B = dt*cB

    Q = 1e-2 * I(nx)
    R = 1e-3 * I(nu)    
    Qf = 500 * I(nx)
    Cxu = rand(1:nx,convert(Int,floor(nx/3)))


    """
        problem is 

        min \\sum x'Qx + q'x + u'Ru + r'u + u'Hx 
        s.t   xk+1 = A[k]xk + B[k]u[k]
              C[k]x_k + D[k]u_k + g[i] = 0
    """

    Q_list = [SizedMatrix{nx,nx}(zeros(Float64,nx,nx)) for i=1:N]
    q_list = [SizedVector{nx}(zeros(Float64,nx)) for i=1:N]
    R_list = [SizedMatrix{nu,nu}(zeros(Float64,nu,nu)) for i=1:N-1]
    r_list = [SizedVector{nu}(zeros(Float64,nu)) for i=1:N-1]
    H_list = [SizedMatrix{nu,nx}(zeros(Float64,nu,nx)) for i=1:N-1]

    A_list = [SizedMatrix{nx,nx}(zeros(Float64,nx,nx)) for i=1:N]
    B_list = [SizedMatrix{nx,nu}(zeros(Float64,nx,nu)) for i=1:N]
    f_list = [SizedVector{nx}(zeros(Float64,nx)) for i=1:N]
    C_list = [SizedMatrix{ncxu,nx}(zeros(Float64,ncxu,nx)) for i=1:N-1]
    D_list = [SizedMatrix{ncxu,nu}(zeros(Float64,ncxu,nu)) for i=1:N-1]
    g_list = [SizedVector{ncxu}(zeros(Float64,ncxu)) for i=1:N-1]
    CN = SizedMatrix{ncxu,nx}(zeros(Float64,ncxu,nx))
    gN = SizedVector{ncxu}(zeros(Float64,ncxu))

    for i=1:N-1
        Q_list[i] .= Q
        R_list[i] .= R
        A_list[i] .= A
        B_list[i] .= B
        if i in Cxu
            C_list[i] = randn(ncxu, nx)
            D_list[i] = randn(ncxu, nu)
            g_list[i] = randn(ncxu, 1)
        end
    end
    Q_list[N] .= Qf
    q_list[N] .= -Qf*xN
    ec = ecLQR(Q_list, q_list, R_list, r_list, H_list,
               A_list, B_list, f_list,
               C_list, D_list, g_list, CN, gN)

    @time ecLQR_backward!(ec)   
    x_list = zeros(N,nx)
    x_list[1,:] .= x0
    for i=1:N-1
        u = ec.Kx_list[i]*x_list[i,:] + ec.kl_list[i]
        x_list[i+1,:] .= A_list[i]*x_list[i,:] + B_list[i]*u
    end

    #final state difference 
    goaL_diff = norm(xN - x_list[N,:])/nx

    # constraint violation 
    con_vio = 0
    for i=1:N-1
        if i in Cxu
            u = ec.Kx_list[i]*x_list[i,:] + ec.kl_list[i]
            con_vio += norm(C_list[i]*x_list[i,:] + D_list[i]*u + g_list[i])
        end
    end
    println("final goal difference:", goaL_diff)
    println("constraint violation :", con_vio)
    
    return x_list
end

x_list = random_test_ecLQR_laine()
N = size(x_list,1)
plot(1:N, x_list,legend = false)
