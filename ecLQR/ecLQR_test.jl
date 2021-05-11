
include("ecLQR.jl")
using Plots

function test_ecLQR_laine()
    # setup problem, 
    # should be the same as that in 
    # https://github.com/paulyang1990/equality-constraint-LQR-compare/blob/master/three_by_three_system_state_and_control.m

    nx = 3
    nu = 3
    ncxu = 3

    x0 = [0.0; 0.0; 0.0]
    xN = [3.0, 2.0, 1.0]

    traj_time = 1
    dt = 0.01
    N::Int = traj_time / dt

    A = I(3) + dt *[-0.4762    0.0576   -0.8775
    -0.1532   -0.9880    0.0183
    -0.8659    0.1432    0.4793]
    B = [-0.6294   -0.4978   -0.5967
         -0.3749   -0.4781    0.7943
         -0.6807    0.7236    0.1143]*dt

    Q = 1e-2 * I(nx)
    R = 1e-3 * I(nu)    # # used in calculation
    # Vxx_list::Vector{SizedMatrix{n,n,T,2,Matrix{T}}}
    # vx_list::Vector{SizedVector{n,T,Vector{T}}}  

    # Hx_list::Vector{Matrix{T}}    # size will change
    # hl_list::Vector{Vector{T}}    # size will change


    # Kx_list::Vector{SizedMatrix{m,n,T,2,Matrix{T}}}   # 1 --- N-1   
    # kl_list::Vector{SizedVector{m,T,Vector{T}}}       # 1 --- N-1

    # # storage
    # mx::SizedVector{n,T,Vector{T}} 
    # mu::SizedVector{m,T,Vector{T}} 
    # Mxx::SizedMatrix{n,n,T,2,Matrix{T}}
    # Muu::SizedMatrix{n,n,T,2,Matrix{T}}
    # Mux::SizedMatrix{m,n,T,2,Matrix{T}}
    # Nx::Matrix{T}                 # size will change
    # Nu::Matrix{T}                 # size will change 
    # nl::Vector{T}                 # size will change


    # Py::Matrix{T}                 # size will change
    # Zw::Matrix{T}                 # size will change 
    # yt::Vector{T}                 # size will change
    # wt::Vector{T}                 # size will change
    Qf = 500 * I(nx)
    Cxu = [N/2]


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
            C_list[i] = I(nx)
            D_list[i] = I(nx)
            g_list[i] = [10.0, 20.0, 30.0]
        end
    end
    Q_list[N] .= Qf
    q_list[N] .= -Qf*xN
    ec = ecLQR(Q_list, q_list, R_list, r_list, H_list,
               A_list, B_list, f_list,
               C_list, D_list, g_list, CN, gN)

    ecLQR_backward!(ec)   
    x_list = zeros(N,3)
    x_list[1,:] .= x0
    for i=1:N-1
        u = ec.Kx_list[i]*x_list[i,:] + ec.kl_list[i]
        x_list[i+1,:] .= A_list[i]*x_list[i,:] + B_list[i]*u
    end
    
    return x_list
end



x_list = test_ecLQR_laine()
N = size(x_list,1)
plot(1:N, x_list)



