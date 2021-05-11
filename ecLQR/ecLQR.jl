using StaticArrays, LinearAlgebra

mutable struct constraint_togo{T}
    Hx_list::Vector{Matrix{T}}    # size will change
    hl_list::Vector{Vector{T}}    # size will change
    function constraint_togo{T}(_Hx_list, _hl_list) where T
        new{T}(_Hx_list, _hl_list)
    end
end

struct ecLQR{T, nx, nu, ncxu, ncxuN, N}
    # T data type
    # nx state dimension
    # nu control dimension
    # ncxu constraint dimension
    # ncxuN final time step constraint dimension 

    # cost 
    Q_list::Vector{SizedMatrix{nx,nx,T,2,Matrix{T}}}    # Q_xx    1 --- N
    q_list::Vector{SizedVector{nx,T,Vector{T}}}        # Q_x     1 --- N
    R_list::Vector{SizedMatrix{nu,nu,T,2,Matrix{T}}}    # Q_uu    1 --- N-1
    r_list::Vector{SizedVector{nu,T,Vector{T}}}        # Q_u     1 --- N-1
    H_list::Vector{SizedMatrix{nu,nx,T,2,Matrix{T}}}    # Q_ux    1 --- N-1

    # dynamics 
    A_list::Vector{SizedMatrix{nx,nx,T,2,Matrix{T}}}    # F_xt    1 --- N
    B_list::Vector{SizedMatrix{nx,nu,T,2,Matrix{T}}}    # F_ut    1 --- N
    f_list::Vector{SizedVector{nx,T,Vector{T}}}        # f       1 --- N
    
    # constraints
    C_list::Vector{SizedMatrix{ncxu,nx,T,2,Matrix{T}}}    # G_xt    1 --- N-1
    D_list::Vector{SizedMatrix{ncxu,nu,T,2,Matrix{T}}}    # G_ut    1 --- N-1
    g_list::Vector{SizedVector{ncxu,T,Vector{T}}}        # g_lt    1 --- N-1
    CN::SizedMatrix{ncxuN,nx,T,2,Matrix{T}}               # G_xT
    gN::SizedVector{ncxuN,T,Vector{T}}                    # g_lT

    # used in calculation
    Vxx_list::Vector{SizedMatrix{nx,nx,T,2,Matrix{T}}}
    vx_list::Vector{SizedVector{nx,T,Vector{T}}}  

    contg::constraint_togo{T}

    Kx_list::Vector{SizedMatrix{nu,nx,T,2,Matrix{T}}}   # 1 --- N-1   
    kl_list::Vector{SizedVector{nu,T,Vector{T}}}       # 1 --- N-1

    # storage
    mx::SizedVector{nx,T,Vector{T}} 
    mu::SizedVector{nu,T,Vector{T}} 
    Mxx::SizedMatrix{nx,nx,T,2,Matrix{T}}
    Muu::SizedMatrix{nx,nx,T,2,Matrix{T}}
    Mux::SizedMatrix{nu,nx,T,2,Matrix{T}}
    Nx::Matrix{T}                 # size will change
    Nu::Matrix{T}                 # size will change 
    nl::Vector{T}                 # size will change


    Py::Matrix{T}                 # size will change
    Zw::Matrix{T}                 # size will change 
    yt::Vector{T}                 # size will change
    wt::Vector{T}                 # size will change

    function ecLQR(_Q_list, _q_list, _R_list, _r_list, _H_list,
                   _A_list, _B_list, _f_list,
                   _C_list, _D_list, _g_list, _CN, _gN)
        T = typeof(_gN[1])
        N = size(_Q_list)[1]
        nx,nu = size(_B_list[1])
        ncxuN = size(_gN)[1]
        ncxu = size(_g_list[1])[1]
        Vxx_list = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:N]
        vx_list =  [SizedVector{nx}(zeros(T,nx)) for i=1:N]
        Kx_list =  [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:N]
        kl_list =  [SizedVector{nu}(zeros(T,nu)) for i=1:N]
        Hx_list = [ Matrix{T}(undef,nx,nx) for i = 1:N]
        hl_list = [ Vector{T}(undef,nx) for i = 1:N]
        contg = constraint_togo{T}(Hx_list, hl_list)

        mx = SizedVector{nx}(zeros(T,nx)) 
        mu = SizedVector{nu}(zeros(T,nu)) 
        Mxx = SizedMatrix{nx,nx}(zeros(T,nx,nx))
        Muu = SizedMatrix{nu,nu}(zeros(T,nu,nu))
        Mux = SizedMatrix{nu,nx}(zeros(T,nu,nx))

        Nx = zeros(T, nx, nx)
        Nu = zeros(T, nx, nx)
        nl = zeros(T, nx)
        Py = zeros(T, nx, nx)
        Zw = zeros(T, nx, nx)
        yt = zeros(T, nx)
        wt = zeros(T, nx)


        new{T, nx, nu, ncxu, ncxuN, N}(
            _Q_list, _q_list, _R_list, _r_list, _H_list,
            _A_list, _B_list, _f_list,
            _C_list, _D_list, _g_list, _CN, _gN,
            Vxx_list, vx_list, 
            contg,
            Kx_list, kl_list,
            mx, mu, Mxx, Muu, Mux,
            Nx, Nu, nl, Py, Zw, yt, wt
        )
    end
  
end

function ecLQR_backward!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, N}) where {T, nx, nu, ncxu, ncxuN, N}
    ec.Vxx_list[N] .= ec.Q_list[N]
    ec.vx_list[N] .= ec.q_list[N]
    ec.contg.Hx_list[N] = ec.CN
    ec.contg.hl_list[N] = ec.gN

    for i=N-1:-1:1
        Fxt = ec.A_list[i];
        Fut = ec.B_list[i];
        Gxt = ec.C_list[i];
        Gut = ec.D_list[i];
        gl = ec.g_list[i];        

        # equation 12
        ec.mx .= ec.q_list[i] + Fxt'*ec.vx_list[i+1];
        ec.mu .= ec.r_list[i] + Fut'*ec.vx_list[i+1];
        ec.Mxx .= ec.Q_list[i] + Fxt'*ec.Vxx_list[i+1]*Fxt;
        ec.Muu .= ec.R_list[i] + Fut'*ec.Vxx_list[i+1]*Fut;
        ec.Mux .= ec.H_list[i] + Fut'*ec.Vxx_list[i+1]*Fxt;

        Nxt = [Gxt; ec.contg.Hx_list[i+1]*Fxt];
        Nut = [Gut; ec.contg.Hx_list[i+1]*Fut]; 
        nlt = [gl; ec.contg.Hx_list[i+1]*ec.f_list[i] + ec.contg.hl_list[i+1]];

        # svd to get P and Z, equation 13d
        U,S,V = svd(Nut);
        rankNut = length(S[S .> 1e-6])
        if (rankNut == 0)
            Z = V;
            A = Z'*ec.Muu*Z;
            b = Z';
            # equation 17 and 18
            K = -( Z*(A\b)*ec.Mux );
            k = -( Z*(A\b)*ec.mu );
            ec.kl_list[i] .= k;
            ec.Kx_list[i] .= K;
        elseif (rankNut == nu)
            P = V;
            # equation 17 and 18
            K = -( P*pinv(Nut*P)*Nxt);
            k = -( P*pinv(Nut*P)*nlt);
            ec.kl_list[i] .= k;
            ec.Kx_list[i] .= K;
        else
            P = V(:,1:rankNut);
            Z = V(:,rankNut+1:nu);
            # equation 17 and 18
            A = Z'*Muut*Z;
            b = Z';
            K = -( P*pinv(Nut*P)*Nxt + Z*(A\b)*Muxt );
            k = -( P*pinv(Nut*P)*nlt + Z*(A\b)*mult );
            ec.kl_list[i] .= k;
            ec.Kx_list[i] .= K;
        end

        # remove redudant terms, the paragraph below equation 21
        c = [ec.contg.hl_list[i+1] ec.contg.Hx_list[i+1]]
        U,_,_ = svd(c)
        c = U'*c
        rows = size(c,1);
        c = c[1:rows-rankNut,:]
        ec.contg.hl_list[i] = c[:,1]
        ec.contg.Hx_list[i] = c[:,2:end]

        # update cost to go, equation 24 and 25
        tmpmtx = ec.Mxx + 2*ec.Kx_list[i]'*ec.Mux + ec.Kx_list[i]'*ec.Muu*ec.Kx_list[i]
        ec.Vxx_list[i] .= (tmpmtx + tmpmtx') /2
        ec.vx_list[i] .= ec.mx + ec.Kx_list[i]'*ec.mu + (ec.Mux'+ ec.Kx_list[i]'*ec.Muu)*ec.kl_list[i]

    end
end