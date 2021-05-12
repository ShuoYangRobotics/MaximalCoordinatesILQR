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
    Kλ_list::Vector{SizedMatrix{ncxu,nx,T,2,Matrix{T}}}   # 1 --- N-1   
    kλ_list::Vector{SizedVector{ncxu,T,Vector{T}}}       # 1 --- N-1

    # storage
    mx::SizedVector{nx,T,Vector{T}} 
    mu::SizedVector{nu,T,Vector{T}} 
    Mxx::SizedMatrix{nx,nx,T,2,Matrix{T}}
    Muu::SizedMatrix{nu,nu,T,2,Matrix{T}}
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
        Kλ_list =  [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:N]
        kλ_list =  [SizedVector{ncxu}(zeros(T,ncxu)) for i=1:N]

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
            Kλ_list, kλ_list,
            mx, mu, Mxx, Muu, Mux,
            Nx, Nu, nl, Py, Zw, yt, wt
        )
    end
  
    function ecLQR{T}(nx,nu,ncxu,ncxuN,N) where T
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

        Vxx_list = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:N]
        vx_list =  [SizedVector{nx}(zeros(T,nx)) for i=1:N]
        Kx_list =  [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:N]
        kl_list =  [SizedVector{nu}(zeros(T,nu)) for i=1:N]
        Kλ_list =  [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:N]
        kλ_list =  [SizedVector{ncxu}(zeros(T,ncxu)) for i=1:N]
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
        Q_list, q_list, R_list, r_list, H_list,
        A_list, B_list, f_list,
        C_list, D_list, g_list, CN, gN,
        Vxx_list, vx_list, 
        contg,
        Kx_list, kl_list,
        Kλ_list, kλ_list,
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
        ec.mx .= ec.q_list[i];
        mul!(ec.mx, Transpose(Fxt), ec.vx_list[i+1], 1.0, 1.0)
        ec.mu .= ec.r_list[i]; 
        mul!(ec.mu, Transpose(Fut), ec.vx_list[i+1], 1.0, 1.0)
        
        ec.Mxx .= ec.Q_list[i] + Fxt'*ec.Vxx_list[i+1]*Fxt;
        ec.Muu .= ec.R_list[i] + Fut'*ec.Vxx_list[i+1]*Fut;
        ec.Mux .= ec.H_list[i] + Fut'*ec.Vxx_list[i+1]*Fxt;

        Nxt = [Gxt; ec.contg.Hx_list[i+1]*Fxt];
        Nut = [Gut; ec.contg.Hx_list[i+1]*Fut]; 
        nlt = [gl; ec.contg.Hx_list[i+1]*ec.f_list[i] + ec.contg.hl_list[i+1]];

        # svd to get P and Z, equation 13d
        U,S,V = svd(Nut, full=true);
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
            P = V[:,1:rankNut];
            Z = V[:,rankNut+1:nu];
            # equation 17 and 18
            A = Z'*ec.Muu*Z;
            b = Z';
            K = -( P*pinv(Nut*P)*Nxt + Z*(A\b)*ec.Mux );
            k = -( P*pinv(Nut*P)*nlt + Z*(A\b)*ec.mu );
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

# the same function, but use D E as input, return ΔV
function ecLQR_backward!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, N}, D, E) where {T, nx, nu, ncxu, ncxuN, N}
    ΔV_1 = 0
    ΔV_2 = 0
    ec.Vxx_list[N] .= E[N].Q
    ec.vx_list[N] .= E[N].q
    # last time index no constraint (k=N)
    ec.contg.Hx_list[N] = SizedMatrix{ncxu,nx}(zeros(Float64,ncxu,nx))
    ec.contg.hl_list[N] = SizedVector{ncxu}(zeros(Float64,ncxu))

    # notice the nu here is m+ncxu 
    m = nu - ncxu
    for i=N-1:-1:1
        dyn_exp = D[i]
        cost_exp = E[i]
        # Q,q,R,r,H,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.H,cost_exp.c
        # A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G

        Fxt = dyn_exp.A;
        Fut = [dyn_exp.B dyn_exp.C];
        Gxt = dyn_exp.G*dyn_exp.A;
        Gut = [dyn_exp.G*dyn_exp.B dyn_exp.G*dyn_exp.C];
        gl = SizedVector{ncxu}(zeros(Float64,ncxu))       

        # equation 12
        ec.mx .= cost_exp.q;
        mul!(ec.mx, Transpose(Fxt), ec.vx_list[i+1], 1.0, 1.0)
        ec.mu .= [cost_exp.r; zeros(ncxu)]; 
        mul!(ec.mu, Transpose(Fut), ec.vx_list[i+1], 1.0, 1.0)
        
        Raug = zeros(nu,nu)
        Raug[1:m,1:m] .= cost_exp.R

        Haug = zeros(nu,nx)
        Haug[1:m,:] = cost_exp.H

        ec.Mxx .= cost_exp.Q + Fxt'*ec.Vxx_list[i+1]*Fxt;
        ec.Muu .= Raug + Fut'*ec.Vxx_list[i+1]*Fut;
        ec.Mux .= Haug + Fut'*ec.Vxx_list[i+1]*Fxt;

        Nxt = [Gxt; ec.contg.Hx_list[i+1]*Fxt];
        Nut = [Gut; ec.contg.Hx_list[i+1]*Fut]; 
        nlt = [gl; ec.contg.Hx_list[i+1]*ec.f_list[i] + ec.contg.hl_list[i+1]];

        # svd to get P and Z, equation 13d
        U,S,V = svd(Nut, full=true);
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
            P = V[:,1:rankNut];
            Z = V[:,rankNut+1:nu];
            # equation 17 and 18
            A = Z'*ec.Muu*Z;
            b = Z';
            K = -( P*pinv(Nut*P)*Nxt + Z*(A\b)*ec.Mux );
            k = -( P*pinv(Nut*P)*nlt + Z*(A\b)*ec.mu );
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
   
        
        ΔV_1 += ec.kl_list[i]' * ec.mu
        ΔV_2 += 0.5*ec.kl_list[i]' * ec.Muu * ec.kl_list[i]
    end
    return ΔV_1, ΔV_2

end

# Follow Jan's 
function ecLQR_backward_Jan!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, N}, D, E) where {T, nx, nu, ncxu, ncxuN, N}    ΔV = 0
    ΔV_1 = 0
    ΔV_2 = 0
    ec.Vxx_list[N] .= E[N].Q
    ec.vx_list[N] .= E[N].q

    n = nx
    m = nu
    p = ncxu
    for i=N-1:-1:1
        dyn_exp = D[i]
        cost_exp = E[i]
        Q,q,R,r,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.c 
        A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G


        Dmtx = B - C/(G*C)*G*B
        M11 = R + Dmtx'*ec.Vxx_list[i+1]*B
        M12 = Dmtx'*ec.Vxx_list[i+1]*C
        M21 = G*B
        M22 = G*C

        M = [M11 M12;M21 M22]
        b = [Dmtx'*ec.Vxx_list[i+1];G]*A

        K_all = M\b
        Ku = K_all[1:m,:]
        Kλ = K_all[m .+ (1:p),:]

        l_all = M\[r + Dmtx'*ec.vx_list[i+1]; zeros(p)]
        lu = l_all[1:m]
        lλ = l_all[m .+ (1:p)]

        ec.kl_list[i] .= -lu;
        ec.Kx_list[i] .= -Ku;

        ec.Kλ_list[i] .= -Kλ;
        ec.kλ_list[i] .= -lλ;

        Abar = A -B*Ku -C*Kλ
        bbar = -B*lu 

        t1 = -lu'*r -ec.vx_list[i+1]'*(B*lu)
        t2 = 0.5*lu'*R*lu + lu'*(B'*ec.Vxx_list[i+1]*B)*lu

        ec.Vxx_list[i] .= Q + Ku'*R*Ku + Abar'*ec.Vxx_list[i+1]*Abar
        ec.vx_list[i] .= q - Ku'*r + Ku'*R*lu + Abar'*ec.Vxx_list[i+1]*bbar + Abar'*ec.vx_list[i+1]

        ΔV_1 += t1
        ΔV_2 += t2
    end
    return ΔV_1, ΔV_2
end