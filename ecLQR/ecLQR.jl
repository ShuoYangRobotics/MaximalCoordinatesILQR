using StaticArrays, LinearAlgebra

mutable struct constraint_togo{T}
    Hx_list::Vector{Matrix{T}}    # size will change
    hl_list::Vector{Vector{T}}    # size will change
    function constraint_togo{T}(_Hx_list, _hl_list) where T
        new{T}(_Hx_list, _hl_list)
    end
end

struct ecLQR{T, nx, nu, ncxu, ncxuN, nk, N}
    # T data type
    # nx state dimension
    # nu control dimension
    # ncxu constraint dimension
    # ncxuN final time step constraint dimension 
    # nk = nk

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

    # variables usedin ec_zac 
    M::SizedMatrix{nk,nk,T,2,Matrix{T}}
    b::SizedMatrix{nk,nx,T,2,Matrix{T}}
    d::SizedVector{nk,T,Vector{T}} 
    K_all::SizedMatrix{nk,nx,T,2,Matrix{T}}
    l_all::SizedVector{nk,T,Vector{T}} 
    Ku::SizedMatrix{nu,nx,T,2,Matrix{T}}
    Kλ::SizedMatrix{ncxu,nx,T,2,Matrix{T}}
    lu::SizedVector{nu,T,Vector{T}} 
    lλ::SizedVector{ncxu,T,Vector{T}} 
    Abar::SizedMatrix{nx,nx,T,2,Matrix{T}}
    bbar::SizedVector{nx,T,Vector{T}}
    
    tmp_nxnx::Vector{SizedMatrix{nx,nx,T,2,Matrix{T}}}
    tmp_nunu::Vector{SizedMatrix{nu,nu,T,2,Matrix{T}}}
    tmp_nxnu::Vector{SizedMatrix{nx,nu,T,2,Matrix{T}}}
    tmp_nunx::Vector{SizedMatrix{nu,nx,T,2,Matrix{T}}}
    tmp_nxncxu::Vector{SizedMatrix{nx,ncxu,T,2,Matrix{T}}}
    tmp_ncxunx::Vector{SizedMatrix{ncxu,nx,T,2,Matrix{T}}}
    tmp_nuncxu::Vector{SizedMatrix{nu,ncxu,T,2,Matrix{T}}}
    tmp_ncxunu::Vector{SizedMatrix{ncxu,nu,T,2,Matrix{T}}}
    tmp_ncxuncxu::Vector{SizedMatrix{ncxu,ncxu,T,2,Matrix{T}}}

    tmp_nx::Vector{SizedVector{nx,T,Vector{T}} }
    tmp_nu::Vector{SizedVector{nu,T,Vector{T}} }
    tmp_ncxu::Vector{SizedVector{ncxu,T,Vector{T}} }
    

    function ecLQR(_Q_list, _q_list, _R_list, _r_list, _H_list,
                   _A_list, _B_list, _f_list,
                   _C_list, _D_list, _g_list, _CN, _gN)
        T = typeof(_gN[1])
        N = size(_Q_list)[1]
        nx,nu = size(_B_list[1])
        ncxuN = size(_gN)[1]
        ncxu = size(_g_list[1])[1]
        nk = nu+2*ncxu
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

        M = SizedMatrix{nk,nk}(zeros(T,nk,nk))
        b = SizedMatrix{nk,nx}(zeros(T,nk,nx))
        d = SizedVector{nk}(zeros(T,nk))
        K_all = SizedMatrix{nk,nx}(zeros(T,nk,nx))
        l_all = SizedVector{nk}(zeros(T,nk))
        Ku = SizedMatrix{nu,nx}(zeros(T,nu,nx))
        Kλ = SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx))
        lu = SizedVector{nu}(zeros(T,nu))
        lλ = SizedVector{ncxu}(zeros(T,ncxu))
        Abar = SizedMatrix{nx,nx}(zeros(T,nx,nx))
        bbar = SizedVector{nx}(zeros(T,nx))

        tmp_nxnx = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:5]
        tmp_nunu = [SizedMatrix{nu,nu}(zeros(T,nu,nu)) for i=1:5]
        tmp_nxnu = [SizedMatrix{nx,nu}(zeros(T,nx,nu)) for i=1:5]
        tmp_nunx = [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:5]
        tmp_nxncxu = [SizedMatrix{nx,ncxu}(zeros(T,nx,ncxu)) for i=1:5]
        tmp_ncxunx = [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:5]
        tmp_nuncxu = [SizedMatrix{nu,ncxu}(zeros(T,nu,ncxu)) for i=1:5]
        tmp_ncxunu = [SizedMatrix{ncxu,nu}(zeros(T,ncxu,nu)) for i=1:5]
        tmp_ncxuncxu = [SizedMatrix{ncxu,ncxu}(zeros(T,ncxu,ncxu)) for i=1:5]
        tmp_nx = [SizedVector{nx}(zeros(T,nx))  for i=1:5]
        tmp_nu = [SizedVector{nu}(zeros(T,nu))  for i=1:5]
        tmp_ncxu = [SizedVector{ncxu}(zeros(T,ncxu))  for i=1:5]


        new{T, nx, nu, ncxu, ncxuN, nk, N}(
            _Q_list, _q_list, _R_list, _r_list, _H_list,
            _A_list, _B_list, _f_list,
            _C_list, _D_list, _g_list, _CN, _gN,
            Vxx_list, vx_list, 
            contg,
            Kx_list, kl_list,
            Kλ_list, kλ_list,
            mx, mu, Mxx, Muu, Mux,
            Nx, Nu, nl, Py, Zw, yt, wt,
            M,b,d,
            K_all, l_all,
            Ku, Kλ, lu, lλ,
            Abar, bbar,
            tmp_nxnx, tmp_nunu,
            tmp_nxnu, tmp_nunx,
            tmp_nxncxu, tmp_ncxunx, tmp_nuncxu, tmp_ncxunu,
            tmp_ncxuncxu,
            tmp_nx, tmp_nu, tmp_ncxu
        )
    end
  
    function ecLQR{T}(nx,nu,ncxu,ncxuN,N) where T
        _Q_list = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:N]
        _q_list = [SizedVector{nx}(zeros(T,nx)) for i=1:N]
        _R_list = [SizedMatrix{nu,nu}(zeros(T,nu,nu)) for i=1:N-1]
        _r_list = [SizedVector{nu}(zeros(T,nu)) for i=1:N-1]
        _H_list = [SizedMatrix{nu,nx}(zeros(T,nu,nx)) for i=1:N-1]
    
        _A_list = [SizedMatrix{nx,nx}(zeros(T,nx,nx)) for i=1:N]
        _B_list = [SizedMatrix{nx,nu}(zeros(T,nx,nu)) for i=1:N]
        _f_list = [SizedVector{nx}(zeros(T,nx)) for i=1:N]
        _C_list = [SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx)) for i=1:N-1]
        _D_list = [SizedMatrix{ncxu,nu}(zeros(T,ncxu,nu)) for i=1:N-1]
        _g_list = [SizedVector{ncxu}(zeros(T,ncxu)) for i=1:N-1]
        _CN = SizedMatrix{ncxu,nx}(zeros(T,ncxu,nx))
        _gN = SizedVector{ncxu}(zeros(T,ncxu))

        ecLQR(_Q_list, _q_list, _R_list, _r_list, _H_list,
        _A_list, _B_list, _f_list,
        _C_list, _D_list, _g_list, _CN, _gN)
    end
end

function ecLQR_backward!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, nk, N}) where {T, nx, nu, ncxu, ncxuN, nk, N}
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
function ecLQR_backward!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, nk, N}, D, E) where {T, nx, nu, ncxu, ncxuN, nk, N}
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
function ecLQR_backward_Jan!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, nk, N}, D, E) where {T, nx, nu, ncxu, ncxuN, nk, N}    ΔV = 0
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

# Zac's new derivation 0512
function ecLQR_backward_Zac!(ec::ecLQR{T, nx, nu, ncxu, ncxuN, nk, N}, D, E) where {T, nx, nu, ncxu, ncxuN, nk, N}    ΔV = 0
    ΔV_1::Float64 = 0
    ΔV_2::Float64 = 0
    ec.Vxx_list[N] .= E[N].Q
    ec.vx_list[N] .= E[N].q
    β::Float64 = 1e-6

    n::Int = nx
    m::Int = nu
    p::Int = ncxu
    nm1::Int = N-1
    idx2::Int = nu+1
    idx3::Int = nu+ncxu
    idx4::Int = nu+ncxu+1
    idx5::Int = nk
    t1::Float64 = 0
    t2::Float64 = 0
    for i=nm1:-1:1
        # these should not create memories
            dyn_exp = D[i]
            cost_exp = E[i]
            Q,q,R,r,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.c 
            A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G
            Vxx = ec.Vxx_list[i+1]
            vx = ec.vx_list[i+1]

            # M         (1:nu) (idx2:idx3)(idx4:idx5)
            #    (1:nu)
            #    idx2
            #    idx3
            #    idx4
            #    idx5
            # M = [R+B'*Vxx*B   B'*Vxx*C             B'*G';
            #      C'*Vxx*B     β*I(ncxu)+C'*Vxx*C   C'*G';
            #      G*B           G*C                   -β*I(ncxu)]
            # b = [B'*Vxx;C'*Vxx;G]*A
            ec.tmp_nunu[1] .= R
            mul!(ec.tmp_nxnu[1], Vxx, B)  # Vxx*B 
            mul!(ec.tmp_nunu[1], Transpose(B), ec.tmp_nxnu[1], 1.0, 1.0)  # R+B'*Vxx*B
            ec.M[1:nu,1:nu] .= ec.tmp_nunu[1]

            mul!(ec.tmp_nxncxu[1], Vxx, C)  # Vxx*C 
            mul!(ec.tmp_nuncxu[1], Transpose(B), ec.tmp_nxncxu[1])  # B'Vxx*C 
            ec.M[1:nu,idx2:idx3] .= ec.tmp_nuncxu[1]
            ec.M[idx2:idx3,1:nu] .= Transpose(ec.tmp_nuncxu[1])


            mul!(ec.tmp_nuncxu[1], Transpose(B), Transpose(G))  # B'*G' 
            ec.M[1:nu,idx4:idx5] .= ec.tmp_nuncxu[1]
            ec.M[idx4:idx5,1:nu] .= Transpose(ec.tmp_nuncxu[1])


            mul!(ec.tmp_ncxuncxu[1], Transpose(C), Transpose(G))  # C'*G' 
            ec.M[idx2:idx3,idx4:idx5] .= ec.tmp_ncxuncxu[1]
            ec.M[idx4:idx5,idx2:idx3] .= Transpose(ec.tmp_ncxuncxu[1])

            ec.tmp_ncxuncxu[1] .= 0
            for j=1:ncxu
                ec.tmp_ncxuncxu[1][j,j] = 1.0
            end
            ec.tmp_ncxuncxu[1] .*= β

        # @time begin
            ec.tmp_ncxuncxu[2] .= ec.tmp_ncxuncxu[1]   # β*I(ncxu)
            mul!(ec.tmp_ncxuncxu[2], Transpose(C), ec.tmp_nxncxu[1], 1.0, 1.0)  # β*I(ncxu) + C'Vxx*C
            ec.M[idx2:idx3,idx2:idx3] .= ec.tmp_ncxuncxu[2]

            ec.tmp_ncxuncxu[1] .*= -1.0
            ec.M[idx4:idx5,idx4:idx5] .= ec.tmp_ncxuncxu[1]   # -β*I(ncxu)


            mul!(ec.tmp_nxnx[1], Vxx, A)  # Vxx*A
            mul!(ec.tmp_nunx[1], Transpose(B), ec.tmp_nxnx[1])  # B'Vxx*A
            mul!(ec.tmp_ncxunx[1], Transpose(C), ec.tmp_nxnx[1])  # C'Vxx*A
            mul!(ec.tmp_ncxunx[2], G, A)  # G*A
            ec.b[1:nu,:] .= ec.tmp_nunx[1]
            ec.b[idx2:idx3,:] .= ec.tmp_ncxunx[1]
            ec.b[idx4:idx5,:] .= ec.tmp_ncxunx[2]

            ec.K_all .= ec.M\ec.b
            for ii=1:m
                for jj=1:nx
                    ec.Ku[ii,jj] = ec.K_all[ii,jj]
                end
            end
            
            for ii=1:p
                for jj=1:nx
                    ec.Kλ[ii,jj] = ec.K_all[m+ii,jj]
                end
            end

            ec.tmp_nu[1] .= r
            mul!(ec.tmp_nu[1], Transpose(B), vx, 1.0 ,1.0)
            ec.d[1:nu] .= ec.tmp_nu[1]

            mul!(ec.tmp_ncxu[1], Transpose(C), vx)
            ec.d[idx2:idx3] .= ec.tmp_ncxu[1]
            
            ec.l_all .= ec.M\ec.d

            # ec.lu .= ec.l_all[1:m]
            for ii=1:m
                ec.lu[ii] = ec.l_all[ii]
            end
            # ec.lλ .= ec.l_all[m .+ (1:p)]
            for ii=1:p
                ec.lλ[ii] = ec.l_all[m+ii]
            end

            ec.kl_list[i] .= ec.lu;
            lmul!(-1.0, ec.kl_list[i])
            ec.Kx_list[i] .= ec.Ku;
            lmul!(-1.0, ec.Kx_list[i])

            ec.Kλ_list[i] .= ec.Kλ;
            lmul!(-1.0, ec.Kλ_list[i])
            ec.kλ_list[i] .= ec.lλ;
            lmul!(-1.0, ec.kλ_list[i])

            ec.Abar .= A 
            mul!(ec.Abar, B, ec.Ku, -1.0, 1.0)  # A - BKu
            mul!(ec.Abar, C, ec.Kλ, -1.0, 1.0)

            ec.bbar .= 0
            mul!(ec.bbar, B, ec.lu, -1.0, 1.0)
            mul!(ec.bbar, C, ec.lλ, -1.0, 1.0)
        # end
        # @time begin
            mul!(ec.tmp_nx[1], B, ec.lu)
            t1 = dot(ec.lu,r)  # compare to -ec.lu'*r, no memory allocation
            t1 = -1.0*t1 
            t1 -= dot(vx,ec.tmp_nx[1])


            mul!(ec.tmp_nu[1], R, ec.lu)
            t2 = dot(ec.lu,ec.tmp_nu[1])     # compare to 0.5*ec.lu'*tmp_nu[1], no memory allocation
            t2 = 0.5*t2
            mul!(ec.tmp_nxnu[1], Vxx, B)
            mul!(ec.tmp_nunu[1], Transpose(B), ec.tmp_nxnu[1])

            mul!(ec.tmp_nu[2], ec.tmp_nunu[1], ec.lu)
            t2 += dot(ec.lu, ec.tmp_nu[2])

        # end
            #ec.Vxx_list[i] .= Q + Ku'*R*Ku + Abar'*ec.Vxx_list[i+1]*Abar + β*Kλ'*Kλ
            ec.Vxx_list[i] .= Q
            mul!(ec.tmp_nunx[1], R, ec.Ku)  # R*Ku
            mul!(ec.tmp_nxnx[1], Transpose(ec.Ku), ec.tmp_nunx[1])  # Ku'*R*Ku
            ec.Vxx_list[i] .+= ec.tmp_nxnx[1]

            mul!(ec.tmp_nxnx[2], Vxx, ec.Abar)  # ec.Vxx_list[i+1]*Abar
            mul!(ec.tmp_nxnx[3], Transpose(ec.Abar), ec.tmp_nxnx[2])  # Abar'*ec.Vxx_list[i+1]*Abar
            ec.Vxx_list[i] .+= ec.tmp_nxnx[3]

            mul!(ec.tmp_nxnx[4], Transpose(ec.Kλ), ec.Kλ) #Kλ'*Kλ
            ec.tmp_nxnx[4] .*= β
            ec.Vxx_list[i] .+= ec.tmp_nxnx[4]

            #ec.vx_list[i] .= q - Ku'*r + Ku'*R*lu + β*Kλ'*lλ + Abar'*ec.Vxx_list[i+1]*bbar + Abar'*ec.vx_list[i+1]
            ec.vx_list[i] .= q
            mul!(ec.tmp_nx[1], Transpose(ec.Ku), r)  # Ku'*r
            ec.vx_list[i] .-= ec.tmp_nx[1]     # q - Ku'*r

            mul!(ec.tmp_nu[1], R, ec.lu)  # R*lu
            mul!(ec.tmp_nx[2], Transpose(ec.Ku), ec.tmp_nu[1])  # Ku'*R*lu
            ec.vx_list[i] .+= ec.tmp_nx[2]     # q - Ku'*r + Ku'*R*lu

            mul!(ec.tmp_nx[3], Transpose(ec.Kλ), ec.lλ)  # Kλ'*lλ
            ec.tmp_nx[3] .*= β
            ec.vx_list[i] .+= ec.tmp_nx[3]    # q - Ku'*r + Ku'*R*lu + β*Kλ'*lλ

            mul!(ec.tmp_nx[4], Vxx, ec.bbar)
            mul!(ec.tmp_nx[5], Transpose(ec.Abar), ec.tmp_nx[4]) #Abar'*ec.Vxx_list[i+1]*bbar
            ec.vx_list[i] .+= ec.tmp_nx[5] 

            mul!(ec.tmp_nx[1], Transpose(ec.Abar), vx)
            ec.vx_list[i] .+= ec.tmp_nx[1]                            # + Abar'*ec.vx_list[i+1]
            
            ΔV_1 += t1
            ΔV_2 += t2
    end
    return ΔV_1, ΔV_2
end