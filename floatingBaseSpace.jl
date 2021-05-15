using RobotDynamics
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using SparseArrays
using BenchmarkTools
using Test
using TrajectoryOptimization
using Altro
using Plots
using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis
const CC = ConstrainedControl

# the robot is body_link     =>      arm_1     ==>     arm_2    ...   ==>      arm_nb 
# t                        joint1             joint2       ...     joint_nb
# the arm extends along positive x direction
struct FloatingSpace{R,T,n,n̄,p,nd,n̄d} <: LieGroupModelMC{R}
    body_mass::T
    body_size::T
    arm_mass::T
    arm_width::T
    arm_length::T 
    body_mass_mtx::SizedMatrix{3,3,T,2,Matrix{T}}
    arm_mass_mtx::SizedMatrix{3,3,T,2,Matrix{T}}
    body_inertias::Diagonal{T,Array{T,1}}
    arm_inertias::Diagonal{T,Array{T,1}}
    joint_directions::Vector{SizedVector{3,T,Vector{T}}}
    joint_vertices::Vector{SizedVector{6,T,Vector{T}}}   # this is a term used by Jan 
                                          # each joint has two vertices, so joint_vertices[i] is a 6x1 vector
    joint_cmat::Vector{SizedMatrix{2,3,T,2,Matrix{T}}}  # constraint matrix of joint
    g::T 

    nb::Integer     # number of arm links, total rigid bodies in the system will be nb+1
    p::Integer      # the body has no constraint, then each one more link brings in 5 more constraints because all joints are rotational
    ns::Integer     # total state size 13*(nb+1), it equals to n

    # storage for calculate dynamics 
    # g_val::SizedVector{p,T,Vector{T}}
    # Dgmtx::SizedMatrix{p,n,T,2,Matrix{T}}
    # attiG::SizedMatrix{n,n̄,T,2,Matrix{T}}
    # fdyn_vec::SizedVector{n,T,Vector{T}}
    # Dfmtx::SizedMatrix{n,nd,T,2,Matrix{T}}
    # fdyn_attiG::SizedMatrix{nd,n̄d,T,2,Matrix{T}}    
    g_val::SizedVector{p,T,Vector{T}}
    Dgmtx::SizedMatrix{p,n,T,2,Matrix{T}}
    attiG::SizedMatrix{n,n̄,T,2,Matrix{T}}
    fdyn_vec::SizedVector{n,T,Vector{T}}
    Dfmtx::SizedMatrix{n,nd,T,2,Matrix{T}}
    Dfmtx_error::SizedMatrix{n,n̄d,T,2,Matrix{T}}
    fdyn_attiG::SizedMatrix{nd,n̄d,T,2,Matrix{T}}   
    λ_tmp::SizedVector{p,T,Vector{T}}
    liestate::RD.LieState

    ## some small but used repeatedly memories
    vmat::SizedMatrix{3,4,T,2,Matrix{T}}
    hmat::SizedMatrix{4,3,T,2,Matrix{T}}

    # we know these matrices size exactly
    dldq::SArray{Tuple{16,4},T,2,64}
    dlTdq::SArray{Tuple{16,4},T,2,64}
    drdq::SArray{Tuple{16,4},T,2,64}
    drTdq::SArray{Tuple{16,4},T,2,64}

    #kron matrix used in the vec trick for calculation of drerivatives
    kron3_mtx::SizedMatrix{4,12,T,2,Matrix{T}}    # must be 4x12
    kron4_mtx::SizedMatrix{4,16,T,2,Matrix{T}}   # must be 4x16

    lmat::SizedMatrix{4,4,T,2,Matrix{T}}
    rmat::SizedMatrix{4,4,T,2,Matrix{T}}
    mat23::SizedMatrix{2,3,T,2,Matrix{T}}
    mat24::SizedMatrix{2,4,T,2,Matrix{T}}
    mat33::SizedMatrix{3,3,T,2,Matrix{T}}
    mat44::SizedMatrix{4,4,T,2,Matrix{T}}
    mat442::SizedMatrix{4,4,T,2,Matrix{T}}
    mat43::SizedMatrix{4,3,T,2,Matrix{T}}
    mat432::SizedMatrix{4,3,T,2,Matrix{T}}
    mat34::SizedMatrix{3,4,T,2,Matrix{T}}
    mat342::SizedMatrix{3,4,T,2,Matrix{T}}
    mat35::SizedMatrix{3,5,T,2,Matrix{T}}
    mat53::SizedMatrix{5,3,T,2,Matrix{T}}
    mat54::SizedMatrix{5,4,T,2,Matrix{T}}
    mat542::SizedMatrix{5,4,T,2,Matrix{T}}
    vec2::SizedVector{2,T,Vector{T}}
    vec3::SizedVector{3,T,Vector{T}}
    vec4::SizedVector{4,T,Vector{T}}
    vec5::SizedVector{5,T,Vector{T}}

    r_ainds::SizedVector{3,Int,Vector{Int}}
    v_ainds::SizedVector{3,Int,Vector{Int}}
    q_ainds::SizedVector{4,Int,Vector{Int}}
    w_ainds::SizedVector{3,Int,Vector{Int}}
    r_binds::SizedVector{3,Int,Vector{Int}}
    v_binds::SizedVector{3,Int,Vector{Int}}
    q_binds::SizedVector{4,Int,Vector{Int}}
    w_binds::SizedVector{3,Int,Vector{Int}}
    r_zinds::SizedVector{3,Int,Vector{Int}}
    v_zinds::SizedVector{3,Int,Vector{Int}}
    q_zinds::SizedVector{4,Int,Vector{Int}}
    w_zinds::SizedVector{3,Int,Vector{Int}}

    rat1::SizedVector{3,T,Vector{T}}
    vat1::SizedVector{3,T,Vector{T}}
    qat1::SizedVector{4,T,Vector{T}}
    wat1::SizedVector{3,T,Vector{T}}

    rat::SizedVector{3,T,Vector{T}}
    vat::SizedVector{3,T,Vector{T}}
    qat::SizedVector{4,T,Vector{T}}
    wat::SizedVector{3,T,Vector{T}}
    rbt::SizedVector{3,T,Vector{T}}
    vbt::SizedVector{3,T,Vector{T}}
    qbt::SizedVector{4,T,Vector{T}}
    wbt::SizedVector{3,T,Vector{T}}
    rbt1::SizedVector{3,T,Vector{T}}
    vbt1::SizedVector{3,T,Vector{T}}
    qbt1::SizedVector{4,T,Vector{T}}
    wbt1::SizedVector{3,T,Vector{T}}
    rzt1::SizedVector{3,T,Vector{T}}
    vzt1::SizedVector{3,T,Vector{T}}
    qzt1::SizedVector{4,T,Vector{T}}
    wzt1::SizedVector{3,T,Vector{T}}
    Gqamtx::SizedMatrix{5,3,T,2,Matrix{T}}
    Gqbmtx::SizedMatrix{5,3,T,2,Matrix{T}}
    Gqzmtx::SizedMatrix{5,3,T,2,Matrix{T}}
    dGqaTλdqa::SizedMatrix{3,4,T,2,Matrix{T}}
    dGqaTλdqb::SizedMatrix{3,4,T,2,Matrix{T}}
    dGqbTλdqa::SizedMatrix{3,4,T,2,Matrix{T}}
    dGqbTλdqb::SizedMatrix{3,4,T,2,Matrix{T}}

    ## variables used in discrete_dynamics (MC)

    err_vec::Vector{T} 
    G::SizedMatrix{p,n̄,T,2,Matrix{T}}
    Fdyn::SizedMatrix{n,n̄d,T,2,Matrix{T}}
    F::SparseMatrixCSC{T,Int}      
    Δx⁺::Vector{T} 
    Δs::Vector{T}       
    FΔs::Vector{T}    


    function FloatingSpace{R,T}(nb,_joint_directions) where {R<:Rotation, T<:Real}
        # problem size
        n = 13*(nb+1)
        n̄ = 12*(nb+1)
        nc = 5          #size of one joint constraint 
        np = nc*nb        # all joints are revolute
        ns = n
        nu = 6+nb       # size of all control ut
        nd = n*2 + nu + np  # the size of [xt1; xt; ut; λt_block]
        n̄d = n̄*2 + nu + np  # the size of the error state of [xt1; xt; ut; λt_block]

        m0 = 10.0
        m1 = 1.0
        g = 0      # in space, no gravity!
        body_size = 0.5
        arm_length = 1.0
        joint_vertices = [[body_size/2, 0, 0, -arm_length/2, 0, 0]]
        for i=1:nb-1
            push!(joint_vertices,[arm_length/2, 0, 0, -arm_length/2, 0, 0])
        end

        joint_cmat = [zeros(T,2,3) for i=1:nb]
        for i=1:nb
            if _joint_directions[i] == [0,0,1.0]
                joint_cmat[i] .= [0 1.0 0; 
                                  1.0 0 0]
            elseif _joint_directions[i] == [0,1.0, 0]
                joint_cmat[i] .= [0 0 1.0; 
                                  1 0 0]
            else _joint_directions[i] == [1.0, 0, 0]
                joint_cmat[i] .= [0 0 1.0; 
                                  0 1 0]
            end
        end

        g_val = SizedVector{np}(zeros(T,np))
        Dgmtx = SizedMatrix{np,n}(zeros(T,np,n))
        attiG = SizedMatrix{n,n̄}(zeros(T,n,n̄))
        fdyn_vec = SizedVector{n}(zeros(T,n))
        Dfmtx = SizedMatrix{n,nd}(zeros(n,nd))
        Dfmtx_error = SizedMatrix{n,n̄d}(zeros(n,n̄d))
        fdyn_attiG = SizedMatrix{nd,n̄d}(zeros(T,nd,n̄d))
        body_mass_mtx = diagm([m0,m0,m0])
        arm_mass_mtx = diagm([m1,m1,m1])
        λ_tmp = SizedVector{np}(zeros(T,np))
        liestate = RD.LieState(R, (6,fill(9, nb)..., 3))

        vmat = SizedMatrix{3,4}(zeros(T,3,4))
        vmat[1,2] = 1.0
        vmat[2,3] = 1.0
        vmat[3,4] = 1.0
        hmat = SizedMatrix{4,3}(zeros(T,4,3))
        hmat[2,1] = 1.0
        hmat[3,2] = 1.0
        hmat[4,3] = 1.0

        dldq = SA{T}[
                1 0 0 0
                0 1 0 0
                0 0 1 0
                0 0 0 1
        
                0 -1 0 0
                1 0 0 0
                0 0 0 1
                0 0 -1 0
        
                0 0 -1 0
                0 0 0 -1
                1 0 0 0
                0 1 0 0
                
                0 0 0 -1
                0 0 1 0
                0 -1 0 0
                1 0 0 0
            ]
        dlTdq = SA{T}[
                1 0 0 0
                0 -1 0 0
                0 0 -1 0
                0 0 0 -1
                
                0 1 0 0
                1 0 0 0
                0 0 0 -1
                0 0 1 0
                
                0 0 1 0
                0 0 0 1
                1 0 0 0
                0 -1 0 0
                
                0 0 0 1
                0 0 -1 0
                0 1 0 0
                1 0 0 0
            ]
        drdq = SA{T}[
                1 0 0 0
                0 1 0 0
                0 0 1 0
                0 0 0 1
                
                0 -1 0 0
                1 0 0 0
                0 0 0 -1
                0 0 1 0
                
                0 0 -1 0
                0 0 0 1
                1 0 0 0
                0 -1 0 0
                
                0 0 0 -1
                0 0 -1 0
                0 1 0 0
                1 0 0 0
            ]
        drTdq = SA{T}[
                1 0 0 0
                0 -1 0 0
                0 0 -1 0
                0 0 0 -1
                
                0 1 0 0
                1 0 0 0
                0 0 0 1
                0 0 -1 0
                
                0 0 1 0
                0 0 0 -1
                1 0 0 0
                0 1 0 0
                
                0 0 0 1
                0 0 1 0
                0 -1 0 0
                1 0 0 0
            ]

        kron3_mtx = SizedMatrix{4,12}(zeros(T,4,12))
        kron4_mtx = SizedMatrix{4,16}(zeros(T,4,16))

        lmat = SizedMatrix{4,4}(zeros(T,4,4))
        rmat = SizedMatrix{4,4}(zeros(T,4,4))
        mat23 = SizedMatrix{2,3}(zeros(T,2,3))
        mat24 = SizedMatrix{2,4}(zeros(T,2,4))
        mat33 = SizedMatrix{3,3}(zeros(T,3,3))
        mat44 = SizedMatrix{4,4}(zeros(T,4,4))
        mat442 = SizedMatrix{4,4}(zeros(T,4,4))
        mat43 = SizedMatrix{4,3}(zeros(T,4,3))
        mat432 = SizedMatrix{4,3}(zeros(T,4,3))
        mat34 = SizedMatrix{3,4}(zeros(T,3,4))
        mat342 = SizedMatrix{3,4}(zeros(T,3,4))
        mat35 = SizedMatrix{3,5}(zeros(T,3,5))
        mat53 = SizedMatrix{5,3}(zeros(T,5,3))
        mat54 = SizedMatrix{5,4}(zeros(T,5,4))
        mat542 = SizedMatrix{5,4}(zeros(T,5,4))
        vec2 = SizedVector{2}(zeros(T,2))
        vec3 = SizedVector{3}(SizedVector{3}(zeros(T,3)))
        vec4 = SizedVector{4}(zeros(T,4))
        vec5 = SizedVector{5}(zeros(T,5))

        r_ainds = SizedVector{3}(zeros(Int,3))
        v_ainds = SizedVector{3}(zeros(Int,3))
        q_ainds = SizedVector{4}(zeros(Int,4))
        w_ainds = SizedVector{3}(zeros(Int,3))
        r_binds = SizedVector{3}(zeros(Int,3))
        v_binds = SizedVector{3}(zeros(Int,3))
        q_binds = SizedVector{4}(zeros(Int,4))
        w_binds = SizedVector{3}(zeros(Int,3))
        r_zinds = SizedVector{3}(zeros(Int,3))
        v_zinds = SizedVector{3}(zeros(Int,3))
        q_zinds = SizedVector{4}(zeros(Int,4))
        w_zinds = SizedVector{3}(zeros(Int,3))
    
        rat1 = SizedVector{3}(zeros(T,3))
        vat1 = SizedVector{3}(zeros(T,3))
        qat1 = SizedVector{4}(zeros(T,4))
        wat1 = SizedVector{3}(zeros(T,3))
    
        rat = SizedVector{3}(zeros(T,3))
        vat = SizedVector{3}(zeros(T,3))
        qat = SizedVector{4}(zeros(T,4))
        wat = SizedVector{3}(zeros(T,3))
        rbt = SizedVector{3}(zeros(T,3))
        vbt = SizedVector{3}(zeros(T,3))
        qbt = SizedVector{4}(zeros(T,4))
        wbt = SizedVector{3}(zeros(T,3))
        rbt1 = SizedVector{3}(zeros(T,3))
        vbt1 = SizedVector{3}(zeros(T,3))
        qbt1 = SizedVector{4}(zeros(T,4))
        wbt1 = SizedVector{3}(zeros(T,3))
        rzt1 = SizedVector{3}(zeros(T,3))
        vzt1 = SizedVector{3}(zeros(T,3))
        qzt1 = SizedVector{4}(zeros(T,4))
        wzt1 = SizedVector{3}(zeros(T,3))
        Gqamtx = SizedMatrix{5,3}(zeros(T,5,3))
        Gqbmtx = SizedMatrix{5,3}(zeros(T,5,3))
        Gqzmtx = SizedMatrix{5,3}(zeros(T,5,3))
        dGqaTλdqa = SizedMatrix{3,4}(zeros(T,3,4))
        dGqaTλdqb = SizedMatrix{3,4}(zeros(T,3,4))
        dGqbTλdqa = SizedMatrix{3,4}(zeros(T,3,4))
        dGqbTλdqb = SizedMatrix{3,4}(zeros(T,3,4))


        err_vec = zeros(ns+np)
        G = SizedMatrix{np,n̄}(zeros(T,np,n̄))
        Fdyn = SizedMatrix{n,n̄d}(zeros(T,n,n̄d))
        F = spzeros(n+np,n̄+np)       
        Δx⁺ = zeros(n̄)
        Δs = zeros(n̄+np)        
        FΔs = zeros(n+np)  

        new{R,T,n,n̄,np,nd,n̄d}(m0, body_size, m1, 0.1, arm_length, 
            body_mass_mtx, arm_mass_mtx,
            Diagonal(1 / 12 * m0 * diagm([0.5^2 + 0.5^2;0.5^2 + 0.5^2;0.5^2 + 0.5^2])),  # body inertia
            Diagonal(1 / 12 * m1 * diagm([0.1^2 + 0.1^2;0.1^2 + 1.0^2;1.0^2 + 0.1^2])),   # arm inertia
            _joint_directions,
            joint_vertices,
            joint_cmat,
            0, # space robot, no gravity
            nb, np,       # 5 because we assume all joints are revolute joints
            13*(nb+1),
            g_val,
            Dgmtx,
            attiG,
            fdyn_vec,
            Dfmtx,
            Dfmtx_error,
            fdyn_attiG,
            λ_tmp,
            liestate,
            vmat,
            hmat,
            dldq,
            dlTdq,
            drdq,
            drTdq,
            kron3_mtx,
            kron4_mtx,
            lmat,
            rmat,
            mat23,
            mat24,
            mat33,
            mat44,
            mat442,
            mat43,
            mat432,
            mat34,
            mat342,
            mat35,
            mat53,
            mat54,
            mat542,
            vec2,
            vec3,
            vec4,
            vec5,
            r_ainds,
            v_ainds,
            q_ainds,
            w_ainds,
            r_binds,
            v_binds,
            q_binds,
            w_binds,
            r_zinds,
            v_zinds,
            q_zinds,
            w_zinds,
        
            rat1,
            vat1,
            qat1,
            wat1,
        
            rat,
            vat,
            qat,
            wat,
            rbt,
            vbt,
            qbt,
            wbt,
            rbt1,
            vbt1,
            qbt1,
            wbt1,
            rzt1,
            vzt1,
            qzt1,
            wzt1,
            Gqamtx,
            Gqbmtx,
            Gqzmtx,
            dGqaTλdqa,
            dGqaTλdqb,
            dGqbTλdqa,
            dGqbTλdqb,


            err_vec,
            G,
            Fdyn,
            F,      
            Δx⁺,
            Δs,        
            FΔs 
        )
    end
    function FloatingSpace()
        _joint_directions = [[0.0;0.0;1.0]]
        FloatingSpace{UnitQuaternion{Float64},Float64}(1,_joint_directions)
    end

    # all z axis rotation
    function FloatingSpace(nb::Integer)
        @assert nb >= 1
        _joint_directions = [[0.0;0.0;1.0] for i=1:nb]
        FloatingSpace{UnitQuaternion{Float64},Float64}(nb,_joint_directions)
    end
end 

# joint axes are orthogonal to each other #space snake?
# Z -> Y -> Z -> Y ...
function FloatingSpaceOrth(nb::Integer)
    @assert nb >= 1
    _joint_directions = [ i % 2 == 0 ? [0.0;1.0;0.0] : [0.0;0.0;1.0] for i=1:nb]
    FloatingSpace{UnitQuaternion{Float64},Float64}(nb,_joint_directions)
end


# arrange state as Jan
# x v q w, x v q w,...   for each q, q[1] == w  q[2] == x q[3] == y  q[4] ==  z
#1,2,3, 4,5,6, 7,8,9,10, 11,12,13
Altro.config_size(model::FloatingSpace) = 7*(model.nb+1)
Lie_P(model::FloatingSpace) = (6,fill(9, model.nb)..., 3)
RD.LieState(model::FloatingSpace{R}) where R = RD.LieState(R,Lie_P(model))
RD.control_dim(model::FloatingSpace) = model.nb + 6  # 6 because we assume the body is fully actuated

# extract the x v q w indices of ith link
function fullargsinds(model::FloatingSpace, i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end

function statea_inds!(model::FloatingSpace, i)
    # x, v, q, ω
    model.r_ainds .= 13*(i-1); model.r_ainds[1] += 1; model.r_ainds[2] += 2; model.r_ainds[3] += 3;
    model.v_ainds .= 13*(i-1); model.v_ainds[1] += 4; model.v_ainds[2] += 5; model.v_ainds[3] += 6; 
    model.q_ainds .= 13*(i-1); model.q_ainds[1] += 7; model.q_ainds[2] += 8; model.q_ainds[3] += 9; model.q_ainds[4] += 10;  
    model.w_ainds .= 13*(i-1); model.w_ainds[1] += 11; model.w_ainds[2] += 12; model.w_ainds[3] += 13; 
end

function stateb_inds!(model::FloatingSpace, i)
    # x, v, q, ω
    model.r_binds .= 13*(i-1); model.r_binds[1] += 1; model.r_binds[2] += 2; model.r_binds[3] += 3;
    model.v_binds .= 13*(i-1); model.v_binds[1] += 4; model.v_binds[2] += 5; model.v_binds[3] += 6; 
    model.q_binds .= 13*(i-1); model.q_binds[1] += 7; model.q_binds[2] += 8; model.q_binds[3] += 9; model.q_binds[4] += 10;  
    model.w_binds .= 13*(i-1); model.w_binds[1] += 11; model.w_binds[2] += 12; model.w_binds[3] += 13; 
end

function statez_inds!(model::FloatingSpace, i)
    # x, v, q, ω
    model.r_zinds .= 13*(i-1); model.r_zinds[1] += 1; model.r_zinds[2] += 2; model.r_zinds[3] += 3;
    model.v_zinds .= 13*(i-1); model.v_zinds[1] += 4; model.v_zinds[2] += 5; model.v_zinds[3] += 6; 
    model.q_zinds .= 13*(i-1); model.q_zinds[1] += 7; model.q_zinds[2] += 8; model.q_zinds[3] += 9; model.q_zinds[4] += 10;  
    model.w_zinds .= 13*(i-1); model.w_zinds[1] += 11; model.w_zinds[2] += 12; model.w_zinds[3] += 13; 
end

# state config x v q w 
# the body pose is the position and z orientation of the body link, then rotations are rotation matrices of joint angles
function generate_config(model::FloatingSpace, body_pose, rotations)
    # com position of the body link 
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))   # TODO: improve body_pose to contain full rotations?

    state = [pin;zeros(3);RS.params(prev_q);zeros(3)]
    pin = pin+prev_q * [model.body_size/2,0,0]   
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end
    return state
end

# body pose is x y z thetaz, θ is just angle 
function generate_config(model::FloatingSpace, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    @assert length(θ) == model.nb
    rotations = []
    for i=1:length(θ)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(θ[i], axis[1], axis[2], axis[3]))
        )
    end
    return generate_config(model, body_pose, rotations)
end

### add random velocity to generated state
# state config x v q w 
# the body pose is the position and z orientation of the body link, then rotations are rotation matrices of joint angles
function generate_config_with_rand_vel(model::FloatingSpace, body_pose, rotations)
    # com position of the body link 
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))   # TODO: improve body_pose to contain full rotations?

    state = [pin;0.01*randn(3);RS.params(prev_q);0.01*randn(3)]
    pin = pin+prev_q * [model.body_size/2,0,0]   
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;0.01*randn(3);RS.params(link_q);0.01*randn(3)]

        prev_q = link_q
        pin += 2*delta
    end
    return state
end

# body pose is x y z thetaz, θ is just angle 
function generate_config_with_rand_vel(model::FloatingSpace, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    @assert length(θ) == model.nb
    rotations = []
    for i=1:length(θ)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(θ[i], axis[1], axis[2], axis[3]))
        )
    end
    return generate_config_with_rand_vel(model, body_pose, rotations)
end

# begin
#     # basic state genereation test
#     model = FloatingSpace()
#     x0 = generate_config(model, [2.0;2.0;1.0;0], [0])
#     println(x0)
# end

# this function returns a mech object, which is a constrainedDynamics object so that we can visualize the robot using 
# constraineddynamics viz
function vis_mech_generation(model::FloatingSpace)
    origin = CD.Origin{Float64}()
    link0 = CD.Box(model.body_size, model.body_size, model.body_size, 1., color = CD.RGBA(1., 1., 0.))
    link0.m = model.body_mass # set base mass
    world2base = CD.EqualityConstraint(Floating(origin, link0)) # free floating
    
    arm_links = [CD.Box(model.arm_length, model.arm_width, model.arm_width, 1.,color = CD.RGBA(0.1*i, 0.2*i, 1.0/i)) for i = 1:model.nb]
    # joint 1 
    vert01 = model.joint_vertices[1][1:3] # connection offset from body to joint1
    vert11 = model.joint_vertices[1][4:6] # connection offset from arm_link1 to joint1

    joint1 = CD.EqualityConstraint(CD.Revolute(link0, arm_links[1], model.joint_directions[1]; p1=vert01,p2=vert11)) # joint1 : body to link1

    links = [link0; arm_links]
    constraints = [world2base; joint1]
    if model.nb > 1
        for i=2:model.nb
            vert01 = model.joint_vertices[i][1:3] # connection offset from armi-1 to jointi
            vert11 = model.joint_vertices[i][4:6] # connection offset from armi to jointi
        
            jointi = CD.EqualityConstraint(CD.Revolute(arm_links[i-1], arm_links[i], model.joint_directions[i]; p1=vert01,p2=vert11)) # joint1 : armi-1 to larmi
            constraints = [constraints;jointi]
        end
    end
    mech = CD.Mechanism(origin, links, constraints, g=-model.g)
    return mech
end

function setStates!(model::FloatingSpace, mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(model,i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

# test: visualize 
# begin
#     model = FloatingSpaceOrth(2)
#     x0 = generate_config(model, [2.0;2.0;1.0;pi/2], [pi/4,pi/4]);
#     println(reshape(x0,(13,model.nb+1))')
#     mech = vis_mech_generation(model)
#     setStates!(model,mech,x0)
#     steps = Base.OneTo(1)
#     storage = CD.Storage{Float64}(steps,length(mech.bodies))
#     for i=1:model.nb+1
#         storage.x[i][1] = mech.bodies[i].state.xc
#         storage.v[i][1] = mech.bodies[i].state.vc
#         storage.q[i][1] = mech.bodies[i].state.qc
#         storage.ω[i][1] = mech.bodies[i].state.ωc
#     end
#     visualize(mech,storage, env = "editor")
# end

#inplace rotation related calculate
function q_lmult!(model::FloatingSpace, q::AbstractVector{T}) where T
    model.lmat[1,1] =  q[1]; model.lmat[1,2] = -q[2]; model.lmat[1,3] = -q[3]; model.lmat[1,4] = -q[4];    
    model.lmat[2,1] =  q[2]; model.lmat[2,2] =  q[1]; model.lmat[2,3] = -q[4]; model.lmat[2,4] =  q[3];    
    model.lmat[3,1] =  q[3]; model.lmat[3,2] =  q[4]; model.lmat[3,3] =  q[1]; model.lmat[3,4] = -q[2];   
    model.lmat[4,1] =  q[4]; model.lmat[4,2] = -q[3]; model.lmat[4,3] =  q[2]; model.lmat[4,4] =  q[1];   
end
function q_rmult!(model::FloatingSpace, q::AbstractVector{T}) where T
    model.rmat[1,1] =  q[1]; model.rmat[1,2] = -q[2]; model.rmat[1,3] = -q[3]; model.rmat[1,4] = -q[4];    
    model.rmat[2,1] =  q[2]; model.rmat[2,2] =  q[1]; model.rmat[2,3] =  q[4]; model.rmat[2,4] = -q[3];    
    model.rmat[3,1] =  q[3]; model.rmat[3,2] = -q[4]; model.rmat[3,3] =  q[1]; model.rmat[3,4] =  q[2];   
    model.rmat[4,1] =  q[4]; model.rmat[4,2] =  q[3]; model.rmat[4,3] = -q[2]; model.rmat[4,4] =  q[1];   
end
function q_lmult_T!(model::FloatingSpace, q::AbstractVector{T}) where T
    model.lmat[1,1] =  q[1]; model.lmat[1,2] =  q[2]; model.lmat[1,3] =  q[3]; model.lmat[1,4] =  q[4];    
    model.lmat[2,1] = -q[2]; model.lmat[2,2] =  q[1]; model.lmat[2,3] =  q[4]; model.lmat[2,4] = -q[3];    
    model.lmat[3,1] = -q[3]; model.lmat[3,2] = -q[4]; model.lmat[3,3] =  q[1]; model.lmat[3,4] =  q[2];   
    model.lmat[4,1] = -q[4]; model.lmat[4,2] =  q[3]; model.lmat[4,3] = -q[2]; model.lmat[4,4] =  q[1];   
end
function q_rmult_T!(model::FloatingSpace, q::AbstractVector{T}) where T
    model.rmat[1,1] =  q[1]; model.rmat[1,2] =  q[2]; model.rmat[1,3] =  q[3]; model.rmat[1,4] =  q[4];    
    model.rmat[2,1] = -q[2]; model.rmat[2,2] =  q[1]; model.rmat[2,3] = -q[4]; model.rmat[2,4] =  q[3];    
    model.rmat[3,1] = -q[3]; model.rmat[3,2] =  q[4]; model.rmat[3,3] =  q[1]; model.rmat[3,4] = -q[2];   
    model.rmat[4,1] = -q[4]; model.rmat[4,2] = -q[3]; model.rmat[4,3] =  q[2]; model.rmat[4,4] =  q[1];   
end

# the position constraint g
function g(model::FloatingSpace, x)
    # we have nb joints, so the dimension of constraint is p=5*nb
    g_val = zeros(eltype(x),model.p)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        r_b = SVector{3}(x[r_binds])
        q_a = SVector{4}(x[q_ainds])
        q_b = SVector{4}(x[q_binds])

        val = view(g_val, (5*(i-2)).+(1:5))
        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]

        val[1:3] = (r_b + RS.vmat()*RS.rmult(q_b)'*RS.lmult(q_b)*RS.hmat()*vertex2) - 
        (r_a + RS.vmat()*RS.rmult(q_a)'*RS.lmult(q_a)*RS.hmat()*vertex1)
        tmp = RS.vmat()*RS.lmult(q_a)'*q_b
        # the joint constraint map, it depends on the joint rotation direction 
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end
        val[4:5] = cmat*tmp  
    end
    return g_val
end

# the position constraint g
function g!(model::FloatingSpace, x)
    # we have nb joints, so the dimension of constraint is p=5*nb
    # g_val = zeros(eltype(x),model.p)
    n::Int = model.nb+1
    prev::Int = 0
    for i=2:n   # i is the rigidbody index
        prev = 5*(i-2)
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        statea_inds!(model,i-1)
        # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        stateb_inds!(model,i)

        model.rat[1] = x[model.r_ainds[1]]
        model.rat[2] = x[model.r_ainds[2]]
        model.rat[3] = x[model.r_ainds[3]]

        model.rbt[1] = x[model.r_binds[1]]
        model.rbt[2] = x[model.r_binds[2]]
        model.rbt[3] = x[model.r_binds[3]]

        model.qat[1] = x[model.q_ainds[1]]
        model.qat[2] = x[model.q_ainds[2]]
        model.qat[3] = x[model.q_ainds[3]]
        model.qat[4] = x[model.q_ainds[4]]

        model.qbt[1] = x[model.q_binds[1]]
        model.qbt[2] = x[model.q_binds[2]]
        model.qbt[3] = x[model.q_binds[3]]
        model.qbt[4] = x[model.q_binds[4]]
     
        model.vec3[1] = model.joint_vertices[i-1][4]
        model.vec3[2] = model.joint_vertices[i-1][5]
        model.vec3[3] = model.joint_vertices[i-1][6]

        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][4:6]
        q_rmult_T!(model, model.qbt)                                  # RS.rmult(q_b)'
        q_lmult!(model, model.qbt)                                    # RS.lmult(q_b)
        mul!(model.mat44, model.rmat, model.lmat)                     # RS.rmult(q_a)'*RS.lmult(q_b)
        mul!(model.mat34, model.vmat, model.mat44)                    # model.vmat*RS.rmult(q_a)'*RS.lmult(q_b)
        mul!(model.vec3, model.mat34, model.vec4)                     # model.vmat*RS.rmult(q_b)'*RS.lmult(q_b)*model.hmat*model.joint_vertices[i-1][4:6]

        # model.g_val[(5*(i-2)).+(1:3)] = model.rbt
        model.g_val[prev+1] = model.rbt[1]
        model.g_val[prev+2] = model.rbt[2]
        model.g_val[prev+3] = model.rbt[3]

        model.g_val[prev+1] = model.g_val[prev+1] + model.vec3[1]
        model.g_val[prev+2] = model.g_val[prev+2] + model.vec3[2]
        model.g_val[prev+3] = model.g_val[prev+3] + model.vec3[3]

        model.vec3[1] = model.joint_vertices[i-1][1]
        model.vec3[2] = model.joint_vertices[i-1][2]
        model.vec3[3] = model.joint_vertices[i-1][3]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][1:3]
        q_lmult!(model, model.qat)                                    # RS.lmult(q_a)
        q_rmult_T!(model, model.qat)                                  # RS.rmult(q_a)'
        mul!(model.mat44, model.rmat, model.lmat)                     # RS.rmult(q_a)'*RS.lmult(q_a)
        mul!(model.mat34, model.vmat, model.mat44)                    # model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)
        mul!(model.vec3, model.mat34, model.vec4)                     # model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)*model.hmat*model.joint_vertices[i-1][1:3]

        model.g_val[prev+1] = model.g_val[prev+1] - model.rat[1]
        model.g_val[prev+2] = model.g_val[prev+2] - model.rat[2]
        model.g_val[prev+3] = model.g_val[prev+3] - model.rat[3]

        model.g_val[prev+1] = model.g_val[prev+1] - model.vec3[1]
        model.g_val[prev+2] = model.g_val[prev+2] - model.vec3[2]
        model.g_val[prev+3] = model.g_val[prev+3] - model.vec3[3]

        # model.g_val[(5*(i-2)).+(1:3)] = (r_b + model.vmat*RS.rmult(q_b)'*RS.lmult(q_b)*model.hmat*model.joint_vertices[i-1][4:6]) - 
        # (r_a + model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)*model.hmat*model.joint_vertices[i-1][1:3])

        q_lmult_T!(model, model.qat) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_a)'
        mul!(model.vec3, model.mat34, model.qbt)                     # model.vmat*RS.lmult(q_a)'*model.qbt
        model.mat23 .= model.joint_cmat[i-1]
        mul!(model.vec2, model.mat23, model.vec3)          # model.joint_cmat[i-1]*tmp  
        model.g_val[prev+4] = model.vec2[1]
        model.g_val[prev+5] = model.vec2[2]
        # tmp = model.vmat*RS.lmult(q_a)'*model.qbt
        # model.g_val[(5*(i-2)).+(4:5)] = model.joint_cmat[i-1]*tmp  
    end
    return 
end

# jacobian of g, treat quaternion as normal 4 vectors
function Dg(model::FloatingSpace, x)
    Dgmtx = spzeros(model.p,model.ns)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body

        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end

        Dgblock = view(Dgmtx, (5*(i-2)).+(1:5),:)

        q_a = SVector{4}(x[q_ainds])
        q_b = SVector{4}(x[q_binds])
        Dgblock[:,r_ainds] = [-I;zeros(2,3)]  # dg/dra
        Dgblock[:,r_binds]  = [I;zeros(2,3)] # dg/drb
        Dgblock[:,q_ainds] = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertex1);
                                -cmat*RS.vmat()*RS.lmult(q_b)'
                               ]
        Dgblock[:,q_binds] = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertex2);
                                cmat*RS.vmat()*RS.lmult(q_a)'
                               ]
    end
    return Dgmtx
end

function Dg!(model::FloatingSpace, x)
    # Dgmtx = spzeros(model.p,model.ns)
    n::Int=model.nb+1
    prev::Int = 0
    for i=2:n   # i is the rigidbody index
        prev = 5*(i-2)
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body

        statea_inds!(model,i-1)
        # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        stateb_inds!(model,i)

        model.qat[1] = x[model.q_ainds[1]]
        model.qat[2] = x[model.q_ainds[2]]
        model.qat[3] = x[model.q_ainds[3]]
        model.qat[4] = x[model.q_ainds[4]]

        model.qbt[1] = x[model.q_binds[1]]
        model.qbt[2] = x[model.q_binds[2]]
        model.qbt[3] = x[model.q_binds[3]]
        model.qbt[4] = x[model.q_binds[4]]

        model.Dgmtx[prev+1,model.r_ainds[1]] = -1  # dg/dra
        model.Dgmtx[prev+2,model.r_ainds[2]] = -1  # dg/dra
        model.Dgmtx[prev+3,model.r_ainds[3]] = -1  # dg/dra

        model.Dgmtx[prev.+(4:5),model.r_ainds] .= 0

        model.Dgmtx[prev+1,model.r_binds[1]] = 1  # dg/dra
        model.Dgmtx[prev+2,model.r_binds[2]] = 1  # dg/dra
        model.Dgmtx[prev+3,model.r_binds[3]] = 1  # dg/dra

        model.Dgmtx[prev.+(4:5),model.r_binds] .= 0


        model.vec3[1] = model.joint_vertices[i-1][1]
        model.vec3[2] = model.joint_vertices[i-1][2]
        model.vec3[3] = model.joint_vertices[i-1][3]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][1:3]
        q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
        model.mat44 .= model.rmat
        q_rmult_T!(model, model.qat)                                  # RS.rmult(model.qat)'
        mul!(model.mat34, model.vmat, model.rmat)                     # model.vmat*RS.rmult(model.qat)'
        mul!(model.mat342, model.mat34, model.mat44)          # model.vmat*RS.rmult(model.qat)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])

        model.mat342 .*= -2.0
        model.Dgmtx[prev.+(1:3),model.q_ainds] .= model.mat342

        q_lmult_T!(model, model.qbt) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
        mul!(model.mat24, model.joint_cmat[i-1], model.mat34) 
        model.mat24 .*= -1.0
        model.Dgmtx[prev.+(4:5),model.q_ainds] .= model.mat24
        # model.Dgmtx[(5*(i-2)).+(1:5),model.q_ainds] = [-2*model.vmat*RS.rmult(model.qbt)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3]);
        #                         -model.joint_cmat[i-1]*model.vmat*RS.lmult(q_b)'
        #                        ]

        model.vec3[1] = model.joint_vertices[i-1][4]
        model.vec3[2] = model.joint_vertices[i-1][5]
        model.vec3[3] = model.joint_vertices[i-1][6]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][4:6]
        q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][4:6])
        model.mat44 .= model.rmat
        q_rmult_T!(model, model.qbt)                                  # RS.rmult(model.qbt)'
        mul!(model.mat34, model.vmat, model.rmat)                     # model.vmat*RS.rmult(model.qbt)'
        mul!(model.mat342, model.mat34, model.mat44)          #  2*model.vmat*RS.rmult(model.qbt)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])

        model.mat342 .*= 2.0
        model.Dgmtx[prev.+(1:3),model.q_binds] .= model.mat342

        q_lmult_T!(model, model.qat) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
        mul!(model.mat24, model.joint_cmat[i-1], model.mat34) 
        model.Dgmtx[prev.+(4:5),model.q_binds] .= model.mat24
        # model.Dgmtx[(5*(i-2)).+(1:5),model.q_binds] = [2*model.vmat*RS.rmult(q_b)'*RS.rmult(model.hmat*model.joint_vertices[i-1][4:6]);
        #                                                model.joint_cmat[i-1]*model.vmat*RS.lmult(model.qat)'
        #                                               ]
    end
    return 
end


# this calculates a part of Dg*attiG, only related to G_qa , dim is 5x3
function Gqa(q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},vertices, joint_direction,cmat)  where T
    Dgmtx = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertices[1:3]);
             -cmat*RS.vmat()*RS.lmult(q_b)'
            ]
    return Dgmtx*RS.lmult(q_a)*RS.hmat()
end

# this calculates a part of Dg*attiG, only related to G_qb, dim is 5x3
function Gqb(q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},vertices, joint_direction,cmat)  where T
    Dgmtx = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertices[4:6]);
             cmat*RS.vmat()*RS.lmult(q_a)'
            ]
    return Dgmtx*RS.lmult(q_b)*RS.hmat()
end

# inplace version
function Gqa!(model, mtx, q_a,q_b,vertices,cmat) 
    
    model.vec3[1] = vertices[1]
    model.vec3[2] = vertices[2]
    model.vec3[3] = vertices[3]
    mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][1:3]
    q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
    model.mat44 .= model.rmat
    q_rmult_T!(model, q_a)                                  # RS.rmult(model.qat)'
    mul!(model.mat34, model.vmat, model.rmat)                     # model.vmat*RS.rmult(model.qat)'
    mul!(model.mat342, model.mat34, model.mat44)          # model.vmat*RS.rmult(model.qat)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
    
    model.mat342 .*= -2.0
    model.mat54[1:3,:] .= model.mat342

    q_lmult_T!(model, q_b) 
    mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
    mul!(model.mat24, cmat, model.mat34)
    model.mat24 .*= -1.0 
    model.mat54[4:5,:] .= model.mat24

    q_lmult!(model, q_a)
    mul!(model.mat43, model.lmat, model.hmat)

    mul!(mtx, model.mat54, model.mat43)
    return 
end

# this calculates a part of Dg*attiG, only related to G_qb, dim is 5x3
function Gqb!(model, mtx, q_a, q_b,vertices,cmat) 
    model.vec3[1] = vertices[4]
    model.vec3[2] = vertices[5]
    model.vec3[3] = vertices[6]
    mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][4:6]
    q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][4:6])
    model.mat44 .= model.rmat
    q_rmult_T!(model, q_b)                                  #  RS.rmult(model.qbt)'
    mul!(model.mat34, model.vmat, model.rmat)               #  model.vmat*RS.rmult(model.qbt)'
    mul!(model.mat342, model.mat34, model.mat44)            #  2*model.vmat*RS.rmult(model.qbt)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])

    model.mat342 .*= 2.0
    model.mat54[1:3,:] .= model.mat342

    q_lmult_T!(model, q_a) 
    mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
    mul!(model.mat24, cmat, model.mat34) 
    model.mat54[4:5,:]  .= model.mat24

    
    q_lmult!(model, q_b)
    mul!(model.mat43, model.lmat, model.hmat)
    mul!(mtx, model.mat54, model.mat43)
    return
end


# This is similar to g, but we need to propogate state
function gp1(model::FloatingSpace, x, dt)
    g_val = zeros(eltype(x),model.p)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        v_a = SVector{3}(x[v_ainds])
        r_b = SVector{3}(x[r_binds])
        v_b = SVector{3}(x[v_binds])
        q_a = SVector{4}(x[q_ainds])
        w_a = SVector{3}(x[w_ainds])
        q_b = SVector{4}(x[q_binds])
        w_b = SVector{3}(x[w_binds])
        # propagate states 
        r_a1 = r_a + v_a*dt
        r_b1 = r_b + v_b*dt
    
        q_a1 = dt/2*RS.lmult(q_a)*SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a])
        q_b1 = dt/2*RS.lmult(q_b)*SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b])

        # impose constraint on r_a1, r_b1, q_a1, q_b1
        val = view(g_val, (5*(i-2)).+(1:5))
        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]

        val[1:3] = (r_b1 + RS.vmat()*RS.rmult(q_b1)'*RS.lmult(q_b1)*RS.hmat()*vertex2) - 
                   (r_a1 + RS.vmat()*RS.rmult(q_a1)'*RS.lmult(q_a1)*RS.hmat()*vertex1)
        tmp = RS.vmat()*RS.lmult(q_a1)'*q_b1
        # the joint constraint map, it depends on the joint rotation direction 
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end
        val[4:5] = cmat*tmp  
    end
    return g_val
end

function gp1!(model::FloatingSpace, x, dt)
    n::Int=model.nb+1
    prev::Int = 0
    for i=2:n   # i is the rigidbody index
        prev = 5*(i-2)
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        statea_inds!(model,i-1)
        # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        stateb_inds!(model,i)

        model.rat[1] = x[model.r_ainds[1]]
        model.rat[2] = x[model.r_ainds[2]]
        model.rat[3] = x[model.r_ainds[3]]

        model.rbt[1] = x[model.r_binds[1]]
        model.rbt[2] = x[model.r_binds[2]]
        model.rbt[3] = x[model.r_binds[3]]

        model.qat[1] = x[model.q_ainds[1]]
        model.qat[2] = x[model.q_ainds[2]]
        model.qat[3] = x[model.q_ainds[3]]
        model.qat[4] = x[model.q_ainds[4]]

        model.qbt[1] = x[model.q_binds[1]]
        model.qbt[2] = x[model.q_binds[2]]
        model.qbt[3] = x[model.q_binds[3]]
        model.qbt[4] = x[model.q_binds[4]]

        model.vat[1] = x[model.v_ainds[1]]
        model.vat[2] = x[model.v_ainds[2]]
        model.vat[3] = x[model.v_ainds[3]]

        model.vbt[1] = x[model.v_binds[1]]
        model.vbt[2] = x[model.v_binds[2]]
        model.vbt[3] = x[model.v_binds[3]]

        model.wat[1] = x[model.w_ainds[1]]
        model.wat[2] = x[model.w_ainds[2]]
        model.wat[3] = x[model.w_ainds[3]]

        model.wbt[1] = x[model.w_binds[1]]
        model.wbt[2] = x[model.w_binds[2]]
        model.wbt[3] = x[model.w_binds[3]]


        # propagate states 
        # r_a1 = r_a + v_a*dt

        model.rat1[1] = model.rat[1] + model.vat[1]*dt
        model.rat1[2] = model.rat[2] + model.vat[2]*dt
        model.rat1[3] = model.rat[3] + model.vat[3]*dt

        model.rbt1[1] = model.rbt[1] + model.vbt[1]*dt
        model.rbt1[2] = model.rbt[2] + model.vbt[2]*dt
        model.rbt1[3] = model.rbt[3] + model.vbt[3]*dt
    
        q_lmult!(model, model.qat)
        model.vec4[1] = sqrt(4/dt^2 -model.wat'*model.wat)
        model.vec4[2:4] .= model.wat
        mul!(model.qat1, model.lmat, model.vec4)
        model.qat1 .*= dt/2
        # model.qat1 .= dt/2*model.lmat*[sqrt(4/dt^2 -model.wat'*model.wat);model.wat]
        q_lmult!(model, model.qbt)
        model.vec4[1] = sqrt(4/dt^2 -model.wbt'*model.wbt)
        model.vec4[2:4] .= model.wbt
        mul!(model.qbt1, model.lmat, model.vec4)
        model.qbt1 .*= dt/2
        # model.qbt1 .= dt/2*model.lmat*[sqrt(4/dt^2 -model.wbt'*model.wbt);model.wbt]

        # impose constraint on r_a1, r_b1, q_a1, q_b1
        model.vec3[1] = model.joint_vertices[i-1][4]
        model.vec3[2] = model.joint_vertices[i-1][5]
        model.vec3[3] = model.joint_vertices[i-1][6]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][4:6]
        q_rmult_T!(model, model.qbt1)                                  # RS.rmult(q_b)'
        q_lmult!(model, model.qbt1)                                    # RS.lmult(q_b)
        mul!(model.mat44, model.rmat, model.lmat)                     # RS.rmult(q_a)'*RS.lmult(q_b)
        mul!(model.mat34, model.vmat, model.mat44)                    # model.vmat*RS.rmult(q_a)'*RS.lmult(q_b)
        mul!(model.vec3, model.mat34, model.vec4)                     # model.vmat*RS.rmult(q_b)'*RS.lmult(q_b)*model.hmat*model.joint_vertices[i-1][4:6]

        # model.g_val[(5*(i-2)).+(1:3)] = model.rbt
        model.g_val[prev+1] = model.rbt1[1]
        model.g_val[prev+2] = model.rbt1[2]
        model.g_val[prev+3] = model.rbt1[3]

        model.g_val[prev+1] = model.g_val[prev+1] + model.vec3[1]
        model.g_val[prev+2] = model.g_val[prev+2] + model.vec3[2]
        model.g_val[prev+3] = model.g_val[prev+3] + model.vec3[3]

        model.vec3[1] = model.joint_vertices[i-1][1]
        model.vec3[2] = model.joint_vertices[i-1][2]
        model.vec3[3] = model.joint_vertices[i-1][3]
        mul!(model.vec4, model.hmat, model.vec3) # model.hmat*model.joint_vertices[i-1][1:3]
        q_lmult!(model, model.qat1)                                    # RS.lmult(q_a)
        q_rmult_T!(model, model.qat1)                                  # RS.rmult(q_a)'
        mul!(model.mat44, model.rmat, model.lmat)                     # RS.rmult(q_a)'*RS.lmult(q_a)
        mul!(model.mat34, model.vmat, model.mat44)                    # model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)
        mul!(model.vec3, model.mat34, model.vec4)                     # model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)*model.hmat*model.joint_vertices[i-1][1:3]

        model.g_val[prev+1] = model.g_val[prev+1] - model.rat1[1]
        model.g_val[prev+2] = model.g_val[prev+2] - model.rat1[2]
        model.g_val[prev+3] = model.g_val[prev+3] - model.rat1[3]

        model.g_val[prev+1] = model.g_val[prev+1] - model.vec3[1]
        model.g_val[prev+2] = model.g_val[prev+2] - model.vec3[2]
        model.g_val[prev+3] = model.g_val[prev+3] - model.vec3[3]

        # model.g_val[(5*(i-2)).+(1:3)] = (r_b + model.vmat*RS.rmult(q_b)'*RS.lmult(q_b)*model.hmat*model.joint_vertices[i-1][4:6]) - 
        # (r_a + model.vmat*RS.rmult(q_a)'*RS.lmult(q_a)*model.hmat*model.joint_vertices[i-1][1:3])

        q_lmult_T!(model, model.qat1) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_a)'
        mul!(model.vec3, model.mat34, model.qbt1)                     # model.vmat*RS.lmult(q_a)'*model.qbt
        mul!(model.vec2, model.joint_cmat[i-1], model.vec3)          # model.joint_cmat[i-1]*tmp  
        model.g_val[prev+4] = model.vec2[1]
        model.g_val[prev+5] = model.vec2[2] 
    end
    return
end

# function Dgp1, the jacobian of gp1
# jacobian of gp1, treat quaternion as normal 4 vectors
function Dgp1(model::FloatingSpace, x, dt)
    Dgmtx = spzeros(model.p,model.ns)
    for i=2:model.nb+1   # i is the rigidbody index
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        r_a = SVector{3}(x[r_ainds])
        v_a = SVector{3}(x[v_ainds])
        r_b = SVector{3}(x[r_binds])
        v_b = SVector{3}(x[v_binds])
        q_a = SVector{4}(x[q_ainds])
        w_a = SVector{3}(x[w_ainds])
        q_b = SVector{4}(x[q_binds])
        w_b = SVector{3}(x[w_binds])
        # propagate states 
        r_a1 = r_a + v_a*dt
        r_b1 = r_b + v_b*dt
    
        q_a1 = dt/2*RS.lmult(q_a)*SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a])
        q_b1 = dt/2*RS.lmult(q_b)*SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b])


        vertex1 = model.joint_vertices[i-1][1:3]
        vertex2 = model.joint_vertices[i-1][4:6]
        cmat = [0 0 1; 
                1 0 0]
        if model.joint_directions[i-1] == [0,0,1]
            cmat = [0 1 0; 
                   1 0 0]
        else
            cmat = [0 0 1; 
                    1 0 0]
        end

        Dgblock = view(Dgmtx, (5*(i-2)).+(1:5),:)

        ∂dgp1∂dra1 = [-I;zeros(2,3)]
        ∂dgp1∂drb1 = [ I;zeros(2,3)]
        ∂dgp1∂dqa1 = [-2*RS.vmat()*RS.rmult(q_a1)'*RS.rmult(RS.hmat()*vertex1);
                      -cmat*RS.vmat()*RS.lmult(q_b)'
                    ]
        ∂dgp1∂dqb1 =[2*RS.vmat()*RS.rmult(q_b1)'*RS.rmult(RS.hmat()*vertex2);
                        cmat*RS.vmat()*RS.lmult(q_a)'
                    ]
        ∂dra1∂dva = I(3)*dt
        ∂drb1∂dvb = I(3)*dt   
        ∂dqa1∂dqa = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -w_a'*w_a);w_a]))      
        ∂dqa1∂dwa = dt/2*(-q_a*w_a'/sqrt(4/dt^2 -w_a'*w_a) + RS.lmult(q_a)*RS.hmat())    

        ∂dqb1∂dqb = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b]))      
        ∂dqb1∂dwb = dt/2*(-q_b*w_b'/sqrt(4/dt^2 -w_b'*w_b) + RS.lmult(q_b)*RS.hmat())  

        Dgblock[:,r_ainds] =  ∂dgp1∂dra1 # dg/dra
        Dgblock[:,v_ainds] =  ∂dgp1∂dra1*∂dra1∂dva# dg/dva

        Dgblock[:,r_binds]  = ∂dgp1∂drb1  # dg/drb
        Dgblock[:,v_binds]  =  ∂dgp1∂drb1*∂drb1∂dvb# dg/dvb

        Dgblock[:,q_ainds] = ∂dgp1∂dqa1*∂dqa1∂dqa# dg/dqa
        Dgblock[:,w_ainds] = ∂dgp1∂dqa1*∂dqa1∂dwa# dg/dwa
        Dgblock[:,q_binds] =  ∂dgp1∂dqb1*∂dqb1∂dqb# dg/dqb
        Dgblock[:,w_binds] =  ∂dgp1∂dqb1*∂dqb1∂dwb# dg/dwb
    end
    return Dgmtx
end

function Dgp1!(model::FloatingSpace, x, dt)
    n::Int=model.nb+1
    prev::Int = 0
    for i=2:n   # i is the rigidbody index
        prev = 5*(i-2)
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, i-1) # a is the previous rigid body
        statea_inds!(model,i-1)
        # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, i)   # b is the next rigid body
        stateb_inds!(model,i)

        model.rat[1] = x[model.r_ainds[1]]
        model.rat[2] = x[model.r_ainds[2]]
        model.rat[3] = x[model.r_ainds[3]]

        model.rbt[1] = x[model.r_binds[1]]
        model.rbt[2] = x[model.r_binds[2]]
        model.rbt[3] = x[model.r_binds[3]]

        model.qat[1] = x[model.q_ainds[1]]
        model.qat[2] = x[model.q_ainds[2]]
        model.qat[3] = x[model.q_ainds[3]]
        model.qat[4] = x[model.q_ainds[4]]

        model.qbt[1] = x[model.q_binds[1]]
        model.qbt[2] = x[model.q_binds[2]]
        model.qbt[3] = x[model.q_binds[3]]
        model.qbt[4] = x[model.q_binds[4]]

        model.vat[1] = x[model.v_ainds[1]]
        model.vat[2] = x[model.v_ainds[2]]
        model.vat[3] = x[model.v_ainds[3]]

        model.vbt[1] = x[model.v_binds[1]]
        model.vbt[2] = x[model.v_binds[2]]
        model.vbt[3] = x[model.v_binds[3]]

        model.wat[1] = x[model.w_ainds[1]]
        model.wat[2] = x[model.w_ainds[2]]
        model.wat[3] = x[model.w_ainds[3]]

        model.wbt[1] = x[model.w_binds[1]]
        model.wbt[2] = x[model.w_binds[2]]
        model.wbt[3] = x[model.w_binds[3]]


        # propagate states 
        # r_a1 = r_a + v_a*dt

        model.rat1[1] = model.rat[1] + model.vat[1]*dt
        model.rat1[2] = model.rat[2] + model.vat[2]*dt
        model.rat1[3] = model.rat[3] + model.vat[3]*dt

        model.rbt1[1] = model.rbt[1] + model.vbt[1]*dt
        model.rbt1[2] = model.rbt[2] + model.vbt[2]*dt
        model.rbt1[3] = model.rbt[3] + model.vbt[3]*dt
    
        q_lmult!(model, model.qat)
        model.vec4[1] = sqrt(4/dt^2 -model.wat'*model.wat)
        model.vec4[2:4] .= model.wat
        mul!(model.qat1, model.lmat, model.vec4)
        model.qat1 .*= dt/2
        # model.qat1 .= dt/2*model.lmat*[sqrt(4/dt^2 -model.wat'*model.wat);model.wat]
        q_lmult!(model, model.qbt)
        model.vec4[1] = sqrt(4/dt^2 -model.wbt'*model.wbt)
        model.vec4[2:4] .= model.wbt
        mul!(model.qbt1, model.lmat, model.vec4)
        model.qbt1 .*= dt/2
        # model.qbt1 .= dt/2*model.lmat*[sqrt(4/dt^2 -model.wbt'*model.wbt);model.wbt]

        # ∂dgp1∂dra1 = [-I;zeros(2,3)]
        # ∂dgp1∂drb1 = [ I;zeros(2,3)]

        # ∂dra1∂dva = I(3)*dt
        # ∂drb1∂dvb = I(3)*dt   


        model.Dgmtx[prev+1,model.r_ainds[1]] = -1  # dg/dra
        model.Dgmtx[prev+2,model.r_ainds[2]] = -1  # dg/dra
        model.Dgmtx[prev+3,model.r_ainds[3]] = -1  # dg/dra
        model.Dgmtx[prev+1,model.v_ainds[1]] = -dt  # dg/dva
        model.Dgmtx[prev+2,model.v_ainds[2]] = -dt  # dg/dva
        model.Dgmtx[prev+3,model.v_ainds[3]] = -dt  # dg/dva

        model.Dgmtx[prev+1,model.r_binds[1]] = 1  # dg/dra
        model.Dgmtx[prev+2,model.r_binds[2]] = 1  # dg/dra
        model.Dgmtx[prev+3,model.r_binds[3]] = 1  # dg/dra
        model.Dgmtx[prev+1,model.v_binds[1]] = dt  # dg/dva
        model.Dgmtx[prev+2,model.v_binds[2]] = dt  # dg/dva
        model.Dgmtx[prev+3,model.v_binds[3]] = dt  # dg/dva

        # ∂dgp1∂dqa1 = [-2*model.vmat*RS.rmult(q_a1)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3]);
        #               -model.joint_cmat[i-1]*model.vmat*RS.lmult(q_b)'
        #             ]
        model.vec3[1] = model.joint_vertices[i-1][1]
        model.vec3[2] = model.joint_vertices[i-1][2]
        model.vec3[3] = model.joint_vertices[i-1][3]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][1:3]
        q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
        model.mat44 .= model.rmat
        q_rmult_T!(model, model.qat1)                                  # RS.rmult(model.qat1)'
        mul!(model.mat34, model.vmat, model.rmat)                     # model.vmat*RS.rmult(model.qat)'
        mul!(model.mat342, model.mat34, model.mat44)          # model.vmat*RS.rmult(model.qat)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
        
        model.mat342 .*= -2.0
        model.mat54[1:3,:] = model.mat342
    
        q_lmult_T!(model, model.qbt) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
        mul!(model.mat24, model.joint_cmat[i-1], model.mat34) 
        model.mat24 .*= -1.0
        model.mat54[4:5,:] .= model.mat24
        # ∂dqa1∂dqa = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat]))  
        model.vec4[1] = sqrt(4/dt^2 -model.wat'*model.wat)
        model.vec4[2:4] .= model.wat
        q_rmult!(model, model.vec4)
        model.mat44 .= model.rmat
        model.mat44 .*= dt/2
        mul!(model.mat542, model.mat54, model.mat44) 
        model.Dgmtx[prev.+(1:5),model.q_ainds] .= model.mat542# dg/dqa1* ∂dqa1∂dqa
        
        # ∂dqa1∂dwa = dt/2*(-model.qat*model.wat'/sqrt(4/dt^2 -model.wat'*model.wat) + model.lmat*model.hmat)  
        q_lmult!(model, model.qat)
        mul!(model.mat43, model.lmat, model.hmat)
        mul!(model.mat432, model.qat, Transpose(model.wat))
        model.mat432 ./= sqrt(4/dt^2 -model.wat'*model.wat)
        model.mat43 .-= model.mat432
        model.mat43 .*= dt/2
        mul!(model.mat53, model.mat54, model.mat43) 
        model.Dgmtx[prev.+(1:5),model.w_ainds] .= model.mat53 # dg/dqa1* ∂dqa1∂dwa


        # ∂dgp1∂dqb1 =[2*model.vmat*RS.rmult(q_b1)'*RS.rmult(model.hmat*model.joint_vertices[i-1][4:6]);
        #                 model.joint_cmat[i-1]*model.vmat*RS.lmult(q_a)'
        #             ]
        model.vec3[1] = model.joint_vertices[i-1][4]
        model.vec3[2] = model.joint_vertices[i-1][5]
        model.vec3[3] = model.joint_vertices[i-1][6]
        mul!(model.vec4, model.hmat, model.vec3)  # model.hmat*model.joint_vertices[i-1][4:6]
        q_rmult!(model, model.vec4)                                   # RS.rmult(model.hmat*model.joint_vertices[i-1][4:6])
        model.mat44 .= model.rmat
        q_rmult_T!(model, model.qbt1)                                  # RS.rmult(model.qbt)'
        mul!(model.mat34, model.vmat, model.rmat)                     # model.vmat*RS.rmult(model.qbt)'
        mul!(model.mat342, model.mat34, model.mat44)          #  2*model.vmat*RS.rmult(model.qbt)'*RS.rmult(model.hmat*model.joint_vertices[i-1][1:3])
    
        model.mat342 .*= 2.0
        model.mat54[1:3,:] = model.mat342
        q_lmult_T!(model, model.qat) 
        mul!(model.mat34, model.vmat, model.lmat)                    # model.vmat*RS.lmult(q_b)'
        mul!(model.mat24, model.joint_cmat[i-1], model.mat34) 
        model.mat54[4:5,:] .= model.mat24


        # ∂dqb1∂dqb = dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -w_b'*w_b);w_b]))      
        model.vec4[1] = sqrt(4/dt^2 -model.wbt'*model.wbt)
        model.vec4[2:4] .= model.wbt
        q_rmult!(model, model.vec4)
        model.mat44 .= model.rmat
        model.mat44 .*= dt/2
        mul!(model.mat542, model.mat54, model.mat44) 
        model.Dgmtx[prev.+(1:5),model.q_binds] .= model.mat542# dg/dqb1* ∂dqb1∂dqb
        # ∂dqb1∂dwb = dt/2*(-q_b*w_b'/sqrt(4/dt^2 -w_b'*w_b) + RS.lmult(q_b)*model.hmat)  
        q_lmult!(model, model.qbt)
        mul!(model.mat43, model.lmat, model.hmat)
        mul!(model.mat432, model.qbt, Transpose(model.wbt))
        model.mat432 ./= -model.vec4[1]
        model.mat43  .+= model.mat432
        model.mat43 .*= dt/2
        mul!(model.mat53, model.mat54, model.mat43) 
        model.Dgmtx[prev.+(1:5),model.w_binds] .= model.mat53 # dg/dqb1* ∂dqb1∂dwb
    end
    return
end

function state_diff_jac(model::FloatingSpace,x::AbstractArray{T}) where T
    n,m = size(model)
    n̄ = state_diff_size(model)

    G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
    RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, Lie_P(model)) , SVector{n}(x))

    return G
end

function state_diff_jac!(model::FloatingSpace,x::AbstractArray{T}) where T

    RD.state_diff_jacobian!(model.attiG, model.liestate , SVector{model.ns}(x))
    
    return
end


# implicity dynamics function fdyn
# return f(x_t1, x_t, u_t, λt) = 0
# TODO: 5 and 13 are sort of magic number that should be put in constraint
# TODO: what if all masses of links are different
function fdyn(model::FloatingSpace,xt1, xt, ut, λt, dt)
    fdyn_vec = zeros(eltype(xt1),model.ns)
    u_joint = ut[7:end]
    for link_id=1:model.nb+1
        fdyn_vec_block = view(fdyn_vec, (13*(link_id-1)).+(1:13))
        joint_before_id = link_id-1
        joint_after_id  = link_id
        # iterate through all rigid bodies
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, link_id) # a is the current link

        # get state from xt1
        rat1 = xt1[r_ainds]
        vat1 = xt1[v_ainds]
        qat1 = SVector{4}(xt1[q_ainds])
        wat1 = xt1[w_ainds]
        # get state from xt
        rat = xt[r_ainds]
        vat = xt[v_ainds]
        qat = SVector{4}(xt[q_ainds])
        wat = xt[w_ainds]

        # link_id==1 (the body) need special attention 
        # link_id==nb+1 (the last arm link)
        if (link_id == 1)  #the body link
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])

            # only the body link use these forces and torques
            Ft = ut[1:3]
            taut = ut[4:6]
            tau_joint = u_joint[joint_after_id]
            λt_block = λt[(5*(link_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)

            # velocity
            # Ma = diagm([model.body_mass,model.body_mass,model.body_mass])
            aa = model.body_mass_mtx*(vat1-vat) + model.body_mass_mtx*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa - Ft*dt - [-I(3);zeros(2,3)]'*λt_block*dt   # Gra'λ

            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])

            # angular velocity
            vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            Gqamtx = Gqa(qat1,qbt1,vertices, joint_direction,model.joint_cmat[joint_after_id])  
            Ja = model.body_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k = - 2*taut + 2*tau_joint*joint_direction 
            fdyn_vec_block[11:13] = a+k - Gqamtx'*λt_block

        elseif (link_id >= 2 && link_id < model.nb+1) # normal arm link
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])

            next_tau_joint = u_joint[joint_after_id]   # next == after
            prev_tau_joint = u_joint[joint_before_id]  # perv == before

            next_λt_block = λt[(5*(link_id-1)).+(1:5)]
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]

            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)
            # velocity 
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            aa = model.arm_mass_mtx*(vat1-vat) + model.arm_mass_mtx*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa -[-I(3);zeros(2,3)]'*next_λt_block*dt -[I(3);zeros(2,3)]'*prev_λt_block*dt
            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])
            # angular velocity (need to add previous joint constraint)
            # joint between a and b # use Gra
            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id])  
            # joint between z and a  # use Grb
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])

            Gqzmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction,model.joint_cmat[joint_before_id])  

            Ja = model.arm_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k =  - 2*prev_tau_joint*prev_joint_direction + 2*next_tau_joint*next_joint_direction 
            fdyn_vec_block[11:13] = a+k - Gqamtx'*next_λt_block - Gqzmtx'*prev_λt_block

        else # the last link 
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])
            prev_tau_joint = u_joint[joint_before_id]  # perv == before
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] = rat1 - (rat + vat*dt)
            # velocity (only different from link_id == 1 is no force, and different mass)
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            aa = model.arm_mass_mtx*(vat1-vat) + model.arm_mass_mtx*[0;0;model.g]*dt
            fdyn_vec_block[4:6] =  aa -  [I(3);zeros(2,3)]'*prev_λt_block*dt
            # orientation
            fdyn_vec_block[7:10] = qat1 - dt/2*RS.lmult(qat)*SVector{4}([sqrt(4/dt^2 -wat'*wat);wat])
            # angular velocity (need to add previous joint constraint)
            # joint between z and a 
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])
            Gqzmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction,model.joint_cmat[joint_before_id]) 

            Ja = model.arm_inertias
            a = Ja * wat1 * sqrt(4/dt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/dt^2 - wat'*wat) + cross(wat,(Ja * wat))
            k =  - 2*prev_tau_joint*prev_joint_direction

            fdyn_vec_block[11:13] = a+k- Gqzmtx'*prev_λt_block

        end
    end
    return fdyn_vec
end

# implicity dynamics function fdyn!
# calculate f(x_t1, x_t, u_t, λt), save its result in model.fdyn_vec
# TODO: 5 and 13 are sort of magic number that should be put in constraint
# TODO: what if all masses of links are different
function fdyn!(model::FloatingSpace,xt1, xt, ut, λt, dt)
    # fdyn_vec = zeros(eltype(xt1),model.ns)        
    # only the body link use these forces and torques
    Ft = ut[1:3]
    taut = ut[4:6]
    u_joint = ut[7:end]
    Gqamtx = zeros(5,3)  # common storage
    Gqzmtx = zeros(5,3)  # common storage
    joint_before_id = 0
    joint_after_id = 0    
    λt_block = zeros(5)
    next_λt_block = zeros(5)
    prev_λt_block = zeros(5)
    for link_id=1:model.nb+1
        begin
        fdyn_vec_block = view(model.fdyn_vec, (13*(link_id-1)).+(1:13))
        joint_before_id = link_id-1
        joint_after_id  = link_id
        # iterate through all rigid bodies
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, link_id) # a is the current link
        statea_inds!(model,link_id)

        # get state from xt1
        model.rat1[1] = xt1[model.r_ainds[1]]
        model.rat1[2] = xt1[model.r_ainds[2]]
        model.rat1[3] = xt1[model.r_ainds[3]]

        model.vat1[1] = xt1[model.v_ainds[1]]
        model.vat1[2] = xt1[model.v_ainds[2]]
        model.vat1[3] = xt1[model.v_ainds[3]]

        model.qat1[1] = xt1[model.q_ainds[1]]
        model.qat1[2] = xt1[model.q_ainds[2]]
        model.qat1[3] = xt1[model.q_ainds[3]]
        model.qat1[4] = xt1[model.q_ainds[4]]
        
        model.wat1[1] = xt1[model.w_ainds[1]]
        model.wat1[2] = xt1[model.w_ainds[2]]
        model.wat1[3] = xt1[model.w_ainds[3]]
        # get state from xt
        model.rat[1] = xt[model.r_ainds[1]]
        model.rat[2] = xt[model.r_ainds[2]]
        model.rat[3] = xt[model.r_ainds[3]]

        model.vat[1] = xt[model.v_ainds[1]]
        model.vat[2] = xt[model.v_ainds[2]]
        model.vat[3] = xt[model.v_ainds[3]]

        model.qat[1] = xt[model.q_ainds[1]]
        model.qat[2] = xt[model.q_ainds[2]]
        model.qat[3] = xt[model.q_ainds[3]]
        model.qat[4] = xt[model.q_ainds[4]]
        
        model.wat[1] = xt[model.w_ainds[1]]
        model.wat[2] = xt[model.w_ainds[2]]
        model.wat[3] = xt[model.w_ainds[3]]

        # link_id==1 (the body) need special attention 
        # link_id==nb+1 (the last arm link)
        if (link_id == 1)  #the body link
            # get next link state from xt1
            # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            stateb_inds!(model,link_id+1)
            model.rbt1[1] = xt1[model.r_binds[1]]
            model.rbt1[2] = xt1[model.r_binds[2]]
            model.rbt1[3] = xt1[model.r_binds[3]]
            
            model.qbt1[1] = xt1[model.q_binds[1]]
            model.qbt1[2] = xt1[model.q_binds[2]]
            model.qbt1[3] = xt1[model.q_binds[3]]
            model.qbt1[4] = xt1[model.q_binds[4]]


            # tau_joint = u_joint[joint_after_id]
            λt_block .= λt[(5*(link_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] .= model.rat1 - (model.rat + model.vat*dt)

            # velocity
            # Ma = diagm([model.body_mass,model.body_mass,model.body_mass])
            aa = model.body_mass_mtx*(model.vat1-model.vat) + model.body_mass_mtx*[0;0;model.g]*dt
            fdyn_vec_block[4:6] .=  aa - Ft*dt + λt[(5*(link_id-1)).+(1:3)]*dt   # Gra'λ

            # orientation
            fdyn_vec_block[7:10] .= model.qat1 - dt/2*RS.lmult(SVector{4}(model.qat))*SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat])

            # angular velocity
            # Gqamtx = Gqa(qat1,qbt1,model.joint_vertices[joint_after_id], model.joint_directions[joint_after_id],model.joint_cmat[joint_after_id])
            Gqa!(model, model.Gqamtx, model.qat1, model.qbt1, model.joint_vertices[joint_after_id], model.joint_cmat[joint_after_id])  
            Ja = model.body_inertias
            a = Ja * model.wat1 * sqrt(4/dt^2 -model.wat1'*model.wat1) + cross(model.wat1, (Ja * model.wat1)) - Ja * model.wat  * sqrt(4/dt^2 - model.wat'*model.wat) + cross(model.wat,(Ja * model.wat))
            k = - 2*taut + 2*u_joint[joint_after_id]*model.joint_directions[joint_after_id] 
            fdyn_vec_block[11:13] .= a+k - model.Gqamtx'*λt_block

        elseif (link_id >= 2 && link_id < model.nb+1) # normal arm link
            # get next link state from xt1
            stateb_inds!(model,link_id+1)
            model.rbt1[1] = xt1[model.r_binds[1]]
            model.rbt1[2] = xt1[model.r_binds[2]]
            model.rbt1[3] = xt1[model.r_binds[3]]
            
            model.qbt1[1] = xt1[model.q_binds[1]]
            model.qbt1[2] = xt1[model.q_binds[2]]
            model.qbt1[3] = xt1[model.q_binds[3]]
            model.qbt1[4] = xt1[model.q_binds[4]]
            # get previous link state from xt1
            statez_inds!(model,link_id-1)
            model.rzt1[1] = xt1[model.r_zinds[1]]
            model.rzt1[2] = xt1[model.r_zinds[2]]
            model.rzt1[3] = xt1[model.r_zinds[3]]
            
            model.qzt1[1] = xt1[model.q_zinds[1]]
            model.qzt1[2] = xt1[model.q_zinds[2]]
            model.qzt1[3] = xt1[model.q_zinds[3]]
            model.qzt1[4] = xt1[model.q_zinds[4]]

            # next_tau_joint = u_joint[joint_after_id]   # next == after
            # prev_tau_joint = u_joint[joint_before_id]  # perv == before

            next_λt_block .= λt[(5*(link_id-1)).+(1:5)]
            prev_λt_block .= λt[(5*(joint_before_id-1)).+(1:5)]

            # position
            fdyn_vec_block[1:3] .= model.rat1 - (model.rat + model.vat*dt)
            # velocity 
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            Ma = SMatrix{3,3}(model.arm_mass_mtx)
            v = SVector{3}(model.vat)
            v1 = SVector{3}(model.vat1)
            model.vec3 .= 0
            model.vec3[3] = model.g*dt
            model.vec3 .+= model.vat1
            model.vec3 .-= model.vat
            # aa = Ma*model.vec3
            fdyn_vec_block[4:6] .=  Ma*SVector{3}(model.vec3) +λt[(5*(link_id-1)).+(1:3)]*dt -λt[(5*(joint_before_id-1)).+(1:3)]*dt
            # orientation
            w = SVector{3}(model.wat)
            q_lmult!(model, model.qat)
            fdyn_vec_block[7:10] .= SVector{4}(model.qat1) - dt/2*SMatrix{4,4}(model.lmat)*SVector{4}([sqrt(4/dt^2 -w'*w);w])
            # angular velocity (need to add previous joint constraint)
            # joint between a and b # use Gra
            # next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            # next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            # Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id])  
            Gqa!(model, model.Gqamtx, model.qat1, model.qbt1, model.joint_vertices[joint_after_id], model.joint_cmat[joint_after_id])  
            # joint between z and a  # use Grb
            # prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            # prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])

            # Gqb!(model,Gqzmtx,qzt1,qat1,prev_vertices,model.joint_cmat[joint_before_id])  
            Gqb!(model, model.Gqzmtx, model.qzt1, model.qat1, model.joint_vertices[joint_before_id], model.joint_cmat[joint_before_id])  

            # 2021-04-27 I learned this does not allocation additional memory, and it unroll the calculate 
            Ja = SMatrix{3,3}(model.arm_inertias)
            w1 = SVector{3}(model.wat1)
            model.vec3 .= Ja * w1 * sqrt(4/dt^2 -w1'*w1) + cross(w1, (Ja * w1)) - Ja * w  * sqrt(4/dt^2 - w'*w) + cross(w,(Ja * w))
            k =  - 2*u_joint[joint_before_id]*model.joint_directions[joint_before_id] + 2*u_joint[joint_after_id]*model.joint_directions[joint_after_id] 
            fdyn_vec_block[11:13] .= model.vec3+k 
            mul!(model.vec3, Transpose(model.Gqamtx),next_λt_block)
            fdyn_vec_block[11:13] .-=model.vec3
            mul!(model.vec3, Transpose(model.Gqzmtx),prev_λt_block)
            fdyn_vec_block[11:13] .-=model.vec3
            

        else # the last link 
            # get previous link state from xt1
            # r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            statez_inds!(model,link_id-1)
            model.rzt1[1] = xt1[model.r_zinds[1]]
            model.rzt1[2] = xt1[model.r_zinds[2]]
            model.rzt1[3] = xt1[model.r_zinds[3]]
            
            model.qzt1[1] = xt1[model.q_zinds[1]]
            model.qzt1[2] = xt1[model.q_zinds[2]]
            model.qzt1[3] = xt1[model.q_zinds[3]]
            model.qzt1[4] = xt1[model.q_zinds[4]]

            prev_tau_joint = u_joint[joint_before_id]  # perv == before
            prev_λt_block .= λt[(5*(joint_before_id-1)).+(1:5)]
            # position
            fdyn_vec_block[1:3] .= model.rat1 - (model.rat + model.vat*dt)
            # velocity (only different from link_id == 1 is no force, and different mass)
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            aa = model.arm_mass_mtx*(model.vat1-model.vat) + model.arm_mass_mtx*[0;0;model.g]*dt
            fdyn_vec_block[4:6] .=  aa -λt[(5*(joint_before_id-1)).+(1:3)]*dt
            # orientation
            fdyn_vec_block[7:10] .= model.qat1 - dt/2*RS.lmult(SVector{4}(model.qat))*SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat])
            # angular velocity (need to add previous joint constraint)
            # joint between z and a 
            # prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            # prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])
            Gqb!(model, model.Gqzmtx, model.qzt1, model.qat1, model.joint_vertices[joint_before_id], model.joint_cmat[joint_before_id])

            Ja = model.arm_inertias
            a = Ja * model.wat1 * sqrt(4/dt^2 -model.wat1'*model.wat1) + cross(model.wat1, (Ja * model.wat1)) - Ja * model.wat  * sqrt(4/dt^2 - model.wat'*model.wat) + cross(model.wat,(Ja * model.wat))
            k =  - 2*prev_tau_joint*model.joint_directions[joint_before_id]

            fdyn_vec_block[11:13] .= a+k- model.Gqzmtx'*prev_λt_block

        end

        end
    end
    return 
end

# helper functions for calculation jacobian
# following functions are from Jan's ConstrainedDynamics/src/util/quaternion.jl
function ∂L∂q() 
    return SA{Float64}[
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1

        0 -1 0 0
        1 0 0 0
        0 0 0 1
        0 0 -1 0

        0 0 -1 0
        0 0 0 -1
        1 0 0 0
        0 1 0 0
        
        0 0 0 -1
        0 0 1 0
        0 -1 0 0
        1 0 0 0
    ]
end
function ∂Lᵀ∂q() 
    return SA{Float64}[
        1 0 0 0
        0 -1 0 0
        0 0 -1 0
        0 0 0 -1
        
        0 1 0 0
        1 0 0 0
        0 0 0 -1
        0 0 1 0
        
        0 0 1 0
        0 0 0 1
        1 0 0 0
        0 -1 0 0
        
        0 0 0 1
        0 0 -1 0
        0 1 0 0
        1 0 0 0
    ]
end
function ∂R∂q() 
    return SA{Float64}[
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        
        0 -1 0 0
        1 0 0 0
        0 0 0 -1
        0 0 1 0
        
        0 0 -1 0
        0 0 0 1
        1 0 0 0
        0 -1 0 0
        
        0 0 0 -1
        0 0 -1 0
        0 1 0 0
        1 0 0 0
    ]
end
function ∂Rᵀ∂q() 
    return SA{Float64}[
        1 0 0 0
        0 -1 0 0
        0 0 -1 0
        0 0 0 -1
        
        0 1 0 0
        1 0 0 0
        0 0 0 1
        0 0 -1 0
        
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 1 0 0
        
        0 0 0 1
        0 0 1 0
        0 -1 0 0
        1 0 0 0
    ]
end

# this function is used in the inplace version of jacobians, it treats x as a 1x3 vector and do kron(x, I(4))
# the result is saved in model.kron4_mtx
function kron3!(model::FloatingSpace, x::AbstractVector{T}) where T
    model.kron3_mtx .= 0

    model.kron3_mtx[1,1] = x[1]
    model.kron3_mtx[2,2] = x[1]
    model.kron3_mtx[3,3] = x[1]
    model.kron3_mtx[4,4] = x[1]

    model.kron3_mtx[1,5] = x[2]
    model.kron3_mtx[2,6] = x[2]
    model.kron3_mtx[3,7] = x[2]
    model.kron3_mtx[4,8] = x[2]

    model.kron3_mtx[1,9] =  x[3]
    model.kron3_mtx[2,10] = x[3]
    model.kron3_mtx[3,11] = x[3]
    model.kron3_mtx[4,12] = x[3]

end
# this function is used in the inplace version of jacobians, it treats x as a 1x4 vector and do kron(x, I(4))
# the result is saved in model.kron4_mtx
function kron4!(model::FloatingSpace, x::AbstractVector{T}) where T
    model.kron4_mtx .= 0

    model.kron4_mtx[1,1] = x[1]
    model.kron4_mtx[2,2] = x[1]
    model.kron4_mtx[3,3] = x[1]
    model.kron4_mtx[4,4] = x[1]

    model.kron4_mtx[1,5] = x[2]
    model.kron4_mtx[2,6] = x[2]
    model.kron4_mtx[3,7] = x[2]
    model.kron4_mtx[4,8] = x[2]

    model.kron4_mtx[1,9] =  x[3]
    model.kron4_mtx[2,10] = x[3]
    model.kron4_mtx[3,11] = x[3]
    model.kron4_mtx[4,12] = x[3]

    model.kron4_mtx[1,13] = x[4]
    model.kron4_mtx[2,14] = x[4]
    model.kron4_mtx[3,15] = x[4]
    model.kron4_mtx[4,16] = x[4]
end

# the jacobian of Gqaᵀλ wrt to qa
function ∂Gqaᵀλ∂qa(model::FloatingSpace, q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},λt,vertices, joint_direction,cmat)    where T 
    vertex1 = SVector{4}(float([0.0;vertices[1:3]]))                      # pa

    a = -2*model.vmat*kron((RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()
    b = -2*model.vmat*RS.lmult(q_a)'*RS.rmult(vertex1)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    c = -model.vmat*kron((RS.lmult(q_b)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    return a+b+c
end

# (inplace version) the jacobian of Gqaᵀλ wrt to qa
function ∂Gqaᵀλ∂qa!(model::FloatingSpace, mtx, q_a::AbstractVector{T}, q_b::AbstractVector{T}, λt::AbstractVector{T}, vertices,cmat)    where T 
    # vertex1 = SVector{4}([0.0;vertices[1:3]])                      # pa
    # a = -2*model.vmat*kron((RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()
    model.vec4[1] = 0.0
    model.vec4[2] = vertices[1]
    model.vec4[3] = vertices[2]
    model.vec4[4] = vertices[3]
    q_rmult_T!(model, model.vec4)     # RS.rmult(vertex1)'
    model.mat44 .= model.rmat
    q_rmult!(model, q_a)                        # RS.rmult(q_a)
    mul!(model.mat43, model.rmat, model.hmat)   # RS.rmult(q_a)*model.hmat
    mul!(model.mat432, model.mat44, model.mat43)   # RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat
    
    model.vec3[1] = λt[1]
    model.vec3[2] = λt[2]
    model.vec3[3] = λt[3]
    mul!(model.vec4, model.mat432, model.vec3)   # RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat*λt[1:3]
    kron4!(model, model.vec4)                       # kron((RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat*λt[1:3])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dlTdq)  # kron((RS.rmult(vertex1)'*RS.rmult(q_a)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()

    mul!(model.mat34, model.vmat, model.mat44)
    mtx .= model.mat34
    mtx .*= -2.0

    # b = -2*model.vmat*RS.lmult(q_a)'*RS.rmult(vertex1)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    # model.rmat still keeps RS.rmult(vertex1)'
    model.vec3[1] = λt[1]
    model.vec3[2] = λt[2]
    model.vec3[3] = λt[3]
    mul!(model.vec4, model.hmat, model.vec3) # model.hmat*λt[1:3]
    kron4!(model, model.vec4)                       # kron((model.hmat*λt[1:3])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.drdq)  # kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    q_lmult_T!(model, q_a)                          # RS.lmult(q_a)'
    model.vec4[1] = 0.0
    model.vec4[2] = vertices[1]
    model.vec4[3] = vertices[2]
    model.vec4[4] = vertices[3]
    q_rmult_T!(model, model.vec4)     # RS.rmult(vertex1)'
    mul!(model.mat442, model.rmat, model.mat44)     # RS.rmult(vertex1)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    mul!(model.mat44, model.lmat, model.mat442)     # RS.lmult(q_a)'*RS.rmult(vertex1)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    mul!(model.mat34, model.vmat, model.mat44)
    model.mat34 .*= -2.0
    mtx .+= model.mat34

    #c = -model.vmat*kron((RS.lmult(q_b)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    q_lmult!(model, q_b)                          # RS.lmult(q_b)
    mul!(model.mat43, model.lmat, model.hmat)     # RS.lmult(q_b)*model.hmat
    
    model.vec2[1] = λt[4]
    model.vec2[2] = λt[5]
    mul!(model.vec3, Transpose(cmat), model.vec2)     # cmat'*λt[4:5]
    mul!(model.vec4, model.mat43, model.vec3)     # RS.lmult(q_b)*model.hmat*cmat'*λt[4:5]
    kron4!(model, model.vec4)                       # kron((RS.lmult(q_b)*model.hmat*cmat'*λt[4:5])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dlTdq)  #kron((RS.lmult(q_b)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    mul!(model.mat34, model.vmat, model.mat44)       #model.vmat*kron((RS.lmult(q_b)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    model.mat34 .*= -1.0
    mtx .+= model.mat34
    return 
end
# the jacobian of Gqaᵀλ wrt to qb
function ∂Gqaᵀλ∂qb(model::FloatingSpace, q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},λt,vertices, joint_direction, cmat)    where T
  
    return -model.vmat*RS.lmult(q_a)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
end
# (inplace version) the jacobian of Gqaᵀλ wrt to qb
function ∂Gqaᵀλ∂qb!(model::FloatingSpace, mtx, q_a::AbstractVector{T}, q_b::AbstractVector{T}, λt::AbstractVector{T}, vertices,cmat) where T
    # -model.vmat*RS.lmult(q_a)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    
    model.vec2[1] = λt[4]
    model.vec2[2] = λt[5]
    mul!(model.vec3, Transpose(cmat), model.vec2)        # cmat'*λt[4:5]
    mul!(model.vec4, model.hmat, model.vec3)         # model.hmat*cmat'*λt[4:5]
    kron4!(model, model.vec4)                        # kron((model.hmat*cmat'*λt[4:5])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dldq)   # kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    q_lmult_T!(model, q_a)                           # RS.lmult(q_a)'
    mul!(model.mat442, model.lmat, model.mat44)      # RS.lmult(q_a)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    mul!(model.mat34, model.vmat, model.mat442)      # RS.lmult(q_a)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    model.mat34 .*= -1.0
    mtx .= model.mat34
    return 
end

# the jacobian of Gqbᵀλ wrt to qa
function ∂Gqbᵀλ∂qa(model::FloatingSpace, q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},λt,vertices, joint_direction, cmat)    where T

    return model.vmat*RS.lmult(q_b)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
end
# (inplace version) the jacobian of Gqbᵀλ wrt to qa
function ∂Gqbᵀλ∂qa!(model::FloatingSpace, mtx, q_a::AbstractVector{T}, q_b::AbstractVector{T}, λt::AbstractVector{T}, vertices,cmat) where T
    # model.vmat*RS.lmult(q_b)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    
    model.vec2[1] = λt[4]
    model.vec2[2] = λt[5]
    mul!(model.vec3, Transpose(cmat), model.vec2)        # cmat'*λt[4:5]
    mul!(model.vec4, model.hmat, model.vec3)         # model.hmat*cmat'*λt[4:5]
    kron4!(model, model.vec4)                        # kron((model.hmat*cmat'*λt[4:5])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dldq)   # kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    q_lmult_T!(model, q_b)                           # RS.lmult(q_b)'
    mul!(model.mat442, model.lmat, model.mat44)      # RS.lmult(q_b)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    mul!(model.mat34, model.vmat, model.mat442)      # RS.lmult(q_b)'*kron((model.hmat*cmat'*λt[4:5])',I(4))*∂L∂q()
    mtx .= model.mat34
    return 
end

# the jacobian of Gqbᵀλ wrt to qb
function ∂Gqbᵀλ∂qb(model::FloatingSpace, q_a::SArray{Tuple{4},T,1,4},q_b::SArray{Tuple{4},T,1,4},λt,vertices, joint_direction,cmat)   where T
    vertex2 = SVector{4}(float([0.0;vertices[4:6]]))                      # pb

    a = 2*model.vmat*kron((RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()
    b = 2*model.vmat*RS.lmult(q_b)'*RS.rmult(vertex2)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    c = model.vmat*kron((RS.lmult(q_a)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    return a+b+c 
end

# (inplace version) the jacobian of Gqbᵀλ wrt to qb
function ∂Gqbᵀλ∂qb!(model::FloatingSpace, mtx, q_a::AbstractVector{T}, q_b::AbstractVector{T}, λt::AbstractVector{T}, vertices,cmat) where T
    # a = 2*model.vmat*kron((RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()
    model.vec4[1] = 0.0
    model.vec4[2] = vertices[4]
    model.vec4[3] = vertices[5]
    model.vec4[4] = vertices[6]
    q_rmult_T!(model, model.vec4)     # RS.rmult(vertex2)'
    model.mat44 .= model.rmat
    q_rmult!(model, q_b)                        # RS.rmult(q_b)
    mul!(model.mat43, model.rmat, model.hmat)   # RS.rmult(q_b)*model.hmat
    mul!(model.mat432, model.mat44, model.mat43)   # RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat
     
    model.vec3[1] = λt[1]
    model.vec3[2] = λt[2]
    model.vec3[3] = λt[3]
    mul!(model.vec4, model.mat432, model.vec3)   # RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat*λt[1:3]
    kron4!(model, model.vec4)                       # kron((RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat*λt[1:3])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dlTdq)  # kron((RS.rmult(vertex2)'*RS.rmult(q_b)*model.hmat*λt[1:3])',I(4))*∂Lᵀ∂q()

    mul!(model.mat34, model.vmat, model.mat44)
    mtx .= model.mat34
    mtx .*= 2.0

    # b = 2*model.vmat*RS.lmult(q_b)'*RS.rmult(vertex2)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    
    model.vec3[1] = λt[1]
    model.vec3[2] = λt[2]
    model.vec3[3] = λt[3]
    mul!(model.vec4, model.hmat, model.vec3)           # model.hmat*λt[1:3]
    kron4!(model, model.vec4)                       # kron((model.hmat*λt[1:3])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.drdq)  # kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    q_lmult_T!(model, q_b)                          # RS.lmult(q_b)'
    model.vec4[1] = 0.0
    model.vec4[2] = vertices[4]
    model.vec4[3] = vertices[5]
    model.vec4[4] = vertices[6]
    q_rmult_T!(model, model.vec4)                   # RS.rmult(vertex2)'
    mul!(model.mat442, model.rmat, model.mat44)     # RS.rmult(vertex2)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    mul!(model.mat44, model.lmat, model.mat442)     # RS.lmult(q_b)'*RS.rmult(vertex2)'*kron((model.hmat*λt[1:3])',I(4))*∂R∂q()
    mul!(model.mat34, model.vmat, model.mat44)
    model.mat34 .*= 2.0
    mtx .+= model.mat34

    #c = model.vmat*kron((RS.lmult(q_a)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    q_lmult!(model, q_a)                          # RS.lmult(q_a)
    mul!(model.mat43, model.lmat, model.hmat)     # RS.lmult(q_a)*model.hmat
    
    model.vec2[1] = λt[4]
    model.vec2[2] = λt[5]
    mul!(model.vec3, Transpose(cmat), model.vec2)     # cmat'*λt[4:5]
    mul!(model.vec4, model.mat43, model.vec3)     # RS.lmult(q_a)*model.hmat*cmat'*λt[4:5]
    kron4!(model, model.vec4)                       # kron((RS.lmult(q_a)*model.hmat*cmat'*λt[4:5])',I(4))
    mul!(model.mat44, model.kron4_mtx, model.dlTdq)  #kron((RS.lmult(q_a)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    mul!(model.mat34, model.vmat, model.mat44)       #model.vmat*kron((RS.lmult(q_a)*model.hmat*cmat'*λt[4:5])',I(4))*∂Lᵀ∂q()
    mtx .+= model.mat34
    return 
end

# The jacobian of function Dfdyn
# most complicated function!
function Dfdyn(model::FloatingSpace,xt1, xt, ut, λt, dt)
    nb = model.nb
    ns = model.ns   # ns = 13*(nb+1)
    nr = 13         #size of one rigidbody
    nc = 5          #size of one joint constraint 
    nu = 6+nb       # size of all control ut
    # this function will have a lot of confusing index gymnastics
    nd = ns*2 + nu + nc*(nb)
    Dfmtx = spzeros(model.ns, nd)   # [xt1;xt;ut;λt]

    for link_id=1:model.nb+1
        Dfmtx_block = view(Dfmtx, (nr*(link_id-1)).+(1:nr),:) #13 x nd
        joint_before_id = link_id-1
        joint_after_id  = link_id
        # iterate through all rigid bodies
        r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, link_id) # a is the current link

        # get state from xt1
        rat1 = xt1[r_ainds]
        vat1 = xt1[v_ainds]
        qat1 = SVector{4}(xt1[q_ainds])
        wat1 = xt1[w_ainds]
        # get state from xt
        rat = xt[r_ainds]
        vat = xt[v_ainds]
        qat = SVector{4}(xt[q_ainds])
        wat = xt[w_ainds]
        # link_id==1 (the body) need special attention 
        # link_id==nb+1 (the last arm link)
        if (link_id == 1)  #the body link
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])
            # joint between a and b # use Gra
            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id]) 


            # Ma = diagm([model.body_mass,model.body_mass,model.body_mass])

            # derivative of eqn 1, position
            Dfmtx_block[1:3, r_ainds] .= I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+r_ainds)] .= -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+v_ainds)] .= -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, v_ainds] .= model.body_mass_mtx                   # 3x3
            Dfmtx_block[4:6, (ns .+v_ainds)] .= -model.body_mass_mtx              # 3x3
            Dfmtx_block[4:6, (ns + ns).+(1:3)] .= -I(3)*dt                    # 3x3   u[1:3]
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_after_id-1)).+(1:5))] .= - [-I;zeros(2,3)]'*dt   # 3x3

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, q_ainds] .= I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+q_ainds)] .= -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -wat'*wat);wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+w_ainds)] .= -dt/2*(-qat*wat'/sqrt(4/dt^2 -wat'*wat) + RS.lmult(qat)*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity

            λt_block = λt[(5*(link_id-1)).+(1:5)]
            vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            Gqamtx = Gqa(qat1,qbt1,vertices, joint_direction,model.joint_cmat[joint_after_id])
            ∂Gqaᵀλ∂qa1_mtx = ∂Gqaᵀλ∂qa(model, qat1,qbt1,λt_block,vertices, joint_direction,model.joint_cmat[joint_after_id])
            ∂Gqaᵀλ∂qb1_mtx = ∂Gqaᵀλ∂qb(model, qat1,qbt1,λt_block,vertices, joint_direction,model.joint_cmat[joint_after_id])

            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = model.body_inertias
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = wat1[1]; w2 = wat1[2]; w3 = wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, w_ainds] = [row1 row2 row3]'
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = wat[1]; w2 = wat[2]; w3 = wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, (ns .+w_ainds)] = [row1 row2 row3]'

            # d G_qa1t'*λt /dqa1t 
            Dfmtx_block[11:13, q_ainds] = -∂Gqaᵀλ∂qa1_mtx
            # d G_qa1t'*λt /dqb1t 
            Dfmtx_block[11:13, q_binds] = -∂Gqaᵀλ∂qb1_mtx

            Dfmtx_block[11:13, (ns*2).+(4:6)] =  -2*I(3)
            Dfmtx_block[11:13, (ns*2).+(6+joint_after_id)] =  2*joint_direction
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_after_id-1)).+(1:5)] = -Gqamtx' 

        elseif (link_id >= 2 && link_id < model.nb+1) # normal arm link
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            # get next link state from xt1
            r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            rbt1 = xt1[r_binds]
            qbt1 = SVector{4}(xt1[q_binds])
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])

            next_λt_block = λt[(5*(joint_after_id-1)).+(1:5)]
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]

            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id])  
            ∂Gqaᵀλ∂qa1_mtx = ∂Gqaᵀλ∂qa(model,qat1,qbt1,next_λt_block,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id])
            ∂Gqaᵀλ∂qb1_mtx = ∂Gqaᵀλ∂qb(model,qat1,qbt1,next_λt_block,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id])
            # joint between z and a  # use Grb
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])
            Gqbmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction,model.joint_cmat[joint_before_id])  
            ∂Gqbᵀλ∂qz1_mtx = ∂Gqbᵀλ∂qa(model,qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction,model.joint_cmat[joint_before_id])
            ∂Gqbᵀλ∂qa1_mtx = ∂Gqbᵀλ∂qb(model,qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction,model.joint_cmat[joint_before_id])

            # derivative of eqn 1, position
            Dfmtx_block[1:3, r_ainds] .= I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+r_ainds)] .= -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+v_ainds)] .= -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, v_ainds] .= model.arm_mass_mtx                   # 3x3
            Dfmtx_block[4:6, (ns .+v_ainds)] .= -model.arm_mass_mtx               # 3x3
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_before_id-1)).+(1:5))] .= - [I;zeros(2,3)]'*dt   # 3x3   # this for joint before 
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_after_id-1)).+(1:5))] .= - [-I;zeros(2,3)]'*dt   # 3x3   # this for joint before 

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, q_ainds] .= I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+q_ainds)] .= -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -wat'*wat);wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+w_ainds)] .= -dt/2*(-qat*wat'/sqrt(4/dt^2 -wat'*wat) + RS.lmult(qat)*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity
            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = model.arm_inertias
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = wat1[1]; w2 = wat1[2]; w3 = wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, w_ainds] = [row1 row2 row3]'
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = wat[1]; w2 = wat[2]; w3 = wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, (ns .+w_ainds)] = [row1 row2 row3]'


            # d G_qz1t'*λt /dqz1t  
            Dfmtx_block[11:13, q_zinds] = -∂Gqbᵀλ∂qz1_mtx
            # d G_qz1t'*λt /dqa1t 
            Dfmtx_block[11:13, q_ainds] = -∂Gqbᵀλ∂qa1_mtx -∂Gqaᵀλ∂qa1_mtx
            # d G_qa1t'*λt /dqb1t 
            Dfmtx_block[11:13, q_binds] = -∂Gqaᵀλ∂qb1_mtx

            Dfmtx_block[11:13, (ns*2).+(6+joint_before_id)] =  -2*prev_joint_direction
            Dfmtx_block[11:13, (ns*2).+(6+joint_after_id)] =  2*next_joint_direction
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_before_id-1)).+(1:5)] = -Gqbmtx' 
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_after_id-1)).+(1:5)] = -Gqamtx' 


        else # the last link
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])            
            # get previous link state from xt1
            r_zinds, v_zinds, q_zinds, w_zinds = fullargsinds(model, link_id-1) # z is the previous link
            rzt1 = xt1[r_zinds]
            qzt1 = SVector{4}(xt1[q_zinds])
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])
            Gqbmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id]) 
            ∂Gqbᵀλ∂qz1_mtx = ∂Gqbᵀλ∂qa(model,qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id])
            ∂Gqbᵀλ∂qa1_mtx = ∂Gqbᵀλ∂qb(model,qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id])

            # derivative of eqn 1, position
            Dfmtx_block[1:3, r_ainds] = I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+r_ainds)] = -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+v_ainds)] = -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, v_ainds] = model.arm_mass_mtx                   # 3x3
            Dfmtx_block[4:6, (ns .+v_ainds)] = -model.arm_mass_mtx              # 3x3
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_before_id-1)).+(1:5))] = - [I;zeros(2,3)]'*dt   # 3x3   # this for joint before 

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, q_ainds] = I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+q_ainds)] = -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -wat'*wat);wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+w_ainds)] = -dt/2*(-qat*wat'/sqrt(4/dt^2 -wat'*wat) + RS.lmult(qat)*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity
            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = model.arm_inertias
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = wat1[1]; w2 = wat1[2]; w3 = wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, w_ainds] = [row1 row2 row3]'
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = wat[1]; w2 = wat[2]; w3 = wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, (ns .+w_ainds)] = [row1 row2 row3]'

            # d G_qz1t'*λt /dqz1t 
            Dfmtx_block[11:13, q_zinds] = -∂Gqbᵀλ∂qz1_mtx
            # d G_qz1t'*λt /dqa1t 
            Dfmtx_block[11:13, q_ainds] = -∂Gqbᵀλ∂qa1_mtx

            Dfmtx_block[11:13, (ns*2).+(6+joint_before_id)] =  -2*prev_joint_direction
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_before_id-1)).+(1:5)] = -Gqbmtx' 
        end

    end
    return Dfmtx
end

# most complicated function!, save the jacobian in model.Dfmtx
function Dfdyn!(model::FloatingSpace,xt1, xt, ut, λt, dt)
    nb::Int = model.nb
    ns::Int = model.ns   # ns = 13*(nb+1)
    nr::Int = 13         #size of one rigidbody
    nc::Int = 5          #size of one joint constraint 
    nu::Int = 6+nb       # size of all control ut
    # this function will have a lot of confusing index gymnastics
    nd::Int = ns*2 + nu + nc*(nb)
    # Dfmtx = spzeros(model.ns, nd)   # [xt1;xt;ut;λt]

    J11::Float64 = 0.0;J12::Float64 = 0.0;J13::Float64 = 0.0;
    J21::Float64 = 0.0;J22::Float64 = 0.0;J23::Float64 = 0.0;
    J31::Float64 = 0.0;J32::Float64 = 0.;J33::Float64 = 0.0;
    w1::Float64 = 0.0; w2::Float64 = 0.0; w3::Float64 = 0.0;
    wk::Float64 = 0.0;
    n::Int = model.nb+1
    joint_before_id::Int = 0
    joint_after_id::Int  = 0
    for link_id=1:n
        # @time begin
        Dfmtx_block = view(model.Dfmtx, (nr*(link_id-1)).+(1:nr),:) #13 x nd
        joint_before_id = link_id-1
        joint_after_id  = link_id
        # iterate through all rigid bodies
        # r_ainds, v_ainds, q_ainds, w_ainds = fullargsinds(model, link_id) # a is the current link
        statea_inds!(model,link_id)

        # get state from xt1
        model.rat1[1] = xt1[model.r_ainds[1]]
        model.rat1[2] = xt1[model.r_ainds[2]]
        model.rat1[3] = xt1[model.r_ainds[3]]

        model.vat1[1] = xt1[model.v_ainds[1]]
        model.vat1[2] = xt1[model.v_ainds[2]]
        model.vat1[3] = xt1[model.v_ainds[3]]

        model.qat1[1] = xt1[model.q_ainds[1]]
        model.qat1[2] = xt1[model.q_ainds[2]]
        model.qat1[3] = xt1[model.q_ainds[3]]
        model.qat1[4] = xt1[model.q_ainds[4]]
        
        model.wat1[1] = xt1[model.w_ainds[1]]
        model.wat1[2] = xt1[model.w_ainds[2]]
        model.wat1[3] = xt1[model.w_ainds[3]]
        # get state from xt
        model.rat[1] = xt[model.r_ainds[1]]
        model.rat[2] = xt[model.r_ainds[2]]
        model.rat[3] = xt[model.r_ainds[3]]

        model.vat[1] = xt[model.v_ainds[1]]
        model.vat[2] = xt[model.v_ainds[2]]
        model.vat[3] = xt[model.v_ainds[3]]

        model.qat[1] = xt[model.q_ainds[1]]
        model.qat[2] = xt[model.q_ainds[2]]
        model.qat[3] = xt[model.q_ainds[3]]
        model.qat[4] = xt[model.q_ainds[4]]
        
        model.wat[1] = xt[model.w_ainds[1]]
        model.wat[2] = xt[model.w_ainds[2]]
        model.wat[3] = xt[model.w_ainds[3]]
        # link_id==1 (the body) need special attention 
        # link_id==nb+1 (the last arm link)
        if (link_id == 1)  #the body link
            # get next link state from xt1
            # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            # rbt1 = SVector{3}(xt1[r_binds])
            # qbt1 = SVector{4}(xt1[q_binds])

            stateb_inds!(model,link_id+1)
            model.rbt1[1] = xt1[model.r_binds[1]]
            model.rbt1[2] = xt1[model.r_binds[2]]
            model.rbt1[3] = xt1[model.r_binds[3]]
            
            model.qbt1[1] = xt1[model.q_binds[1]]
            model.qbt1[2] = xt1[model.q_binds[2]]
            model.qbt1[3] = xt1[model.q_binds[3]]
            model.qbt1[4] = xt1[model.q_binds[4]]

            # joint between a and b # use Gra
            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            # Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction,model.joint_cmat[joint_after_id]) 


            # Ma = diagm([model.body_mass,model.body_mass,model.body_mass])

            # derivative of eqn 1, position
            Dfmtx_block[1:3, model.r_ainds] = I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+model.r_ainds)] = -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+model.v_ainds)] = -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, model.v_ainds] .= model.body_mass_mtx                   # 3x3
            Dfmtx_block[4:6, (ns .+model.v_ainds)] .= -model.body_mass_mtx                   # 3x3
            Dfmtx_block[4:6, (ns + ns).+(1:3)] = -I(3)*dt                    # 3x3   u[1:3]
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_after_id-1)).+(1:5))] = - [-I;zeros(2,3)]'*dt   # 3x3

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, model.q_ainds] = I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+model.q_ainds)] = -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+model.w_ainds)] = -dt/2*(-model.qat*model.wat'/sqrt(4/dt^2 -model.wat'*model.wat) + RS.lmult(SVector{4}(model.qat))*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity

            λt_block = λt[(5*(link_id-1)).+(1:5)]
            vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            # Gqamtx = Gqa(qat1,qbt1,vertices, joint_direction, model.joint_cmat[joint_after_id])
            # ∂Gqaᵀλ∂qa1_mtx = ∂Gqaᵀλ∂qa(qat1,qbt1,λt_block,vertices, joint_direction, model.joint_cmat[joint_after_id])
            # ∂Gqaᵀλ∂qb1_mtx = ∂Gqaᵀλ∂qb(qat1,qbt1,λt_block,vertices, joint_direction, model.joint_cmat[joint_after_id])
            Gqa!(model, model.Gqamtx, model.qat1, model.qbt1, model.joint_vertices[joint_after_id], model.joint_cmat[joint_after_id])
            ∂Gqaᵀλ∂qa!(model, model.dGqbTλdqa, model.qat1, model.qbt1, λt_block, vertices,model.joint_cmat[joint_after_id]) 
            ∂Gqaᵀλ∂qb!(model, model.dGqbTλdqb, model.qat1, model.qbt1, λt_block, vertices,model.joint_cmat[joint_after_id]) 

            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = SMatrix{3,3}(model.body_inertias)
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = model.wat1[1]; w2 = model.wat1[2]; w3 = model.wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, model.w_ainds] .= [row1 row2 row3]'
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = model.wat[1]; w2 = model.wat[2]; w3 = model.wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, (ns .+model.w_ainds)] = [row1 row2 row3]'

            # d G_qa1t'*λt /dqa1t 
            Dfmtx_block[11:13, model.q_ainds] .= -model.dGqbTλdqa
            # d G_qa1t'*λt /dqb1t 
            Dfmtx_block[11:13, model.q_binds] .= -model.dGqbTλdqb

            Dfmtx_block[11:13, (ns*2).+(4:6)] .=  -2*I(3)
            Dfmtx_block[11:13, (ns*2).+(6+joint_after_id)] .=  2*model.joint_directions[joint_after_id]
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_after_id-1)).+(1:5)] .= -model.Gqamtx' 

        elseif (link_id >= 2 && link_id < n) # normal arm link
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])
            # get next link state from xt1
            # r_binds, v_binds, q_binds, w_binds = fullargsinds(model, link_id+1) # b is the next link
            # rbt1 = xt1[r_binds]
            # qbt1 = SVector{4}(xt1[q_binds])
            stateb_inds!(model,link_id+1)
            model.rbt1[1] = xt1[model.r_binds[1]]
            model.rbt1[2] = xt1[model.r_binds[2]]
            model.rbt1[3] = xt1[model.r_binds[3]]
            
            model.qbt1[1] = xt1[model.q_binds[1]]
            model.qbt1[2] = xt1[model.q_binds[2]]
            model.qbt1[3] = xt1[model.q_binds[3]]
            model.qbt1[4] = xt1[model.q_binds[4]]

            # get previous link state from xt1
            statez_inds!(model,link_id-1)
            model.rzt1[1] = xt1[model.r_zinds[1]]
            model.rzt1[2] = xt1[model.r_zinds[2]]
            model.rzt1[3] = xt1[model.r_zinds[3]]
            
            model.qzt1[1] = xt1[model.q_zinds[1]]
            model.qzt1[2] = xt1[model.q_zinds[2]]
            model.qzt1[3] = xt1[model.q_zinds[3]]
            model.qzt1[4] = xt1[model.q_zinds[4]]

            next_λt_block = λt[(5*(joint_after_id-1)).+(1:5)]
            prev_λt_block = λt[(5*(joint_before_id-1)).+(1:5)]

            next_vertices = model.joint_vertices[joint_after_id] # notice joint_vertices is 6x1
            next_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_after_id])
            # Gqamtx = Gqa(qat1,qbt1,next_vertices, next_joint_direction, model.joint_cmat[joint_after_id])  
            # ∂Gqaᵀλ∂qa1_mtx = ∂Gqaᵀλ∂qa(qat1,qbt1,next_λt_block,next_vertices, next_joint_direction, model.joint_cmat[joint_after_id])
            # ∂Gqaᵀλ∂qb1_mtx = ∂Gqaᵀλ∂qb(qat1,qbt1,next_λt_block,next_vertices, next_joint_direction, model.joint_cmat[joint_after_id])
            Gqa!(model, model.Gqamtx, model.qat1, model.qbt1, model.joint_vertices[joint_after_id], model.joint_cmat[joint_after_id])
            ∂Gqaᵀλ∂qa!(model, model.dGqaTλdqa, model.qat1, model.qbt1, next_λt_block, next_vertices,model.joint_cmat[joint_after_id]) 
            ∂Gqaᵀλ∂qb!(model, model.dGqaTλdqb, model.qat1, model.qbt1, next_λt_block, next_vertices,model.joint_cmat[joint_after_id]) 
            # joint between z and a  # use Grb
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            prev_joint_direction = convert(Array{Float64,1},model.joint_directions[joint_before_id])
            # Gqbmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id])  
            # ∂Gqbᵀλ∂qz1_mtx = ∂Gqbᵀλ∂qa(qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id])
            # ∂Gqbᵀλ∂qa1_mtx = ∂Gqbᵀλ∂qb(qzt1,qat1,prev_λt_block,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id])
            Gqb!(model, model.Gqbmtx, model.qzt1, model.qat1, model.joint_vertices[joint_before_id], model.joint_cmat[joint_before_id])
            ∂Gqbᵀλ∂qa!(model, model.dGqbTλdqa, model.qzt1, model.qat1, prev_λt_block, prev_vertices,model.joint_cmat[joint_before_id]) 
            ∂Gqbᵀλ∂qb!(model, model.dGqbTλdqb, model.qzt1, model.qat1, prev_λt_block, prev_vertices,model.joint_cmat[joint_before_id]) 

            # derivative of eqn 1, position
            Dfmtx_block[1:3, model.r_ainds] .= I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+model.r_ainds)] .= -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+model.v_ainds)] .= -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, model.v_ainds] .= model.arm_mass_mtx                   # 3x3
            
            model.mat33 .= model.arm_mass_mtx 
            model.mat33 .*= -1.0
            Dfmtx_block[4:6, (ns .+model.v_ainds)] .= model.mat33             # 3x3
            # Dfmtx_block[4:6, (ns .+model.v_ainds)] .= -model.arm_mass_mtx              # 3x3
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_before_id-1)).+(1:5))] .= - [I;zeros(2,3)]'*dt   # 3x3   # this for joint before 
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_after_id-1)).+(1:5))] .= - [-I;zeros(2,3)]'*dt   # 3x3   # this for joint before 

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, model.q_ainds] .= I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+model.q_ainds)] .= -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+model.w_ainds)] .= -dt/2*(-model.qat*model.wat'/sqrt(4/dt^2 -model.wat'*model.wat) + RS.lmult(SVector{4}(model.qat))*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity
            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = SMatrix{3,3}(model.arm_inertias)
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = model.wat1[1]; w2 = model.wat1[2]; w3 = model.wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, model.w_ainds] .= [row1 row2 row3]'
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = model.wat[1]; w2 = model.wat[2]; w3 = model.wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            row1 = [                    J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk, J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk, J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk]
            row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk,                     J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk, J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk]
            row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk, J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk,                     J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk]
            Dfmtx_block[11:13, (ns .+model.w_ainds)] .= [row1 row2 row3]'


            # d G_qz1t'*λt /dqz1t  
            model.mat34 .= model.dGqbTλdqa
            model.mat34 .*= -1.0
            Dfmtx_block[11:13, model.q_zinds] .= model.mat34
            # d G_qz1t'*λt /dqa1t 
            Dfmtx_block[11:13, model.q_ainds] .= -model.dGqbTλdqb 
            Dfmtx_block[11:13, model.q_ainds] .-= model.dGqaTλdqa
            # d G_qa1t'*λt /dqb1t 
            Dfmtx_block[11:13, model.q_binds] .= -model.dGqaTλdqb

            Dfmtx_block[11:13, (ns*2).+(6+joint_before_id)] .=  -2*prev_joint_direction
            Dfmtx_block[11:13, (ns*2).+(6+joint_after_id)] .=  2*next_joint_direction
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_before_id-1)).+(1:5)] .= -Transpose(model.Gqbmtx)
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_after_id-1)).+(1:5)] .= -Transpose(model.Gqamtx) 


        else # the last link
            # Ma = diagm([model.arm_mass,model.arm_mass,model.arm_mass])            
            # get previous link state from xt1
            statez_inds!(model,link_id-1)
            model.rzt1[1] = xt1[model.r_zinds[1]]
            model.rzt1[2] = xt1[model.r_zinds[2]]
            model.rzt1[3] = xt1[model.r_zinds[3]]
            
            model.qzt1[1] = xt1[model.q_zinds[1]]
            model.qzt1[2] = xt1[model.q_zinds[2]]
            model.qzt1[3] = xt1[model.q_zinds[3]]
            model.qzt1[4] = xt1[model.q_zinds[4]]

            prev_λt_block = SVector{5}(λt[(5*(joint_before_id-1)).+(1:5)])
            prev_vertices = model.joint_vertices[joint_before_id] # notice joint_vertices is 6x1
            # prev_joint_direction = SVector{3}(model.joint_directions[joint_before_id])
            # Gqbmtx = Gqb(qzt1,qat1,prev_vertices, prev_joint_direction, model.joint_cmat[joint_before_id]) 

            Gqb!(model, model.Gqzmtx, model.qzt1, model.qat1, model.joint_vertices[joint_before_id], model.joint_cmat[joint_before_id])
            # ∂Gqbᵀλ∂qz1_mtx = ∂Gqbᵀλ∂qa(SVector{4}(model.qzt1),SVector{4}(model.qat1),prev_λt_block,prev_vertices, prev_joint_direction, SMatrix{2,3}(model.joint_cmat[joint_before_id]))
            # ∂Gqbᵀλ∂qa1_mtx = ∂Gqbᵀλ∂qb(SVector{4}(model.qzt1),SVector{4}(model.qat1),prev_λt_block,prev_vertices, prev_joint_direction, SMatrix{2,3}(model.joint_cmat[joint_before_id]))
            ∂Gqbᵀλ∂qa!(model, model.dGqbTλdqa, model.qzt1, model.qat1, prev_λt_block, prev_vertices,model.joint_cmat[joint_before_id]) 
            ∂Gqbᵀλ∂qb!(model, model.dGqbTλdqb, model.qzt1, model.qat1, prev_λt_block, prev_vertices,model.joint_cmat[joint_before_id]) 
            # derivative of eqn 1, position
            Dfmtx_block[1:3, model.r_ainds] .= I(3)                 # 3x3   d[rat1 - (rat + vat*dt)]/d rat1
            Dfmtx_block[1:3, (ns .+model.r_ainds)] .= -I(3)           # 3x3   d[rat1 - (rat + vat*dt)]/d rat
            Dfmtx_block[1:3, (ns .+model.v_ainds)] .= -I(3)*dt        # 3x3   d[rat1 - (rat + vat*dt)]/d vat

            # derivative of eqn 3, velocity 
            Dfmtx_block[4:6, model.v_ainds] .= SMatrix{3,3}(model.arm_mass_mtx)                # 3x3
            Dfmtx_block[4:6, (ns .+model.v_ainds)] .= -SMatrix{3,3}(model.arm_mass_mtx)             # 3x3
            Dfmtx_block[4:6, (ns + ns + nu).+((5*(joint_before_id-1)).+(1:3))] .= - I(3)*dt   # 3x3   # this for joint before 

            # derivative of eqn 5, orientation 
            Dfmtx_block[7:10, model.q_ainds] .= I(4)                        # 4x4
            Dfmtx_block[7:10, (ns .+model.q_ainds)] .= -dt/2*RS.rmult(SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat]))   # 4x4
            Dfmtx_block[7:10, (ns .+model.w_ainds)] .= -dt/2*(-model.qat*model.wat'/sqrt(4/dt^2 -model.wat'*model.wat) + RS.lmult(SVector{4}(model.qat))*model.hmat)   # 4x3

            # derivative of eqn 7, angular velocity
            # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1, following are matlab code 
            Ja = SMatrix{3,3}(model.arm_inertias)
            J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
            J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
            J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
            w1 = model.wat1[1]; w2 = model.wat1[2]; w3 = model.wat1[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            model.mat33[1,1] = J31*w2 - J21*w3 + J11*wk - (w1*(J11*w1 + J12*w2 + J13*w3))/wk
            model.mat33[1,2] = J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*wk - (w2*(J11*w1 + J12*w2 + J13*w3))/wk
            model.mat33[1,3] = J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*wk - (w3*(J11*w1 + J12*w2 + J13*w3))/wk

            model.mat33[2,1] = J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*wk - (w1*(J21*w1 + J22*w2 + J23*w3))/wk
            model.mat33[2,2] = J12*w3 - J32*w1 + J22*wk - (w2*(J21*w1 + J22*w2 + J23*w3))/wk
            model.mat33[2,3] = J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*wk - (w3*(J21*w1 + J22*w2 + J23*w3))/wk

            model.mat33[3,1] = 2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*wk - (w1*(J31*w1 + J32*w2 + J33*w3))/wk
            model.mat33[3,2] = J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*wk - (w2*(J31*w1 + J32*w2 + J33*w3))/wk    
            model.mat33[3,3] = J23*w1 - J13*w2 + J33*wk - (w3*(J31*w1 + J32*w2 + J33*w3))/wk

            Dfmtx_block[11:13, model.w_ainds] .= model.mat33
            # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
            w1 = model.wat[1]; w2 = model.wat[2]; w3 = model.wat[3];
            wk = (4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)
            model.mat33[1,1] = J31*w2 - J21*w3 - J11*wk + (w1*(J11*w1 + J12*w2 + J13*w3))/wk
            model.mat33[1,2] = J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*wk + (w2*(J11*w1 + J12*w2 + J13*w3))/wk
            model.mat33[1,3] = J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*wk + (w3*(J11*w1 + J12*w2 + J13*w3))/wk
            model.mat33[2,1] = J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*wk + (w1*(J21*w1 + J22*w2 + J23*w3))/wk                     
            model.mat33[2,2] = J12*w3 - J32*w1 - J22*wk + (w2*(J21*w1 + J22*w2 + J23*w3))/wk
            model.mat33[2,3] = J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*wk + (w3*(J21*w1 + J22*w2 + J23*w3))/wk
            model.mat33[3,1] = 2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*wk + (w1*(J31*w1 + J32*w2 + J33*w3))/wk
            model.mat33[3,2] = J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*wk + (w2*(J31*w1 + J32*w2 + J33*w3))/wk
            model.mat33[3,3] = J23*w1 - J13*w2 - J33*wk + (w3*(J31*w1 + J32*w2 + J33*w3))/wk
            Dfmtx_block[11:13, (ns .+model.w_ainds)] .= model.mat33

            # d G_qz1t'*λt /dqz1t 
            model.dGqbTλdqa .*= -1.0
            Dfmtx_block[11:13, model.q_zinds] .= model.dGqbTλdqa
            # d G_qz1t'*λt /dqa1t 
            model.dGqbTλdqb .*= -1.0
            Dfmtx_block[11:13, model.q_ainds] .= model.dGqbTλdqb

            Dfmtx_block[11:13, (ns*2).+(6+joint_before_id)] .=  -2*model.joint_directions[joint_before_id]
            Dfmtx_block[11:13, (ns*2 + nu + 5*(joint_before_id-1)).+(1:5)] .= -model.Gqzmtx' 
        end
        # end
    end
    return
end

# function attiG_f
#13*(nb+1)*2 + 6 + (nb) + 5*(nb)   -  12*(nb+1)*2 + 6 + (nb) + 5*(nb) 
# attitude Jacobian So that Dfmtx * attiG_mtx ==> size(26,60) where 60 is the size of error state 
# this is really cumbersome to manually construct. something like state_diff_jacobian in Altro is definitely better
function fdyn_attiG(model::FloatingSpace,xt1, xt)
    n,m = size(model)
    n̄ = state_diff_size(model)
    nb = model.nb

    G1 = state_diff_jac(model,xt1)
    G2 = state_diff_jac(model,xt)
    attiG_mtx = spzeros(n*2 + 6 + (nb) + 5*(nb), n̄*2 + 6 + (nb) + 5*(nb) )
    attiG_mtx[1:n,1:n̄] = G1
    attiG_mtx[n+1:2*n,n̄+1:2*n̄] = G2
    attiG_mtx[2*n+1:end,2*n̄+1:end] = I(6 + (nb) + 5*(nb))
    return attiG_mtx
end
function fdyn_attiG!(model::FloatingSpace,xt1, xt)
    n,m = size(model)
    n̄ = state_diff_size(model)
    nb = model.nb

    state_diff_jac!(model,xt1)
    model.fdyn_attiG[1:n,1:n̄] .= model.attiG
    state_diff_jac!(model,xt)
    model.fdyn_attiG[n+1:2*n,n̄+1:2*n̄] .= model.attiG
    model.fdyn_attiG[2*n+1:end,2*n̄+1:end] .= I(6 + (nb) + 5*(nb))
    return
end

function get_vels(model::FloatingSpace, x)
    nb = model.nb
    vs = [x[(i-1)*13 .+ (4:6)] for i=1:nb+1]
    ωs = [x[(i-1)*13 .+ (11:13)] for i=1:nb+1]
    return vs, ωs
end
function get_configs_ind(model::FloatingSpace)
    n,m = size(model)
    ind = BitArray(undef, n)
    for i=1:model.nb
        ind[(i-1)*13 .+ (1:3)] .= 1
        ind[(i-1)*13 .+ (7:10)] .= 1
    end
    return ind
end

function get_vels_ind(model::FloatingSpace)
    n,m = size(model)
    ind = BitArray(undef, n)
    for i=1:model.nb+1
        ind[(i-1)*13 .+ (4:6)] .= 1
        ind[(i-1)*13 .+ (11:13)] .= 1
    end
    return ind
end
# calculate the position and orientation parts in x⁺ by propagate x
function propagate_config!(model::FloatingSpace, x⁺, x, dt) 
    nb = model.nb
    n,m = size(model)
    P = Lie_P(model)
    lie = RD.LieState(UnitQuaternion{eltype(x)}, P)

    vec = RD.vec_states(lie, x) 
    rot = RD.rot_states(lie, x) 

    vs, ωs = get_vels(model, x)
    for i=1:nb+1
        pos_ind =(i-1)*13 .+  (1:3)
        # due to the irregularity in vec state
        if i == 1
            x⁺[pos_ind] = vec[i][1:3] + vs[i]*dt
        else
            x⁺[pos_ind] = vec[i][4:6] + vs[i]*dt
        end
        
        rot_ind = (i-1)*13 .+ (7:10)
        x⁺[rot_ind] = dt/2 * RS.lmult(rot[i]) * [sqrt(4/dt^2 - ωs[i]'ωs[i]); ωs[i]]
    end

    return 
end

function Altro.is_converged(model::FloatingSpace, x)
    g!(model,x)
    return norm(model.g_val) < 1e-6
end

# x is the current state, x⁺ is the next state
# given current state x and current U
# use newton's method to solve for the vel part of x and the next state x⁺
function discrete_dynamics(model::FloatingSpace, x, u, λ_init, dt)
    n::Int,m::Int = size(model)
    n̄::Int = state_diff_size(model)
    nb1::Int = model.nb+1
    p::Int = model.p
    λ = zeros(eltype(x),5*model.nb)
    λ = λ_init
    x⁺ = Vector(x)
    for id=1:nb1
        x⁺[(13*(id-1)) .+ (1:3)] = x[(13*(id-1)) .+ (1:3)] + x[(13*(id-1)) .+ (4:6)]*dt
        model.wat .= x[(13*(id-1)) .+ (11:13)]
        model.qat .= x[(13*(id-1)) .+ (7:10)]
        x⁺[(13*(id-1)) .+ (7:10)] = dt/2*RS.lmult(SVector{4}(model.qat))*SVector{4}([sqrt(4/dt^2 -model.wat'*model.wat);model.wat])
    end

    x⁺_new, λ_new = copy(x⁺), copy(λ)    

    j::Int =0
    α::Float64 = 1
    ρ::Float64 = 0.5
    c::Float64 = 0.01 
    err::Float64 = 0
    err_new::Float64 = 0

    max_iters, line_iters, ϵ = 50, 10, 1e-6
    for i=1:max_iters  
        # Newton step    
        # 31 = 26 + 5
        fdyn!(model,x⁺, x, u, λ, dt)
        model.err_vec[1:model.ns] .= model.fdyn_vec
        gp1!(model,x⁺,dt)
        model.err_vec[model.ns+1:end] .= model.g_val

        err = norm(model.err_vec)
        # println(" err_vec: ", err)
        # jacobian of x+ and λ
        state_diff_jac!(model, x⁺)
        Dgp1!(model,x⁺,dt)
        # G = model.Dgmtx*model.attiG
        mul!(model.G,model.Dgmtx,model.attiG)

        Dfdyn!(model,x⁺, x, u, λ, dt)
        fdyn_attiG!(model,x⁺,x)
        begin
        mul!(model.Fdyn,model.Dfmtx,model.fdyn_attiG)

        # x⁺ , lambda
        # F = [Fdyn[:,1:n̄] Fdyn[:,n̄*2+6+nb+1:end];
        #         G  spzeros(5*nb,5*nb)]
        model.F[1:n,1:n̄] .= model.Fdyn[:,1:n̄]
        model.F[1:n,n̄+1:n̄+model.p] .= model.Fdyn[:,n̄*2+6+model.nb+1:end]
        model.F[n+1:n+model.p,1:n̄] .= model.G
        model.F[n+1:n+model.p,n̄+1:end] .= 0

        end
        model.Δs .= -model.F\model.err_vec  #(n̄+5*nb)x1
        # ldiv!(Δs, factorize(F), -err_vec)
        # backtracking line search
        j=0
        α = 1
        ρ = 0.5
        c = 0.01 

        err_new = err + 9999
        mul!(model.FΔs, model.F, model.Δs)
        while (err_new > err + c*α*(model.err_vec/err)'*model.FΔs)

            λ_new .= λ 
            λ_new .+= α*model.Δs[(n̄) .+ (1:5*model.nb)]
            model.Δx⁺ .= α*model.Δs[1:n̄] 

            # this is a hack, maps Δx⁺ from size n̄ to N
            # remap_Δx⁺ = atti_G*Δx⁺

            # # update velocities in x⁺
            # vel_inds = get_vels_ind(model)
            # x⁺_new[vel_inds] .= x⁺[vel_inds] + remap_Δx⁺[vel_inds]
            for id=1:nb1
                x⁺_new[(13*(id-1)) .+ (1:3)] = x⁺[(13*(id-1)) .+ (1:3)] + model.Δx⁺[(12*(id-1)) .+ (1:3)]
                x⁺_new[(13*(id-1)) .+ (4:6)] = x⁺[(13*(id-1)) .+ (4:6)] + model.Δx⁺[(12*(id-1)) .+ (4:6)]
                model.vec3 .= model.Δx⁺[(12*(id-1)) .+ (7:9)]
                model.vec4 .= x⁺[(13*(id-1)) .+ (7:10)]
                q_lmult!(model, model.vec4)
                model.vec4[1] = 1.0
                model.vec4[2] = model.vec3[1]
                model.vec4[3] = model.vec3[2]
                model.vec4[4] = model.vec3[3]
                mul!(model.qat1, model.lmat, model.vec4)
                model.qat1 ./=  sqrt(1+norm(model.vec3)^2)

                x⁺_new[(13*(id-1)) .+ (7:10)] .= model.qat1

                x⁺_new[(13*(id-1)) .+ (11:13)] = x⁺[(13*(id-1)) .+ (11:13)] + model.Δx⁺[(12*(id-1)) .+ (10:12)]
            end

            _, ωs⁺ = get_vels(model, x⁺_new)
            if all(1/dt^2 .>= dot(ωs⁺,ωs⁺))
                fdyn!(model,x⁺_new, x, u, λ_new, dt)
                model.err_vec[1:model.ns] .= model.fdyn_vec
                gp1!(model,x⁺_new,dt)
                model.err_vec[model.ns+1:end] .= model.g_val

                err_new = norm(model.err_vec)
            end
            α = α*ρ
            j += 1
        end
        # println(" steps: ", j)
        # println(" err_new: ", err_new)
        x⁺ .= x⁺_new
        λ .= λ_new
        if err_new < ϵ
            # println(round.(fdyn(x⁺, x, u, λ, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices),digits=6)')
            return x⁺, λ
        end
    end
    return x⁺, λ 
end


# end

# test: simulate and visualize 

# model = FloatingSpaceOrth(16)
# x0 = generate_config(model, [0.0;0.0;1.0;pi/2], fill.(pi/4,model.nb));

# Tf =6
# dt = 0.005
# N = Int(Tf/dt)

# mech = vis_mech_generation(model)
# x = x0
# λ_init = zeros(5*model.nb)
# λ = λ_init
# U = 0.3*rand(6+model.nb)
# # U[7] = 0.0001
# steps = Base.OneTo(Int(N))
# storage = CD.Storage{Float64}(steps,length(mech.bodies))
# for idx = 1:N
#     println("step: ",idx)
#     x1, λ1 = discrete_dynamics(model,x, U, λ, dt)
#     println(norm(fdyn(model,x1, x, U, λ1, dt)))
#     println(norm(g(model,x1)))
#     setStates!(model,mech,x1)
#     for i=1:model.nb+1
#         storage.x[i][idx] = mech.bodies[i].state.xc
#         storage.v[i][idx] = mech.bodies[i].state.vc
#         storage.q[i][idx] = mech.bodies[i].state.qc
#         storage.ω[i][idx] = mech.bodies[i].state.ωc
#     end
#     x = x1
#     λ = λ1
# end
# visualize(mech,storage, env = "editor")

# overload functions for Altro
# TODO: try to reuse previous λ_init
function RD.discrete_dynamics(::Type{Q}, model::FloatingSpace, x, u, t, dt) where Q
    # _init = zeros(5*model.nb)
    # init the initial lambda for the first time step. assume no time step is smaller than 1e-7
    if (t <= 1e-7)
        model.λ_tmp .= 0
    end
    x1, λ1 = discrete_dynamics(model,  x, u, model.λ_tmp, dt)
    model.λ_tmp .=λ1
    return x1
end
function Altro.discrete_dynamics_MC(::Type{Q}, model::FloatingSpace, x, u, t, dt) where Q
    # _init = zeros(5*model.nb)
    # init the initial lambda for the first time step. assume no time step is smaller than 1e-7
    if (t <= 1e-7)
        model.λ_tmp .= 0
    end
    x1, λ1 = discrete_dynamics(model,  x, u, model.λ_tmp, dt)
    model.λ_tmp .=λ1
    return x1,λ1
end


function Altro.discrete_jacobian_MC!(::Type{Q}, Dexp, model::FloatingSpace,
    z::AbstractKnotPoint{T,N,M′}, x⁺, λ) where {T,N,M′,Q<:RobotDynamics.Explicit}

    n,m = size(model)
    n̄ = state_diff_size(model)
    x = state(z) 
    u = control(z)
    dt = z.dt

    @assert dt != 0
    # λ_init = 1e-5*randn(5*model.nb)
    # x1, λ1 = discrete_dynamics(model,  x, u, λ_init, dt)

    # x' = f(x,u)  ∈ quaternion 
    # A = E(x')*∂f∂x E(x)


    # d(x', x, u, λ) = 0  ∈ R^n   "planning with attitude"
    # ∂d∂x'*E(x')
    # ∂d∂x*E(x)
    # A = (∂d∂x'*E(x'))^-1 * ∂d∂x*E(x)

    Dfdyn!(model,x⁺, x, u, λ, dt)
    fdyn_attiG!(model, x⁺, x)
    mul!(model.Dfmtx_error, model.Dfmtx, model.fdyn_attiG)
    # ldiv!(Dexp.∇f, factorize(model.Dfmtx[:,1:n]), -model.Dfmtx[:,n+1:end])
    # ldiv!(Dexp.A, factorize(model.Dfmtx_error[:,1:n̄]), -model.Dfmtx[:,n+1:end])
    # ldiv!(Dexp.B, factorize(model.Dfmtx_error[:,1:n̄]), -model.Dfmtx[:,n+1:end])
    # ldiv!(Dexp.C, factorize(model.Dfmtx_error[:,1:n̄]), -model.Dfmtx[:,n+1:end])
    Dexp.A = -sparse(model.Dfmtx_error[:,1:n̄])\model.Dfmtx_error[:,n̄+1:n̄*2]
    Dexp.B = -sparse(model.Dfmtx_error[:,1:n̄])\model.Dfmtx_error[:,n̄*2+1:n̄*2+6+model.nb]
    Dexp.C = -sparse(model.Dfmtx_error[:,1:n̄])\model.Dfmtx_error[:,n̄*2+6+model.nb+1:end]

    # index of q in n̄
    # ind = BitArray(undef, n̄)
    # for i=1:model.nb+1
    #     ind[(i-1)*12 .+ (1:3)] .= 1
    #     ind[(i-1)*12 .+ (7:9)] .= 1
    # end
    Dgp1!(model, x⁺,dt)
    state_diff_jac!(model,x⁺)
    # subattiG = attiG[get_configs_ind(model),ind]
    # G[:,ind] .= Dgmtx[:,get_configs_ind(model)]*subattiG
    # G .= Dgmtx*attiG
    mul!(Dexp.G, model.Dgmtx, model.attiG)
end

# function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::FloatingSpace, G)
#     # do nothing for floatingBaseSpace model 
# end

function generate_config_rc2mc(
    model::FloatingSpace, 
    base_translation, 
    base_rotations,
    base_v,
    base_ω,
    joint_angles)
    pin = zeros(3) # com position of the body link 
    """Base"""
    pin[1] = base_translation[1]
    pin[2] = base_translation[2]
    pin[3] = base_translation[3]
    prev_q = UnitQuaternion(base_rotations...)
    state = [pin;base_v;RS.params(prev_q);base_ω]

    """Arm link"""
    pin = pin+prev_q * [model.body_size/2,0,0]
    # find quaternion from joint angles
    rotations = []
    @assert length(joint_angles) == model.nb
    for i=1:length(joint_angles)
        axis = model.joint_directions[i]
        push!(rotations, 
            UnitQuaternion(AngleAxis(joint_angles[i], axis[1], axis[2], axis[3]))
        )
    end
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [model.arm_length/2,0,0] # assume all arms have equal length
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]
        # arm velocities can be calculated but doesn't matter for visualization
        # TODO: calculate arm velocities
        prev_q = link_q
        pin += 2*delta
    end
    return state
end
