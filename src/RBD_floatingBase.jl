
using RigidBodyDynamics # Need to install latest version without tag
const RBD = RigidBodyDynamics
using Rotations
using RobotDynamics
using StaticArrays, LinearAlgebra
using Rotations 
const RS = Rotations
using TrajectoryOptimization
const TO = TrajectoryOptimization
using ModernRoboticsBook
const MRB = ModernRoboticsBook
# using ConstrainedDynamics only for visualization
using ConstrainedDynamics
using ConstrainedDynamicsVis
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis

using TrajectoryOptimization
using Altro

struct FloatingSpaceRBD{T} <: LieGroupModel
    body_mass::T
    body_size::T
    arm_mass::T
    arm_width::T
    arm_length::T 
    body_mass_mtx::Array{T,2}
    arm_mass_mtx::Array{T,2}
    body_inertias::Diagonal{T,Array{T,1}}
    arm_inertias::Diagonal{T,Array{T,1}}
    joint_directions::Array{Array{T,1},1}
    joint_vertices::Vector{SizedVector{6,T,Vector{T}}} 
    g::T 
    nb::Integer     # number of arm links, total rigid bodies in the system will be nb+1
    Slist::AbstractMatrix 
    Blist::AbstractMatrix 
    M::Vector{AbstractMatrix} # every arm_link has a home position 
    tree::RigidBodyDynamics.Mechanism{Float64}
    function FloatingSpaceRBD{T}(nb, joint_directions) where {T<:Real} 
        g = 0      # in space, no gravity!
        body_mass = 10.0
        body_size = 0.5
        arm_mass = 1.0
        arm_width = 0.1
        arm_length = 1.0
        body_mass_mtx = diagm([body_mass,body_mass,body_mass])
        arm_mass_mtx = diagm([arm_mass,arm_mass,arm_mass])
        body_inertias = Diagonal(1 / 12 * body_mass * diagm([0.5^2 + 0.5^2;0.5^2 + 0.5^2;0.5^2 + 0.5^2]))
        arm_inertias = Diagonal(1 / 12 * arm_mass * diagm([0.1^2 + 0.1^2;0.1^2 + 1.0^2;1.0^2 + 0.1^2]))
        joint_vertices = [[body_size/2, 0, 0, -arm_length/2, 0, 0]]
        for i = 1:nb-1
            push!(joint_vertices,[arm_length/2, 0, 0, -arm_length/2, 0, 0])
        end


        # define zero gravity world
        world = RBD.RigidBody{Float64}("world")
        FloatingSpaceRobot = RBD.Mechanism(world; gravity=[0., 0., g])
    
        joint0 = RBD.Joint("joint0", RBD.QuaternionFloating{Float64}()) # joint connecting base to world
        # inertia frame of base
        base_inertia_frame = RBD.SpatialInertia(
            RBD.frame_after(joint0), 
            com = [0.0, 0.0, 0.0], # center of mass location with respect to joint, should be all zero for the base
            moment_about_com = body_inertias, # inertia matirx
            mass = body_mass
        )
        base = RBD.RigidBody("base", base_inertia_frame) # define body "base" using frame "base_inertia_frame"
        # this describes the position of joint 0 relative to the world
        before_joint0_to_world = RBD.Transform3D(
            RBD.frame_before(joint0), # frame before joint 0 should be world frame
            RBD.default_frame(world), # world is a body, this gets the world's body frame
            one(RotMatrix{3}), # now we just set rotation and translation as unit/zero
            SVector{3}([0., 0., 0.])
        )
        # now we connect "base" to "world" through "joint0" which is located at "before_joint0_to_world"
        RBD.attach!(FloatingSpaceRobot, world, base, joint0, joint_pose=before_joint0_to_world)
        
        """Adding arms"""
        joint = Array{Joint{Float64,RigidBodyDynamics.Revolute{Float64}}}(UndefInitializer(), nb)
        link = Array{RigidBodyDynamics.RigidBody{Float64}}(UndefInitializer(), nb)
    
        # first arm linking to base
        joint[1] = RBD.Joint(string("joint", 1), RBD.Revolute(joint_directions[1]))
        inertia_frame = RBD.SpatialInertia(
            RBD.frame_after(joint[1]), 
            com = [arm_length/2, 0, 0], # center of mass location with respect to joint
            # arm initialized at positive x direction
            moment_about_com = arm_inertias, # inertia matirx
            mass = arm_mass
        )
        link[1] = RBD.RigidBody(string("link", 1), inertia_frame)
        joint_pose = RBD.Transform3D(
                RBD.frame_before(joint[1]), 
                RBD.frame_after(joint0), # base frame
                one(RotMatrix{3}), 
                SVector{3}([body_size/2, 0., 0.]) # joint1 is off by half of the body size to positive x axis
            )
        RBD.attach!(FloatingSpaceRobot, base, link[1], joint[1], joint_pose = joint_pose)
    
        for idx = 2 : nb
            joint[idx] = RBD.Joint(string("joint", idx), RBD.Revolute(joint_directions[idx]))
            inertia_frame = RBD.SpatialInertia(
                RBD.frame_after(joint[idx]), 
                com = [arm_length/2, 0, 0], # center of mass location with respect to joint
                # arm initialized at positive x direction
                moment_about_com = arm_inertias, # inertia matirx
                mass = arm_mass
            )
            link[idx] = RBD.RigidBody(string("link", idx), inertia_frame)
            joint_pose = RBD.Transform3D(
                RBD.frame_before(joint[idx]), 
                RBD.frame_after(joint[idx - 1]), # base frame
                one(RotMatrix{3}), 
                SVector{3}([arm_length, 0., 0.]) # next joint is off by half of the arm length to positive x axis
            )
            RBD.attach!(FloatingSpaceRobot, link[idx-1], link[idx], joint[idx], joint_pose = joint_pose)
        end

        Blist = MRB.ScrewToAxis([-arm_length*(nb - 0.5); 0; 0], joint_directions[1], 0)'
        if nb >= 2
            for idx = 2:nb
                Blist = vcat(Blist, MRB.ScrewToAxis([-arm_length*(nb - idx + 0.5); 0; 0], joint_directions[idx], 0)')
            end
        end
        Blist = Blist'
        
        Slist = MRB.ScrewToAxis([body_size/2; 0; 0], joint_directions[1], 0)'
        if nb >= 2
            for idx = 2:nb
                Slist = vcat(Slist, MRB.ScrewToAxis([body_size/2 + arm_length*(idx-1); 0; 0], joint_directions[idx], 0)')
            end
        end
        Slist = Slist'

        M = []

        for idx=1:nb
            push!(M, [1  0  0  body_size/2 + arm_length*(idx - 0.5) ;
                0  1  0  0 ;
                0  0  1  0 ;
                0  0  0  1 ]);
        end

        new(body_mass,
            body_size,
            arm_mass,
            arm_width,
            arm_length,
            body_mass_mtx,
            arm_mass_mtx,
            body_inertias,
            arm_inertias,
            joint_directions,
            joint_vertices,
            g,
            nb,
            Slist,
            Blist,
            M,
            FloatingSpaceRobot
        )    
    end   
end

function FloatingSpaceOrthRBD(nb::Integer)
    joint_directions = [ i % 2 == 0 ? [0.0;1.0;0.0] : [0.0;0.0;1.0] for i=1:nb]
    FloatingSpaceRBD{Float64}(nb, joint_directions)
end

#helper function to get data from full state(7 + nb + 6 + nb)
function state_parts(model::FloatingSpaceRBD)
    iq = 1:4
    ix = 5:7
    iθ = 8:7+model.nb
    iω = 8+model.nb:10+model.nb 
    iv = 11+model.nb:13+model.nb 
    iϕ = 14+model.nb:13+model.nb*2
    return iq, ix, iθ, iω, iv, iϕ
end

# Specify the state and control dimensions
function RobotDynamics.state_dim(model::FloatingSpaceRBD)
    7 + 6 + model.nb * 2
end
function RobotDynamics.control_dim(model::FloatingSpaceRBD)
    6 + model.nb
end
# specify the lie state
Lie_P(model::FloatingSpaceRBD) = (0, 3 + 6 + model.nb * 2)
RobotDynamics.LieState(model::FloatingSpaceRBD{T}) where T = RobotDynamics.LieState(UnitQuaternion{T},Lie_P(model))

function RobotDynamics.dynamics(RBDmodel::FloatingSpaceRBD, x::AbstractVector{T}, u::AbstractVector{T}) where T
    ẋ =  zeros(eltype(x),length(x))
    rcstate = RBD.MechanismState{T}(RBDmodel.tree)
    result = RBD.DynamicsResult{T}(rcstate.mechanism)
    dynamics!(ẋ, result, rcstate, x, u)
    ẋ
end

function RobotDynamics.discrete_dynamics(RBDmodel::FloatingSpaceRBD, z::AbstractKnotPoint)
    x = state(z) 
    u = control(z)
    t = z.t 
    dt = z.dt
    k1 = RobotDynamics.dynamics(RBDmodel, x, u)
    k2 = RobotDynamics.dynamics(RBDmodel, x+dt/2*k1, u)
    k3 = RobotDynamics.dynamics(RBDmodel, x+dt/2*k2, u)
    k4 = RobotDynamics.dynamics(RBDmodel, x+dt*k3, u)
    return xnext = x + dt/6.0*(k1 + 2 * k2 + 2 * k3 + k4)
end

function generate_config_rc2mc(
    model::FloatingSpaceRBD, 
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

function vis_mech_generation(model::FloatingSpaceRBD)
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

function view_single_state(model::FloatingSpaceRBD, q, v)
    mech = vis_mech_generation(model)
    storage_single = CD.Storage{Float64}(Base.OneTo(1),length(mech.bodies))
    state = generate_config_rc2mc(model, q[5:7], q[1:4], v[4:6], v[1:3], q[8:end])
    for i=1:model.nb+1
        storage_single.x[i][1] = SA[state[(i-1)*13 .+ (1:3)]...] 
        storage_single.v[i][1] = SA[state[(i-1)*13 .+ (4:6)]...] 
        storage_single.q[i][1] = UnitQuaternion(state[(i-1)*13 .+ (7:10)]...)
        storage_single.ω[i][1] = SA[state[(i-1)*13 .+ (11:13)]...] 
    end
    visualize(mech, storage_single, env = "editor")
end

function view_sequence(model::FloatingSpaceRBD, qs, vs)
    N = length(qs) - 1
    mech = vis_mech_generation(model)
    steps = Base.OneTo(Int(N))
    storage = CD.Storage{Float64}(steps,length(mech.bodies))
    for idx = 1:N
        state = generate_config_rc2mc(model, qs[idx][5:7], qs[idx][1:4], vs[idx][4:6], vs[idx][1:3], qs[idx][8:end])
        for i=1:model.nb+1
            storage.x[i][idx] = SA[state[(i-1)*13 .+ (1:3)]...] 
            storage.v[i][idx] = SA[state[(i-1)*13 .+ (4:6)]...] 
            storage.q[i][idx] = UnitQuaternion(state[(i-1)*13 .+ (7:10)]...)
            storage.ω[i][idx] = SA[state[(i-1)*13 .+ (11:13)]...] 
        end
    end
    visualize(mech,storage, env = "editor")
end


#my own quaternion to rotation matrix function because I am afraid Rotations.jl will perform weiredly during ForwardDiff
function q_to_rot(Q::AbstractVector{T}) where T # q must be 4-array
    q0 = Q[1]
    q1 = Q[2]
    q2 = Q[3]
    q3 = Q[4]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
        
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
        
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return SA{T}[r00 r01 r02
                 r10 r11 r12
                 r20 r21 r22]
end
# extract world velocity from state
function world_vel(model::FloatingSpaceRBD{T}, z::AbstractVector{P}) where {T, P}
    iq, ix, iθ, iω, iv, iϕ = state_parts(model)
    n,m=size(model)
    x = z[1:n]
    thetalist = x[iθ]
    ϕlist = x[iϕ]

    fk = MRB.FKinSpace(model.M[model.nb], model.Slist, thetalist)

    body_frame_vel = (MRB.JacobianBody(model.Blist, thetalist)*ϕlist)[4:6]
    base_frame_vel = fk[1:3,1:3] * body_frame_vel
    world_frame_vel = q_to_rot(x[iq])  * base_frame_vel + x[iv] + q_to_rot(x[iq]) * cross(x[iω], fk[1:3,4])

    return world_frame_vel
end

# extract link world pos 
function arm_world_pos(model::FloatingSpaceRBD{T}, z::AbstractVector{P}, link_id::Int)  where {T, P}
    # TODO: check link_id >=1 <=nb
    iq, ix, iθ, iω, iv, iϕ = state_parts(model)
    n,m=size(model)
    x = z[1:n]
    thetalist = x[iθ]

    fk = MRB.FKinSpace(model.M[link_id], model.Slist[:,1:link_id], thetalist[1:link_id])
    pos = fk[1:3,4] + x[ix]
    return pos
end

"""Constraint"""
struct EFVConstraint{S,W,T} <: TO.StageConstraint
	n::Int
	m::Int
	model::FloatingSpaceRBD{T}
    maxV::Float64
	sense::S
    inds::SVector{W, Int}
	function EFVConstraint(n::Int, m::Int, model::FloatingSpaceRBD{T}, maxV::Float64,
			sense::TO.ConstraintSense, inds=1:n+m) where {W, T}
        inds = SVector{m+n}(inds)
		new{typeof(sense),n+m,T}(n,m,model,maxV,sense,inds)
	end
end
TO.sense(con::EFVConstraint) = con.sense
TO.state_dim(con::EFVConstraint) = con.n
TO.control_dim(con::EFVConstraint) = con.m
@inline Base.length(con::EFVConstraint{S,T}) where {S,T} = 6   # v<vmax , v>vmin

function TO.evaluate(con::EFVConstraint{S,T}, z::AbstractKnotPoint) where {S,T}
    v = world_vel(con.model, [state(z);control(z)])
    #v < vmax
    # v > -vmax ----> -v < vmax
    return [v.-con.maxV; (-v).-con.maxV]   #  Inequality  should all smaller than 0
end

# dim should be 6 x size(z)
function TO.jacobian!(∇c, con::EFVConstraint{S,T}, z::AbstractKnotPoint) where {S,T}
    vaug(k) = world_vel(con.model, k)
    Jv = ForwardDiff.jacobian(vaug,[state(z);control(z)])

    ∇c[1:3,:] .= Jv
    ∇c[4:6,:] .= -Jv

	return true
end
# asuume gauss newton
TO.∇jacobian!(G, con::EFVConstraint{S,T}, z::AbstractKnotPoint, λ::AbstractVector) where {S,T} = true # zeros



struct LinkPosConstraint{S,W,T} <: TO.StageConstraint
	n::Int
	m::Int
    p::Int   # size of the constraint 
	model::FloatingSpaceRBD{T}
    max_pos::Array{Float64,1}
    min_pos::Array{Float64,1}
	sense::S
    inds::SVector{W, Int}
	function LinkPosConstraint(n::Int, m::Int, model::FloatingSpaceRBD{T}, max_pos::Array{Float64,1}, min_pos::Array{Float64,1},
			sense::TO.ConstraintSense, inds=1:n+m) where {W, T}
        inds = SVector{m+n}(inds)
        p = 3*model.nb*2
		new{typeof(sense),n+m,T}(n, m, p, model, max_pos, min_pos, sense, inds)
	end
end
TO.sense(con::LinkPosConstraint) = con.sense
TO.state_dim(con::LinkPosConstraint) = con.n
TO.control_dim(con::LinkPosConstraint) = con.m
@inline Base.length(con::LinkPosConstraint{S,T}) where {S,T} = con.p   # each link pos is 3, then min_pos < pos < max_pos

function TO.evaluate(con::LinkPosConstraint{S,W,T}, z::AbstractKnotPoint) where {S,W,T}
    con_val = zeros(T,con.p)
    for idx=1:con.model.nb
        pos = arm_world_pos(con.model, [state(z);control(z)], idx)
        con_val[6*(idx-1).+(1:3)] = pos.-con.max_pos[3*(idx-1).+(1:3)] 
        con_val[6*(idx-1).+(4:6)] = (-pos).+con.min_pos[3*(idx-1).+(1:3)] 
    end
    return con_val   #  Inequality  should all smaller than 0
end

# dim should be 6 x size(z)
function TO.jacobian!(∇c, con::LinkPosConstraint{S,T}, z::AbstractKnotPoint) where {S,T}

    for idx=1:con.model.nb
        posaug(k) = arm_world_pos(con.model, k, idx)
        Jpos = ForwardDiff.jacobian(posaug,[state(z);control(z)])

        ∇c[6*(idx-1).+(1:3),:] .= Jpos
        ∇c[6*(idx-1).+(4:6),:] .= -Jpos
    end

	return true
end
# asuume gauss newton
TO.∇jacobian!(G, con::LinkPosConstraint{S,T}, z::AbstractKnotPoint, λ::AbstractVector) where {S,T} = true # zeros