
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

struct FloatingSpaceRBD_UA{T} <: LieGroupModel
    body_mass::T
    body_size::T
    arm_mass::T
    arm_width::T
    arm_length::T 
    body_mass_mtx::Array{T,2}
    arm_mass_mtx::Array{T,2}
    body_inertias::Diagonal{T,Array{T,1}}
    arm_inertias::Diagonal{T,Array{T,1}}
    _joint_directions::Array{Array{T,1},1}
    g::T 
    nb::Integer     # number of arm links, total rigid bodies in the system will be nb+1
    Slist::AbstractMatrix 
    Blist::AbstractMatrix 
    M::AbstractMatrix 
    tree::RigidBodyDynamics.Mechanism{Float64}
    function FloatingSpaceRBD_UA{T}(nb, _joint_directions) where {T<:Real} 
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


        # define zero gravity world
        world = RBD.RigidBody{Float64}("world")
        FloatingSpaceRobot = RBD.Mechanism(world; gravity=[0., 0., 0.])
    
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
        joint[1] = RBD.Joint(string("joint", 1), RBD.Revolute(_joint_directions[1]))
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
            joint[idx] = RBD.Joint(string("joint", idx), RBD.Revolute(_joint_directions[idx]))
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

        Blist = [
        MRB.ScrewToAxis([-arm_length*2.5; 0; 0], _joint_directions[1], 0)'; 
        MRB.ScrewToAxis([-arm_length*1.5; 0; 0], _joint_directions[2], 0)'; 
        MRB.ScrewToAxis([-arm_length*0.5; 0; 0], _joint_directions[3], 0)']'

        Slist = [
        MRB.ScrewToAxis([body_size/2; 0; 0], _joint_directions[1], 0)'; 
        MRB.ScrewToAxis([body_size/2 + arm_length; 0; 0], _joint_directions[2], 0)'; 
        MRB.ScrewToAxis([body_size/2 + arm_length*2; 0; 0], _joint_directions[3], 0)']'

        M =[1  0  0  body_size/2 + arm_length*2.5 ;
            0  1  0  0 ;
            0  0  1  0 ;
            0  0  0  1 ];

        new(body_mass,
            body_size,
            arm_mass,
            arm_width,
            arm_length,
            body_mass_mtx,
            arm_mass_mtx,
            body_inertias,
            arm_inertias,
            _joint_directions,
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
    _joint_directions = [ i % 2 == 0 ? [0.0;1.0;0.0] : [0.0;0.0;1.0] for i=1:nb]
    FloatingSpaceRBD_UA{Float64}(nb, _joint_directions)
end

#helper function to get data from full state(7 + nb + 6 + nb)
function state_parts(model::FloatingSpaceRBD_UA)
    iq = 1:4
    ix = 5:7
    iθ = 8:7+model.nb
    iω = 8+model.nb:10+model.nb 
    iv = 11+model.nb:13+model.nb 
    iϕ = 14+model.nb:13+model.nb*2
    return iq, ix, iθ, iω, iv, iϕ
end

# Specify the state and control dimensions
function RobotDynamics.state_dim(model::FloatingSpaceRBD_UA)
    7 + 6 + model.nb * 2
end
function RobotDynamics.control_dim(model::FloatingSpaceRBD_UA)
    6 + model.nb - 2
end
# specify the lie state
Lie_P(model::FloatingSpaceRBD_UA) = (0, 3 + 6 + model.nb * 2)
RobotDynamics.LieState(model::FloatingSpaceRBD_UA{T}) where T = RobotDynamics.LieState(UnitQuaternion{T},Lie_P(model))

function RobotDynamics.dynamics(RBDmodel::FloatingSpaceRBD_UA, x::AbstractVector{T}, u::AbstractVector{T}) where T
    ẋ =  zeros(eltype(x),length(x))
    rcstate = RBD.MechanismState{T}(RBDmodel.tree)
    result = RBD.DynamicsResult{T}(rcstate.mechanism)
    dynamics!(ẋ, result, rcstate, x, [0.; 0.; u])
    ẋ
end

function RobotDynamics.discrete_dynamics(RBDmodel::FloatingSpaceRBD_UA, z::AbstractKnotPoint)
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

struct EFVConstraint{S,T} <: TO.StageConstraint
	n::Int
	m::Int
	model::FloatingSpaceRBD_UA{T}
    maxV::Float64
	sense::S
	function EFVConstraint(n::Int, m::Int, model::FloatingSpaceRBD_UA{T}, maxV::Float64,
			sense::TO.ConstraintSense) where {T}
		new{typeof(sense), T}(n,m,model,maxV,sense)
	end
end
TO.sense(con::EFVConstraint) = con.sense
TO.state_dim(con::EFVConstraint) = con.n
TO.control_dim(con::EFVConstraint) = con.m
@inline Base.length(con::EFVConstraint{S,T}) where {S,T} = 6   # v<vmax , v>vmin

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
function world_vel(model::FloatingSpaceRBD_UA{T}, z::AbstractVector{P}) where {T, P}
    iq, ix, iθ, iω, iv, iϕ = state_parts(model)
    n,m=size(model)
    x = z[1:n]
    thetalist = x[iθ]
    ϕlist = x[iϕ]

    fk = MRB.FKinSpace(model.M, model.Slist, thetalist)

    body_frame_vel = (MRB.JacobianBody(model.Blist, thetalist)*ϕlist)[4:6]
    base_frame_vel = fk[1:3,1:3] * body_frame_vel
    world_frame_vel = q_to_rot(x[iq])  * base_frame_vel + x[iv] + q_to_rot(x[iq]) * cross(x[iω], fk[1:3,4])

    return world_frame_vel
end

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