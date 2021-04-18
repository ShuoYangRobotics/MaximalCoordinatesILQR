import Pkg; Pkg.activate(joinpath(@__DIR__)); 
Pkg.instantiate();


using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using StaticArrays
using LinearAlgebra
using SparseArrays
using Rotations
using ForwardDiff
using Test

const RS = Rotations
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis
const CC = ConstrainedControl

""" helper functions for simple two link system """
function fullargsinds(i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end
function getStates(mech, sol=true)
    nb = length(mech.bodies)
    z = zeros(13*nb)
    for (i, body) in enumerate(mech.bodies)        
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        f = sol ? CD.fullargssol : CD.posargsk
        z[xinds],z[vinds],q,z[ωinds] = f(body.state)
        z[qinds] = RS.params(q)
    end
    return z
end

# state config x v q w 
function generate_config(mech, body_pose, rotations)
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))
    state = [pin;zeros(3);RS.params(prev_q);zeros(3)]
    pin = pin+prev_q * [mech.bodies[1].shape.xyz[1]/2,0,0]
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [arm_width/2,0,0]
        link_x = pin+delta
        state = [state; link_x;zeros(3);RS.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end

# state config x v q w 
function generate_config_with_rand_vel(mech, body_pose, rotations)
    pin = zeros(3)
    pin[1] = body_pose[1]
    pin[2] = body_pose[2]
    pin[3] = body_pose[3]
    prev_q = UnitQuaternion(RotZ(body_pose[4]))
    state = [pin;0.01*randn(3);RS.params(prev_q);0.01*randn(3)]
    pin = pin+prev_q * [mech.bodies[1].shape.xyz[1]/2,0,0]
    for i = 1:length(rotations)
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [arm_width/2,0,0]
        link_x = pin+delta
        state = [state; link_x;0.01*randn(3);RS.params(link_q);0.01*randn(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end


# body pose is x y z thetaz, θ only have z rotation
function generate_config(mech, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    rotations = []
    # joints are arranged orthogonally
    for i=1:length(θ)
        push!(rotations, UnitQuaternion(RotZ(θ[i])))
    end
    return generate_config(mech, body_pose, rotations)
end
# body pose is x y z thetaz, θ only have z rotation
function generate_config_with_rand_vel(mech, body_pose::Vector{<:Number}, θ::Vector{<:Number})
    rotations = []
    # joints are arranged orthogonally
    for i=1:length(θ)
        push!(rotations, UnitQuaternion(RotZ(θ[i])))
    end
    return generate_config_with_rand_vel(mech, body_pose, rotations)
end

# setposition setvelocity modifies state.xc qc..... 
function setStates!(mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

function state_parts(mech, x, u)
    xd = SArray{Tuple{3},Float64,1,3}[]
    vd = SArray{Tuple{3},Float64,1,3}[]
    qd = UnitQuaternion{Float64}[]
    ωd = SArray{Tuple{3},Float64,1,3}[]
    Fτd = []
    for i=1:2
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        push!(xd, x[xinds])
        push!(vd, x[vinds])
        push!(qd, UnitQuaternion(x[qinds]...))
        push!(ωd, x[ωinds])
    end

    push!(Fτd, SA[0.;0.;0.;0.;0.;0.])
    push!(Fτd, SA[u[1]])

    return xd, vd, qd, ωd, Fτd
end

""" Parameters """
timeStep = 10.0 # seconds

""" Define the base """
length1 = 0.5
width, depth = 0.5, 0.5
# Corner vectors
corners = [
    [[length1 / 2;length1 / 2;-length1 / 2]]
    [[length1 / 2;-length1 / 2;-length1 / 2]]
    [[-length1 / 2;length1 / 2;-length1 / 2]]
    [[-length1 / 2;-length1 / 2;-length1 / 2]]
    [[length1 / 2;length1 / 2;length1 / 2]]
    [[length1 / 2;-length1 / 2;length1 / 2]]
    [[-length1 / 2;length1 / 2;length1 / 2]]
    [[-length1 / 2;-length1 / 2;length1 / 2]]
]
# Initial orientation
##
ϕ1 = 0;
q1 = UnitQuaternion(RotZ(ϕ1))

# Define base link
origin = Origin{Float64}()
link0 = Box(width, depth, length1, 1., color = RGBA(1., 1., 0.))
link0.m = 10.0 # set base mass

# Constraints on base
# TODO: find what "Friction()" is doing
impacts = [InequalityConstraint(Friction(link0,[0;0;1.0], 0.01; p = corners[i])) for i=1:8] # above ground
world2base = EqualityConstraint(Floating(origin, link0)) # free floating

""" Define arms """
arm_length = 0.1
arm_width = 1.0
arm_depth = 0.1

# Define arm link
link1 = Box(arm_width, arm_depth, arm_length, arm_length, color = RGBA(0., 1., 0.))
link1.m = 1.0
Inerita_a = diagm([1/12*link0.m*(width^2+depth^2),1/12*link0.m*(width^2+depth^2),1/12*link0.m*(width^2+depth^2)])
Inerita_b = diagm([1/12*link1.m*(arm_length^2+arm_depth^2),1/12*link1.m*(arm_width^2+arm_depth^2),1/12*link1.m*(arm_width^2+arm_depth^2)])


# Constraints on the arms
joint1_axis = [0;0;1] # joint 1 rotates about z axis

vert01 = [width/2; 0; 0] # connection offset from link0 to joint1
vert11 = [-arm_width/2; 0; 0] # connection offset from link1 to joint1

joint1 = EqualityConstraint(Revolute(link0, link1, joint1_axis; p1=vert01,p2=vert11)) # joint1 : base to link1

# put them together
links = [link0; link1]
eqcs = [world2base; joint1]
ineqcs = impacts

mech = Mechanism(origin, links, eqcs, ineqcs) # TODO: this function is mutating!!!
# setPosition!(link0, x = [0.;0.;1.])
# setVelocity!(link0, v = [0;0;0], ω = [0;0;0])
# setPosition!(link1, x = [width/2;0.;1.])
# setVelocity!(link1, v = [0;0;0], ω = [0;0;0])


""" test state """
x0 = generate_config(mech, [2.0;2.0;1.0;pi/2], [pi/2]);
u0 = [0]
xd, vd, qd, ωd, Fτd = state_parts(mech, x0,u0)
dr = pi/140
x1 = generate_config(mech, [2.0;2.0;1.0;pi/2+dr], [pi/2+dr]);
xdp, vdp, qdp, ωdp, Fτd = state_parts(mech, x1,u0)
# x0 = generate_config(mech, [0.0;0.0;1.0;0.0], [0.0]);
reshape(x0,(13,2))'
# visualize state 
setStates!(mech,x0)
CD.discretizestate!(mech) # I finally found that this is very important
# each body.state has a lot of terms. setStates only modifies xc qc.. (continous state)
# this discretizestate! convert xc to xk (discretized states)
reshape(round.(getStates(mech)',digits=3),(13,2))'
steps = Base.OneTo(1)
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for i=1:2
    storage.x[i][1] = mech.bodies[i].state.xc
    storage.v[i][1] = mech.bodies[i].state.vc
    storage.q[i][1] = mech.bodies[i].state.qc
    storage.ω[i][1] = mech.bodies[i].state.ωc
end
visualize(mech,storage, env = "editor")

bodyids = getid.(mech.bodies)
eqcids = getid.(mech.eqconstraints)
"""test constraint"""
# this is in CD->main_components->equalityConstraint.jl
constraint1 = CD.g(mech, geteqconstraint(mech, eqcids[2])) 
eqc = geteqconstraint(mech, eqcids[2])
c = CD.g(eqc.constraints[1], getbody(mech, eqc.parentid), getbody(mech, eqc.childids[1]), mech.Δt)# nonzero?
# typeof(eqc.constraints[1]) ===> ConstrainedDynamics.Translational{Float64,3}
# eqc.constraints[1] is a joint
# the above function should be CD->joints->abstract_joint.jl line 14
CD.constraintmat(eqc.constraints[2])
c2 = CD.g(eqc.constraints[2], getbody(mech, eqc.parentid).state, getbody(mech, eqc.childids[1]).state,  mech.Δt)# nonzero?
# the above function should be  CD->joints->joint.jl
# these function uses xk qk . Why state has some many different terms?
xa = CD.posargsnext(getbody(mech, eqc.parentid).state, mech.Δt)
xb = CD.posargsnext(getbody(mech, eqc.childids[1]).state, mech.Δt)
vertices = eqc.constraints[1].vertices # COM pos of front link and back link
CD.vrotate(xb[1] + CD.vrotate(vertices[2], xb[2]) - (xa[1] + CD.vrotate(vertices[1], xa[2])), inv(xa[2]))

""" test constraint jacobian """
# function test(mechanism, eqc::EqualityConstraint{T,N,Nc}, Fτ::AbstractVector) where {T,N,Nc}
#     println(Nc)
#     println(N)
#     @assert length(Fτ)==3*Nc-N
# end

# for (i,id) in enumerate(eqcids)
#     test(mech, geteqconstraint(mech, id), Fτd[i])
# end
A1, B1, C1, G1 = CC.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 

# visualize state 
setStates!(mech,x1)
CD.discretizestate!(mech) 

constraint2 = CD.g(mech, geteqconstraint(mech, eqcids[2])) 
# get the rotation error 


x0 = generate_config_with_rand_vel(mech, [2.0;2.0;1.0;pi/2], [pi/2]);
dr = pi/140
x1 = generate_config_with_rand_vel(mech, [2.0;2.0;1.0;pi/2+dr], [pi/2+dr]);
xd, vd, qd, ωd, Fτd = state_parts(mech, x0,u0)
xdp, vdp, qdp, ωdp, Fτd = state_parts(mech, x1,u0)

# x1 - x0 (x1 = x0 + dx)
function states_error(x1, x0)
    state_error[12*0 .+ (1:3)] = x1[13*0 .+ (1:3)]-x0[13*0 .+ (1:3)]
    state_error[12*0 .+ (4:6)] = x1[13*0 .+ (4:6)]-x0[13*0 .+ (4:6)]
    # about rotation error 
    #   q3 = q2*q1   --> q1 = q2\q3  === q2'*q3 
    #   rotation_error(R1 , R2 )  --->   dR = R2\R1 ==== R2'*R1   ---> R1 = R2*dR 
    state_error[12*0 .+ (7:9)] = RS.rotation_error(x1[13*0 .+ (7:10)],x0[13*0 .+ (7:10)], RS.CayleyMap()) # Eqn 12  phi^{-1}(qdp'*qd)
    state_error[12*0 .+ (10:12)] = x1[13*0 .+ (11:13)]-x0[13*0 .+ (11:13)]
    state_error[12*1 .+ (1:3)] = x1[13*1 .+ (1:3)]-x0[13*1 .+ (1:3)]
    state_error[12*1 .+ (4:6)] = x1[13*1 .+ (4:6)]-x0[13*1 .+ (4:6)]
    state_error[12*1 .+ (7:9)] = RS.rotation_error(x1[13*1 .+ (7:10)],x0[13*1 .+ (7:10)], RS.CayleyMap())
    state_error[12*1 .+ (10:12)] = x1[13*1 .+ (11:13)]-x0[13*1 .+ (11:13)]
end
# x v q w 
state_error = zeros(24)
# x1 - x0
state_error[12*0 .+ (1:3)] = xdp[1]-xd[1]
state_error[12*0 .+ (4:6)] = vdp[1]-vd[1]
# about rotation error 
#   q3 = q2*q1   --> q1 = q2\q3  === q2'*q3 
#   rotation_error(R1 , R2 )  --->   dR = R2\R1 ==== R2'*R1   ---> R1 = R2*dR 
state_error[12*0 .+ (7:9)] = RS.rotation_error(qdp[1],qd[1], RS.CayleyMap()) # Eqn 12  phi^{-1}(qdp'*qd)
state_error[12*0 .+ (10:12)] = ωdp[1]-ωd[1]
state_error[12*1 .+ (1:3)] = xdp[2]-xd[2]
state_error[12*1 .+ (4:6)] = vdp[2]-vd[2]
state_error[12*1 .+ (7:9)] = RS.rotation_error(qdp[2],qd[2], RS.CayleyMap())
state_error[12*1 .+ (10:12)] = ωdp[2]-ωd[2]

G1*state_error # should be very close to zero, but not, so Jan's code may have wrong Jacobian

""" try my own constraint and jacobian """
function g(x,vertices)
    r_a = SVector{3}(x[13*0 .+ (1:3)])
    r_b = SVector{3}(x[13*1 .+ (1:3)])
    q_a = SVector{4}(x[13*0 .+ (7:10)])
    q_b = SVector{4}(x[13*1 .+ (7:10)])

    val = zeros(eltype(x),5)
    val[1:3] = (r_b + RS.vmat()*RS.rmult(q_b)'*RS.lmult(q_b)*RS.hmat()*vertices[2]) - 
    (r_a + RS.vmat()*RS.rmult(q_a)'*RS.lmult(q_a)*RS.hmat()*vertices[1])
    tmp = RS.vmat()*RS.lmult(q_a)'*q_b
    val[4] = tmp[2]
    val[5] = tmp[1]   
    # use cmat = [0 1 0; 
    #             1 0 0]
    # otherwise the jacobian will not agree with this
    return val
end
function Dg(x,vertices)
    # j87y69i
    q_a = SVector{4}(x[13*0 .+ (7:10)])
    q_b = SVector{4}(x[13*1 .+ (7:10)])   #  q_b[1] q_w,  q_b[2] q_v1,  q_b[3] q_v2,  q_b[4] q_v3
    Dgmtx = zeros(5,26)
    Dgmtx[:,13*0 .+ (1:3)] = [-I;zeros(2,3)]  # dg/dra
    Dgmtx[:,13*1 .+ (1:3)]  = [I;zeros(2,3)] # dg/drb
    Dgmtx[:,13*0 .+ (7:10)] = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertices[1]);
                            [q_b[3]  q_b[4] -q_b[1] -q_b[2];
                             q_b[2] -q_b[1] -q_b[4] q_b[3]]
                           ]
    Dgmtx[:,13*1 .+ (7:10)] = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertices[2]);
                            [-q_a[3] -q_a[4]  q_a[1] q_a[2];
                             -q_a[2]  q_a[1]  q_a[4] -q_a[3]]
                           ]
    return Dgmtx
end

# this calculates a part of Dg*attiG, only related to G_qa , dim is 5x3
function Gqa(q_a,q_b,vertices)  
    #  q_b[1] q_w,  q_b[2] q_v1,  q_b[3] q_v2,  q_b[4] q_v3
    Dgmtx = [-2*RS.vmat()*RS.rmult(q_a)'*RS.rmult(RS.hmat()*vertices[1]);
                            [q_b[3]  q_b[4] -q_b[1] -q_b[2];
                             q_b[2] -q_b[1] -q_b[4] q_b[3]]
                           ]
    return Dgmtx*RS.lmult(q_a)*RS.hmat()
end

# this calculates a part of Dg*attiG, only related to G_qb, dim is 5x3
function Gqb(q_a,q_b,vertices)  
    #  q_b[1] q_w,  q_b[2] q_v1,  q_b[3] q_v2,  q_b[4] q_v3
    Dgmtx = [2*RS.vmat()*RS.rmult(q_b)'*RS.rmult(RS.hmat()*vertices[2]);
                            [-q_a[3] -q_a[4]  q_a[1] q_a[2];
                             -q_a[2]  q_a[1]  q_a[4] -q_a[3]]
                           ]
    return Dgmtx*RS.lmult(q_b)*RS.hmat()
end

gval = g(x0,vertices)
Dgmtx = Dg(x0,vertices)

# this is called state_diff_jacobian in Altro
function state_diff_attiG(x)
    q_a0 = SVector{4}(x[13*0 .+ (7:10)])
    q_b0 = SVector{4}(x[13*1 .+ (7:10)])
    sdJ0 = zeros(26,24)
    sdJ0[13*0 .+ (1:3), 12*0 .+ (1:3)] = I(3)
    sdJ0[13*0 .+ (4:6), 12*0 .+ (4:6)] = I(3)
    sdJ0[13*0 .+ (7:10), 12*0 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_a0))
    sdJ0[13*0 .+ (11:13), 12*0 .+ (10:12)] = I(3)
    sdJ0[13*1 .+ (1:3), 12*1 .+ (1:3)] = I(3)
    sdJ0[13*1 .+ (4:6), 12*1 .+ (4:6)] = I(3)
    sdJ0[13*1 .+ (7:10), 12*1 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_b0))
    sdJ0[13*1 .+ (11:13), 12*1 .+ (10:12)] = I(3)
    return sdJ0
end

# q_a1 = SVector{4}(x1[13*0 .+ (7:10)])
# q_b1 = SVector{4}(x1[13*1 .+ (7:10)])
# sdJ1 = zeros(26,24)
# sdJ1[13*0 .+ (1:3), 12*0 .+ (1:3)] = I(3)
# sdJ1[13*0 .+ (4:6), 12*0 .+ (4:6)] = I(3)
# sdJ1[13*0 .+ (7:10), 12*0 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_a1))
# sdJ1[13*0 .+ (11:13), 12*0 .+ (10:12)] = I(3)
# sdJ1[13*1 .+ (1:3), 12*1 .+ (1:3)] = I(3)
# sdJ1[13*1 .+ (4:6), 12*1 .+ (4:6)] = I(3)
# sdJ1[13*1 .+ (7:10), 12*1 .+ (7:9)] = RS.∇differential(UnitQuaternion(q_b1))
# sdJ1[13*1 .+ (11:13), 12*1 .+ (10:12)] = I(3)

# g(x1) = g(x0) + G(x0)*(x1-x0)
# so G(x0)*(x1-x0) =  0
Dgmtx*state_diff_attiG(x0)*state_error   # this is not very close to 0
Dgmtx*state_diff_attiG(x1)*state_error   # this is not very close to 0

# compare with forward diff, the same!
gaug(z) = g(z,vertices)
Dgforward = ForwardDiff.jacobian(gaug,x0)
Dgforward*state_diff_attiG(x0)*state_error

#why this happens? further dig into rotation error
xxx = RS.rotation_error(qdp[1],qd[1], RS.CayleyMap())
qdc  = RS.lmult(qd[1])*1/sqrt(1+norm(xxx)^2)*[1;xxx]
UnitQuaternion(qdc) ≈ qdp[1]
# function q = q
kk = RS.∇differential(UnitQuaternion(q_a0))'*RS.∇differential(UnitQuaternion(q_a0))*xxx
UnitQuaternion(RS.lmult(qd[1])*1/sqrt(1+norm(kk)^2)*[1;kk]) ≈ qdp[1]

"""test A B C jaocbians"""

# xt1 26, xt 26, 
function fdyn(xt1, xt, ut, λt, Δt, ma, mb, Ja,Jb,vertices)
    g = 0.0
    fdyn_vec = zeros(eltype(xt1),26)

    rat1 = xt1[13*0 .+ (1:3)]
    vat1 = xt1[13*0 .+ (4:6)]
    qat1 = SVector{4}(xt1[13*0 .+ (7:10)])
    wat1 = xt1[13*0 .+ (11:13)]


    rbt1 = xt1[13*1 .+ (1:3)]
    vbt1 = xt1[13*1 .+ (4:6)]
    qbt1 = SVector{4}(xt1[13*1 .+ (7:10)])
    wbt1 = xt1[13*1 .+ (11:13)]

    rat = xt[13*0 .+ (1:3)]
    vat = xt[13*0 .+ (4:6)]
    qat = SVector{4}(xt[13*0 .+ (7:10)])
    wat = xt[13*0 .+ (11:13)]

    rbt = xt[13*1 .+ (1:3)]
    vbt = xt[13*1 .+ (4:6)]
    qbt = SVector{4}(xt[13*1 .+ (7:10)])
    wbt = xt[13*1 .+ (11:13)]

    Ft = ut[1:3]
    taut = ut[4:6]
    tau_joint = ut[7]

    fdyn_vec[1:3] = rat1 - (rat + vat*Δt)
    fdyn_vec[4:6] = rbt1 - (rbt + vbt*Δt)
    Ma = diagm([ma,ma,ma])
    Mb = diagm([mb,mb,mb])
    # eqn 3 4 in my notes, notice gravity direction,    Gra's express
    aa = Ma*(vat1-vat)/Δt + Ma*[0;0;g]
    fdyn_vec[7:9] =  aa - Ft - [-I(3);zeros(2,3)]'*λt   # Gra'λ
    bb = Mb*(vbt1-vbt)/Δt + Mb*[0;0;g]
    fdyn_vec[10:12] = bb - [I(3);zeros(2,3)]'*λt # Grb'λ
    
    # println(wat'*wat)
    # println(xt[13*1 .+ (11:13)])
    # println(wbt'*wbt)
    # eqn 5 6 , become harder
    fdyn_vec[13:16] = qat1 - Δt/2*RS.lmult(qat)*SVector{4}([sqrt(4/Δt^2 -wat'*wat);wat])
    fdyn_vec[17:20] = qbt1 - Δt/2*RS.lmult(qbt)*SVector{4}([sqrt(4/Δt^2 -wbt'*wbt);wbt])

    # eqn 7 8
    # println(λt)
    # println(tau_joint)
    Gqamtx = Gqa(qat1,qbt1,vertices) 
    Gqbmtx = Gqb(qat1,qbt1,vertices) 
    # println(4/Δt^2)
    # println(wat1'*wat1)
    # println(wat1)
    a = Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + cross(wat1, (Ja * wat1)) - Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + cross(wat,(Ja * wat))
    k = - 2*taut  - Gqamtx'*λt
    # println(a)

    # println(k)
    # println(a+k)
    fdyn_vec[21:23] = a+k

    b = Jb * wbt1 * sqrt(4/Δt^2 -wbt1'*wbt1) + cross(wbt1, (Jb * wbt1)) - Jb * wbt  * sqrt(4/Δt^2 - wbt'*wbt) + cross(wbt, (Jb * wbt))
    kp = - 2*[0;0;tau_joint] - Gqbmtx'*λt
    fdyn_vec[24:26] = b+kp

                    
    return fdyn_vec
end

# the jacobian of the function fdyn
function Dfdyn(xt1, xt, ut, λt, Δt, ma, mb, Ja,Jb,vertices)
    Dfmtx = spzeros(26, 64)
    qat1 = SVector{4}(xt1[(13*0).+ (7:10)])
    wat1 = view(xt1,(13*0).+ (11:13))
    qbt1 = SVector{4}(xt1[(13*1).+ (7:10)])
    wbt1 = view(xt1,(13*1).+ (11:13))
    qat = SVector{4}(xt[(13*0).+ (7:10)])
    wat = view(xt,(13*0).+ (11:13))
    qbt = SVector{4}(xt[(13*1).+ (7:10)])
    wbt = view(xt,(13*1).+ (11:13))

    Ma = diagm([ma,ma,ma])
    Mb = diagm([mb,mb,mb])
    Gqamtx = Gqa(qat1,qbt1,vertices) 
    Gqbmtx = Gqb(qat1,qbt1,vertices) 
    dt = Δt
    # derivative of eqn 1
    Dfmtx[1:3, (13*0).+(1:3)] = I(3)                           # 3x3 
    Dfmtx[1:3, (26 + 13*0).+(1:3)] = -I(3)                     # 3x3
    Dfmtx[1:3, (26 + 13*0).+(4:6)] = -I(3)*Δt                  # 3x3
    # derivative of eqn 2
    Dfmtx[4:6, (13*1).+(1:3)] = I(3)                           # 3x3
    Dfmtx[4:6, (26 + 13*1).+(1:3)] = -I(3)                     # 3x3
    Dfmtx[4:6, (26 + 13*1).+(4:6)] = -I(3)*Δt                  # 3x3
    # derivative of eqn 3 
    Dfmtx[7:9, (13*0).+(4:6)] = Ma/Δt                             # 3x3
    Dfmtx[7:9, (26 + 13*0).+(4:6)] = -Ma/Δt                       # 3x3
    Dfmtx[7:9, (26 + 26).+(1:3)] = -I(3)                    # 3x3
    Dfmtx[7:9, (26 + 26 + 7).+(1:5)] = - [-I;zeros(2,3)]'   # 3x3
    # derivative of eqn 4 
    Dfmtx[10:12, (13*1).+(4:6)] = Mb/Δt                           # 3x3
    Dfmtx[10:12, (26 + 13*1).+(4:6)] = -Mb/Δt                     # 3x3
    Dfmtx[10:12, (26 + 26 + 7).+(1:5)] = - [I;zeros(2,3)]'  # 3x3
    # derivative of eqn 5   
    Dfmtx[13:16, (13*0).+(7:10)] = I(4)                        # 4x4
    Dfmtx[13:16, (26 + 13*0).+(7:10)] = -Δt/2*RS.rmult(SVector{4}([sqrt(4/Δt^2 -wat'*wat);wat]))   # 4x4
    Dfmtx[13:16, (26 + 13*0).+(11:13)] = -Δt/2*(-qat*wat'/sqrt(4/Δt^2 -wat'*wat) + RS.lmult(qat)*RS.hmat())   # 4x3
    # derivative of eqn 6   
    Dfmtx[17:20, (13*1).+(7:10)] = I(4)                        # 4x4
    Dfmtx[17:20, (26 + 13*1).+(7:10)] = -Δt/2*RS.rmult(SVector{4}([sqrt(4/Δt^2 -wbt'*wbt);wbt]))   # 4x4
    Dfmtx[17:20, (26 + 13*1).+(11:13)] = -Δt/2*(-qbt*wbt'/sqrt(4/Δt^2 -wbt'*wbt) + RS.lmult(qbt)*RS.hmat())   # 4x3
    # derivative of eqn 7 (very challenging)
    # d (Ja * wat1 * sqrt(4/Δt^2 -wat1'*wat1) + wat1 × (Ja * wat1)) / d wat1
    J11 = Ja[1,1];J12 = Ja[1,2];J13 = Ja[1,3];
    J21 = Ja[2,1];J22 = Ja[2,2];J23 = Ja[2,3];
    J31 = Ja[3,1];J32 = Ja[3,2];J33 = Ja[3,3];
    w1 = wat1[1]; w2 = wat1[2]; w3 = wat1[3];
    row1 = [                    J31*w2 - J21*w3 + J11*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J12*w3 - J32*w1 + J22*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J23*w1 - J13*w2 + J33*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    Dfmtx[21:23, (13*0).+(11:13)] = [row1 row2 row3]'


    # d G_qa1t'*λt /dqa1t  3x4. copy matlab code, then replace matlab symbol with Julia variables
    la1 = λt[1];la2 = λt[2];la3 = λt[3];la4 = λt[4];la5 = λt[5];
    qtaw = qat1[1]; qtav1 = qat1[2]; qtav2 = qat1[3]; qtav3 = qat1[4];
    qtbw = qbt1[1]; qtbv1 = qbt1[2]; qtbv2 = qbt1[3]; qtbv3 = qbt1[4];
    pa1 = vertices[1][1]; pa2 = vertices[1][2]; pa3 = vertices[1][3]
    
    row1 = [ la4*qtbv3 - la5*qtbw + la2*(4*pa2*qtav1 + 4*pa3*qtaw) - la3*(4*pa2*qtaw - 4*pa3*qtav1) - la1*(4*pa2*qtav2 + 4*pa3*qtav3),  
             la2*(4*pa2*qtaw - 4*pa3*qtav1) - la5*qtbv1 - la4*qtbv2 + la3*(4*pa2*qtav1 + 4*pa3*qtaw) - la1*(4*pa2*qtav3 - 4*pa3*qtav2), 
             la4*qtbv1 - la5*qtbv2 - la1*(4*pa2*qtaw - 4*pa3*qtav1) - la2*(4*pa2*qtav3 - 4*pa3*qtav2) + la3*(4*pa2*qtav2 + 4*pa3*qtav3), 
           - la4*qtbw - la5*qtbv3 - la1*(4*pa2*qtav1 + 4*pa3*qtaw) - la2*(4*pa2*qtav2 + 4*pa3*qtav3) - la3*(4*pa2*qtav3 - 4*pa3*qtav2)]


    row2 = [ la1*(4*pa1*qtav2 - 4*pa3*qtaw) - la5*qtbv3 - la4*qtbw - la2*(4*pa1*qtav1 + 4*pa3*qtav3) + la3*(4*pa1*qtaw + 4*pa3*qtav2), 
             la5*qtbv2 - la4*qtbv1 + la1*(4*pa1*qtav3 - 4*pa3*qtav1) - la2*(4*pa1*qtaw + 4*pa3*qtav2) - la3*(4*pa1*qtav1 + 4*pa3*qtav3),  
             la1*(4*pa1*qtaw + 4*pa3*qtav2) - la5*qtbv1 - la4*qtbv2 + la2*(4*pa1*qtav3 - 4*pa3*qtav1) - la3*(4*pa1*qtav2 - 4*pa3*qtaw),   
             la5*qtbw - la4*qtbv3 + la1*(4*pa1*qtav1 + 4*pa3*qtav3) + la2*(4*pa1*qtav2 - 4*pa3*qtaw) + la3*(4*pa1*qtav3 - 4*pa3*qtav1)]

    row3 = [la5*qtbv2 - la4*qtbv1 + la1*(4*pa2*qtaw + 4*pa1*qtav3) - la2*(4*pa1*qtaw - 4*pa2*qtav3) - la3*(4*pa1*qtav1 + 4*pa2*qtav2),  
            la4*qtbw + la5*qtbv3 - la1*(4*pa1*qtav2 - 4*pa2*qtav1) + la2*(4*pa1*qtav1 + 4*pa2*qtav2) - la3*(4*pa1*qtaw - 4*pa2*qtav3),  
            la4*qtbv3 - la5*qtbw - la1*(4*pa1*qtav1 + 4*pa2*qtav2) - la2*(4*pa1*qtav2 - 4*pa2*qtav1) - la3*(4*pa2*qtaw + 4*pa1*qtav3),   
            la1*(4*pa1*qtaw - 4*pa2*qtav3) - la5*qtbv1 - la4*qtbv2 + la2*(4*pa2*qtaw + 4*pa1*qtav3) - la3*(4*pa1*qtav2 - 4*pa2*qtav1)]
    Dfmtx[21:23, (13*0).+(7:10)] = -[row1 row2 row3]'  # in the eqn 7 we have -G_qa1t'*λt

    # d (- Ja * wat  * sqrt(4/Δt^2 - wat'*wat) + wat  × (Ja * wat)) / dwat
    w1 = wat[1]; w2 = wat[2]; w3 = wat[3];
    row1 = [                    J31*w2 - J21*w3 - J11*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J12*w3 - J32*w1 - J22*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J23*w1 - J13*w2 - J33*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    Dfmtx[21:23, (26 + 13*0).+(11:13)] = [row1 row2 row3]'

    # d G_qa1t'*λt /dqb1t  3x4.
    row1 = [- la5*qtaw - la4*qtav3,   la4*qtav2 - la5*qtav1, - la4*qtav1 - la5*qtav2,   la4*qtaw - la5*qtav3]
    row2 = [  la5*qtav3 - la4*qtaw, - la4*qtav1 - la5*qtav2,   la5*qtav1 - la4*qtav2, - la5*qtaw - la4*qtav3]
    row3 = [ la4*qtav1 - la5*qtav2,  - la4*qtaw - la5*qtav3,    la5*qtaw - la4*qtav3,  la4*qtav2 + la5*qtav1]

    Dfmtx[21:23, (13*1).+(7:10)] =  -[row1 row2 row3]'  # in the eqn 7 we have -G_qa1t'*λt

    Dfmtx[21:23, (26 + 26).+(4:6)] =  -2*I(3)
    # Dfmtx[21:23, (26 + 26).+(7)] =  [0;0;2]
    Dfmtx[21:23, (26 + 26 + 7).+(1:5)] = -Gqamtx' 
    # derivative of eqn 8 (very challenging)
    # d (Jb * wbt1 * sqrt(4/Δt^2 -wbt1'*wbt1) + wbt1 × (Jb * wbt1)) / d wbt1
    J11 = Jb[1,1];J12 = Jb[1,2];J13 = Jb[1,3];
    J21 = Jb[2,1];J22 = Jb[2,2];J23 = Jb[2,3];
    J31 = Jb[3,1];J32 = Jb[3,2];J33 = Jb[3,3];
    w1 = wbt1[1]; w2 = wbt1[2]; w3 = wbt1[3];
    row1 = [                    J31*w2 - J21*w3 + J11*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 + J12*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 + J13*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 + J21*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J12*w3 - J32*w1 + J22*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 + J23*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 + J31*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w1*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 + J32*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w2*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J23*w1 - J13*w2 + J33*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) - (w3*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    Dfmtx[24:26, (13*1).+(11:13)] = [row1 row2 row3]'

    # d G_qb1t'*λt /dqa1t  3x4.
    pb1 = vertices[2][1]; pb2 = vertices[2][2]; pb3 = vertices[2][3]
    row1 = [ la5*qtbw + la4*qtbv3, la5*qtbv1 - la4*qtbv2, la4*qtbv1 + la5*qtbv2,    la5*qtbv3 - la4*qtbw]
    row2 = [ la4*qtbw - la5*qtbv3, la4*qtbv1 + la5*qtbv2, la4*qtbv2 - la5*qtbv1,    la5*qtbw + la4*qtbv3]
    row3 = [la5*qtbv2 - la4*qtbv1,  la4*qtbw + la5*qtbv3,  la4*qtbv3 - la5*qtbw, - la4*qtbv2 - la5*qtbv1]

    Dfmtx[24:26, (13*0).+(7:10)] =  -[row1 row2 row3]'  # in the eqn 8 we have -G_qb1t'*λt
    # d G_qb1t'*λt /dqb1t  3x4.
    row1 = [ la5*qtaw - la4*qtav3 - la2*(4*pb2*qtbv1 + 4*pb3*qtbw) + la3*(4*pb2*qtbw - 4*pb3*qtbv1) + la1*(4*pb2*qtbv2 + 4*pb3*qtbv3),  la4*qtav2 + la5*qtav1 - la2*(4*pb2*qtbw - 4*pb3*qtbv1) - la3*(4*pb2*qtbv1 + 4*pb3*qtbw) + la1*(4*pb2*qtbv3 - 4*pb3*qtbv2), la5*qtav2 - la4*qtav1 + la1*(4*pb2*qtbw - 4*pb3*qtbv1) + la2*(4*pb2*qtbv3 - 4*pb3*qtbv2) - la3*(4*pb2*qtbv2 + 4*pb3*qtbv3), la4*qtaw + la5*qtav3 + la1*(4*pb2*qtbv1 + 4*pb3*qtbw) + la2*(4*pb2*qtbv2 + 4*pb3*qtbv3) + la3*(4*pb2*qtbv3 - 4*pb3*qtbv2)]
    row2 = [ la4*qtaw + la5*qtav3 - la1*(4*pb1*qtbv2 - 4*pb3*qtbw) + la2*(4*pb1*qtbv1 + 4*pb3*qtbv3) - la3*(4*pb1*qtbw + 4*pb3*qtbv2), la4*qtav1 - la5*qtav2 - la1*(4*pb1*qtbv3 - 4*pb3*qtbv1) + la2*(4*pb1*qtbw + 4*pb3*qtbv2) + la3*(4*pb1*qtbv1 + 4*pb3*qtbv3),  la4*qtav2 + la5*qtav1 - la1*(4*pb1*qtbw + 4*pb3*qtbv2) - la2*(4*pb1*qtbv3 - 4*pb3*qtbv1) + la3*(4*pb1*qtbv2 - 4*pb3*qtbw), la4*qtav3 - la5*qtaw - la1*(4*pb1*qtbv1 + 4*pb3*qtbv3) - la2*(4*pb1*qtbv2 - 4*pb3*qtbw) - la3*(4*pb1*qtbv3 - 4*pb3*qtbv1)]
    row3 = [la4*qtav1 - la5*qtav2 - la1*(4*pb2*qtbw + 4*pb1*qtbv3) + la2*(4*pb1*qtbw - 4*pb2*qtbv3) + la3*(4*pb1*qtbv1 + 4*pb2*qtbv2),  la1*(4*pb1*qtbv2 - 4*pb2*qtbv1) - la5*qtav3 - la4*qtaw - la2*(4*pb1*qtbv1 + 4*pb2*qtbv2) + la3*(4*pb1*qtbw - 4*pb2*qtbv3),  la5*qtaw - la4*qtav3 + la1*(4*pb1*qtbv1 + 4*pb2*qtbv2) + la2*(4*pb1*qtbv2 - 4*pb2*qtbv1) + la3*(4*pb2*qtbw + 4*pb1*qtbv3), la4*qtav2 + la5*qtav1 - la1*(4*pb1*qtbw - 4*pb2*qtbv3) - la2*(4*pb2*qtbw + 4*pb1*qtbv3) + la3*(4*pb1*qtbv2 - 4*pb2*qtbv1)]
    Dfmtx[24:26, (13*1).+(7:10)] =  -[row1 row2 row3]'  # in the eqn 8 we have -G_qb1t'*λt

    # d (- Jb * wbt  * sqrt(4/Δt^2 - wbt'*wbt) + wbt  × (Jb * wbt)) / dwbt
    w1 = wbt[1]; w2 = wbt[2]; w3 = wbt[3];
    row1 = [                    J31*w2 - J21*w3 - J11*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J31*w1 - J22*w3 + 2*J32*w2 + J33*w3 - J12*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J33*w2 - J22*w2 - 2*J23*w3 - J21*w1 - J13*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J11*w1 + J12*w2 + J13*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row2 = [J11*w3 - 2*J31*w1 - J32*w2 - J33*w3 - J21*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J12*w3 - J32*w1 - J22*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J11*w1 + J12*w2 + 2*J13*w3 - J33*w1 - J23*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J21*w1 + J22*w2 + J23*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    row3 = [2*J21*w1 - J11*w2 + J22*w2 + J23*w3 - J31*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w1*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2), J22*w1 - 2*J12*w2 - J13*w3 - J11*w1 - J32*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w2*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2),                     J23*w1 - J13*w2 - J33*(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2) + (w3*(J31*w1 + J32*w2 + J33*w3))/(4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)]
    Dfmtx[24:26, (26 + 13*1).+(11:13)] = [row1 row2 row3]'

    Dfmtx[24:26, (26 + 26).+(7)] =  [0;0;-2]
    Dfmtx[24:26, (26 + 26 + 7).+(1:5)] = -Gqbmtx'
    
    return Dfmtx
end

# attitude Jacobian So that Dfmtx * attiG_mtx ==> size(26,60) where 60 is the size of error state 
# this is really cumbersome to manually construct. something like state_diff_jacobian in Altro is definitely better
function attiG_f(xt1, xt)
    attiG_mtx = spzeros(64, 60)
    qat1 = SVector{4}(xt1[13*0 .+ (7:10)])
    qbt1 = SVector{4}(xt1[13*1 .+ (7:10)])
    qat = SVector{4}(xt[13*0 .+ (7:10)])
    qbt = SVector{4}(xt[13*1 .+ (7:10)])
    attiG_mtx[1:6,1:6] = I(6)
    attiG_mtx[7:10,7:9] = RS.lmult(qat1)*RS.hmat()
    attiG_mtx[11:13,10:12] = I(3)
    attiG_mtx[(13*1).+(1:6), (12*1).+(1:6)] = I(6)
    attiG_mtx[(13*1).+(7:10), (12*1).+(7:9)] = RS.lmult(qbt1)*RS.hmat()
    attiG_mtx[(13*1).+(11:13), (12*1).+(10:12)] = I(3)

    attiG_mtx[(26).+(1:6), (24).+(1:6)] = I(6)
    attiG_mtx[(26).+(7:10), (24).+(7:9)] = RS.lmult(qat)*RS.hmat()
    attiG_mtx[(26).+(11:13), (24).+(10:12)] = I(3)
    attiG_mtx[(26 + 13*1).+(1:6), (24 + 12*1).+(1:6)] = I(6)
    attiG_mtx[(26 + 13*1).+(7:10), (24 + 12*1).+(7:9)] = RS.lmult(qbt)*RS.hmat()
    attiG_mtx[(26 + 13*1).+(11:13), (24 + 12*1).+(10:12)] = I(3)

    attiG_mtx[(26 + 26).+(1:12), (24 + 24).+(1:12)] = I(12)
    return attiG_mtx
end

# basic test of fdyn 
begin

    x0 = generate_config_with_rand_vel(mech, [2.0;2.0;1.0;pi/4], [pi/4]);
    dr = pi/14
    x1 = generate_config_with_rand_vel(mech, [2.0;2.0;1.0;pi/4+dr], [pi/4+dr]);
    u = 2*randn(7)
    du = 0.01*randn(7)
    λ = randn(5)
    dλ = 0.001*randn(5)
    dxv = zeros(26)
    dxv[(13*0).+(4:6)] = randn(3)
    dxv[(13*0).+(11:13)] = randn(3)
    dxv[(13*1).+(4:6)] = randn(3)
    dxv[(13*1).+(11:13)] = randn(3)
    f1 = fdyn(x1, x0, u, λ, 0.1, link0.m, link1.m, Inerita_a,Inerita_a,vertices)
    f2 = fdyn(x1+dxv, x0+dxv, u+du, λ+dλ, 0.1, link0.m, link1.m, Inerita_a,Inerita_a,vertices)
    
    # basic test of Dfyn*attiG 
    # basic test of Dfyn*attiG 
    # basic test of Dfyn*attiG 
    Dfmtx = Dfdyn(x1, x0, u, λ, 0.1, link0.m, link1.m, Inerita_a,Inerita_a,vertices)
    attiG_mtx = attiG_f(x1,x0)
    
    state_diff = zeros(60)
    state_diff[(12*0).+(4:6)] = dxv[(13*0).+(4:6)]
    state_diff[(12*0).+(10:12)] = dxv[(13*0).+(11:13)]
    state_diff[(12*1).+(4:6)] = dxv[(13*1).+(4:6)]
    state_diff[(12*1).+(10:12)] = dxv[(13*1).+(11:13)]
    state_diff[(24+12*0).+(4:6)] = dxv[(13*0).+(4:6)]
    state_diff[(24+12*0).+(10:12)] = dxv[(13*0).+(11:13)]
    state_diff[(24+12*1).+(4:6)] = dxv[(13*1).+(4:6)]
    state_diff[(24+12*1).+(10:12)] = dxv[(13*1).+(11:13)]

    state_diff[(24+24).+(1:7)] = du
    state_diff[(24+24+7).+(1:5)] = dλ
    f2 - (f1 + Dfmtx*attiG_mtx*state_diff)   

    # compare with Forward diff
    faug(z) = fdyn(z[1:26], z[27:52], z[53:59], z[60:64], 0.1, link0.m, link1.m, Inerita_a,Inerita_a,vertices)
    Df2 = ForwardDiff.jacobian(faug,[x1;x0;u;λ])

    f2 - (f1 + Df2*attiG_mtx*state_diff)
    Dfmine = Dfmtx*attiG_mtx
    Dfdiff = Df2*attiG_mtx
    @test Dfmtx*attiG_mtx ≈ Df2*attiG_mtx
end


# x is the current state, x⁺ is the next state
# given current state x and current U
# use newton's method to solve for the vel part of x and the next state x⁺
# u has dimension 7, λ has dimension 5
# should write a struct to describe these variables 
# this function modifies x!
function discrete_dynamics!(x, u, λ_init, dt)
    λ = zeros(eltype(x),5)
    # λ = λ_init
    x⁺ = Vector(x)
    x_iter = copy(x)
    x⁺_new, x_new, λ_new = copy(x⁺), copy(x), copy(λ)

    max_iters, line_iters, ϵ = 500, 80, 1e-3
    for i=1:max_iters  
        # print("iter ", i, ": ")

        # Newton step    
        # 31 = 26 + 5
        err_vec = [fdyn(x⁺, x_iter, u, λ, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices);
                   g(x⁺,vertices)]

        err = norm(err_vec)
        # println(" err_vec: ", err)
        # jacobian of x+ and λ
        G = Dg(x⁺,vertices)*state_diff_attiG(x⁺)
        # faug(z) = fdyn(z[1:26], z[27:52], z[53:59], z[60:64], dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices)
        # Df2 = ForwardDiff.jacobian(faug,[x⁺;x_iter;u;λ])
        # Fdyn = Df2*attiG_f(x⁺, x_iter)
        Fdyn = Dfdyn(x⁺, x_iter, u, λ, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices)*attiG_f(x⁺, x_iter)



        # 31 x 29  (24+5) # x⁺ , lambda
        F = [Fdyn[:,1:24] Fdyn[:,48+7+1:48+7+5];
                       G   spzeros(5,5)]
        Δs = -F\err_vec  #29x1
       
        # backtracking line search
        j=0
        α = 1
        ρ = 0.5
        c = 0.01 

        err_new = err + 9999
        while (err_new > err + c*α*(err_vec/err)'*F*Δs) && (j < line_iters)
            # println("*****")
            Δλ = α*Δs[(24) .+ (1:5)]
            # println(Δs')
            λ_new .= λ + Δλ

            Δx⁺ = α*Δs[1:24] # order according to fdyn: 
            # calculate x⁺_new = x⁺ + Δx⁺
            x⁺_new[13*0 .+ (1:3)] = x⁺[13*0 .+ (1:3)] + Δx⁺[12*0 .+ (1:3)]
            x⁺_new[13*0 .+ (4:6)] = x⁺[13*0 .+ (4:6)] + Δx⁺[12*0 .+ (4:6)]
            phi = Δx⁺[12*0 .+ (7:9)]
            x⁺_new[13*0 .+ (7:10)] = RS.lmult(SVector{4}(x⁺[13*0 .+ (7:10)]))*[1;phi]/(sqrt(1+norm(phi)^2))
            x⁺_new[13*0 .+ (11:13)] = x⁺[13*0 .+ (11:13)] + Δx⁺[12*0 .+ (10:12)]

            x⁺_new[13*1 .+ (1:3)] = x⁺[13*1 .+ (1:3)] + Δx⁺[12*1 .+ (1:3)]
            x⁺_new[13*1 .+ (4:6)] = x⁺[13*1 .+ (4:6)] + Δx⁺[12*1 .+ (4:6)]
            phi = Δx⁺[12*1 .+ (7:9)]
            x⁺_new[13*1 .+ (7:10)] = RS.lmult(SVector{4}(x⁺[13*1 .+ (7:10)]))*[1;phi]/(sqrt(1+norm(phi)^2))
            x⁺_new[13*1 .+ (11:13)] = x⁺[13*1 .+ (11:13)] + Δx⁺[12*1 .+ (10:12)]

            # # update the velocity part of x 
            # Δx = Δs[(24) .+ (1:12)]   # only velocity part
            # x_new[13*0 .+ (4:6)] = x_iter[13*0 .+ (4:6)] + Δx[1:3] # qa linear vel
            # x_new[13*0 .+ (11:13)] = x_iter[13*0 .+ (11:13)] + Δx[4:6] # qa angular vel
            # x_new[13*1 .+ (4:6)] = x_iter[13*1 .+ (4:6)] + Δx[7:9] # qb linear vel
            # x_new[13*1 .+ (11:13)] = x_iter[13*1 .+ (11:13)] + Δx[10:12] # qb angular vel

            ωa⁺ = x⁺_new[13*0 .+ (11:13)]
            ωb⁺ = x⁺_new[13*1 .+ (11:13)]
            # ωa = x_new[13*0 .+ (11:13)]
            # ωb = x_new[13*1 .+ (11:13)]
            # ωs⁺ = [ωa⁺;ωb⁺;ωa;ωb]
            
            if (4/dt^2 >= dot(ωa⁺,ωa⁺)) && (4/dt^2 >= dot(ωb⁺,ωb⁺)) 
                err_vec = [fdyn(x⁺_new, x_iter, u, λ_new, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices);
                            g(x⁺_new,vertices)]
                err_new = norm(err_vec)
                # println(" fdyn: ", norm(fdyn(x⁺_new, x_iter, u, λ_new, dt, 1, 1, diagm([1,1,1]),diagm([1,1,1]),vertices)))
                # println(" g(x⁺_new,vertices): ", g(x⁺_new,vertices))
            end
            α = α*ρ
            j += 1
        end
        # println(" steps: ", j)
        # println(" err_new: ", err_new)
        x⁺ .= x⁺_new
        # x_iter .= x_new
        λ .= λ_new

        # convergence check
        if err_new < ϵ
            x .= x_iter
            # println(round.(fdyn(x⁺, x, u, λ, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices),digits=6)')
            return x⁺, λ
        end
    end 
    x .= x_iter
    # println(round.(fdyn(x⁺, x, u, λ, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices),digits=6)')
    return x⁺, λ   
end


# rigorous test, need to do systme simulation 
# start from x0, simulate forward 
U = [0.0; 0.0; 0.0;
     1.0; 1.0; 0.0;
     1.0]
# U = 0.01*rand(7)
λ_init = zeros(5)
x0 = generate_config(mech, [2.0;2.0;1.0;pi/2], [pi/2]);
x1, λ1 = discrete_dynamics!(x0, U, λ_init, 0.001)
x2, λ2 = discrete_dynamics!(x1, U, λ1, 0.001)
round.(fdyn(x2, x1, U, λ2, 0.01, link0.m, link1.m, Inerita_a,Inerita_a,vertices),digits=6)'


# simulate for 3 seconds, 
# then visualize it using Jan's code
Tf =0.5
dt = 0.005
N = Int(Tf/dt)

x0 = generate_config(mech, [0.1;0.1;1.0;0.0001], [0.001]);
x = x0
λ = zeros(5)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx = 1:N
    println("step: ",idx)
    x1, λ1 = discrete_dynamics!(x, U,λ, dt)
    println(norm(fdyn(x1, x, U, λ1, dt, link0.m, link1.m, Inerita_a,Inerita_a,vertices)))
    println(norm(g(x1,vertices)))
    setStates!(mech,x1)
    for i=1:2
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
    x = x1
    λ = λ1
end
visualize(mech,storage, env = "editor")



# """ Define flotation force through controller """
# baseid = eqcs[1].id # this is the floating base, as a free joint
# mech.g = 0 # disable gravity
# function controller!(mechanism, k)
#     # F = SA[0;0; 9.81 * 10.4]
#     F = SA[0;0; 0]
#     τ = SA[0;0;0.0]
#     setForce!(mechanism, geteqconstraint(mechanism,baseid), [F;τ])
#     return
# end
# """ Start simulation """
# # storage = simulate!(mech, timeStep, record = true)
# # visualize(mech, storage, env = "editor")


