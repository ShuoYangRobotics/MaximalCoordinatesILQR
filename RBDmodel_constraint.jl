"""Constants"""
struct EFVConstraint{S,T} <: TO.StageConstraint
	n::Int
	m::Int
	model::FloatingSpaceRBD{T}
    maxV::Float64
	sense::S
	function EFVConstraint(n::Int, m::Int, model::FloatingSpaceRBD{T}, maxV::Float64,
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
function world_vel(model::FloatingSpaceRBD{T}, z::AbstractVector{P}) where {T, P}
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

# @testset "Constraints" begin
#     lin_upper = EFVConstraint(n,m,RBDmodel,8.0, TO.Inequality())

#     base_x = [0.01, 0.01, 0.01]
#     base_q = RS.params(UnitQuaternion(RotZ(0.01)))
#     base_v = [1., 1., 1.]
#     base_ω = [0., 0., 0.]
#     joint_angles = fill.(0.01,RBDmodel.nb)
#     joint_w = fill.(0.0,RBDmodel.nb)
#     x1 = [base_q;base_x;joint_angles;base_ω;base_v;joint_w]
#     z1 = KnotPoint(x1,zeros(6+RBDmodel.nb),dt)
#     @show TO.evaluate(lin_upper, z1)
#     @test min(TO.evaluate(lin_upper, z1)...) < 0

#     base_x = [0.01, 0.01, 0.01]
#     base_q = RS.params(UnitQuaternion(RotZ(0.01)))
#     base_v = [0., 0., 0.]
#     base_ω = [0., 0., 0.]
#     joint_angles = fill.(0.,RBDmodel.nb)
#     joint_w = [0., 0., 3.]
#     x2 = [base_q;base_x;joint_angles;base_ω;base_v;joint_w]
#     z2 = KnotPoint(x2,zeros(6+RBDmodel.nb),dt)
#     @show TO.evaluate(lin_upper, z2)
#     @test max(TO.evaluate(lin_upper, z2)...) < 0

#     ∇c = zeros(6,n+m)
#     TO.jacobian!(∇c, lin_upper, z2)
#     @show 
# end
# # function TO.jacobian!(∇c, con::EFVConstraint, z::AbstractKnotPoint)
# # 	∇c[:,con.inds[1]:con.inds[end]] .= con.A
# # 	return true
# # end

# # function TO.change_dimension(con::EFVConstraint, n::Int, m::Int, ix=1:n, iu=1:m)
# # 	inds0 = [ix; n .+ iu]  # indices of original z in new z
# # 	inds = inds0[con.inds] # indices of elements in new z
# # 	EFVConstraint(n, m, con.A, con.b, con.sense, inds)
# # end
# iq, ix, iθ, iω, iv, iϕ = state_parts(RBDmodel)

# function evlatz(z::AbstractVector{T}) where T
#     thetalist = z[iθ]
#     ϕlist = z[iϕ]
#     MRB.FKinSpace(RBDmodel.M, RBDmodel.Slist, thetalist)[1:3,1:3]*(MRB.JacobianBody(RBDmodel.Blist, thetalist)*ϕlist)[4:6]
# end
# base_x = [0.01, 0.01, 0.01]
# base_q = RS.params(UnitQuaternion(RotZ(0.01)))
# base_v = [5., 5., 5.]
# base_ω = [0., 1., 0.]
# joint_angles = fill.(0.01,RBDmodel.nb)
# joint_w = fill.(5.0,RBDmodel.nb)
# x1 = [base_q;base_x;joint_angles;base_ω;base_v;joint_w]
# evlatz(x1)