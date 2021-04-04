using Rotations
using LinearAlgebra
using Test
using StaticArrays
using ForwardDiff
const RS = Rotations

# first implement quaternion rotations from scratch
# follow Brian's paper "Planning with Attitude"
const T = Diagonal([1; -ones(3)])
const H = [zeros(1,3); I]

"""Returns the cross product matrix """
function cross_mat(v)
    return [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] -v[1] 0]
end 

"""Given quaternion q returns left multiply quaternion matrix L(q)"""
function Lmat(quat)
    L = zeros(4,4)
    s = quat[1]
    v = quat[2:end]
    L[1,1] = s
    L[1,2:end] = -v'
    L[2:end,1] = v
    L[2:end, 2:end] = s*I + cross_mat(v)
    return L
end 

"""Given quaternion q returns right multiply quaternion matrix Rmat(q)"""
function Rmat(quat)
    L = zeros(4,4)
    s = quat[1]
    v = quat[2:end]
    L[1,1] = s
    L[1,2:end] = -v'
    L[2:end,1] = v
    L[2:end, 2:end] = s*I - cross_mat(v)
    return L
end 

# treat quaternions as 4-vectors 
q1 = RS.UnitQuaternion(RotY(pi/4))
q2 = RS.UnitQuaternion(RotY(pi/5))

q1_vec = RS.params(q1)

qvm = RS.QuatVecMap()
q_error = RS.rotation_error(q1,q2,qvm)

# q1 + 0.5*RS.∇differential(q1)*q_error

# RS.∇differential(q1)  == G(q1)*H
# q1 + 0.5*G(q1)*H*w_b


# 
pa = [0;0;1]
pb = H'*Lmat(q1_vec)*Rmat(q1_vec)'*H*pa

@test pb ≈ q1*pa

w = [1;2;3]

@test RS.∇differential(q1) ≈ Lmat(q1_vec)*H


@test RS.rmult(q1) ≈ Rmat(q1_vec)

v1 = @SVector rand(3)
q = rand(UnitQuaternion)
phi = [0.001; 0.005; -0.007]
q_vec = Lmat(RS.params(q))*[1;phi]/(sqrt(1+norm(phi)^2))

# TODO: find out why this test is failing
@test RS.lmult(q) ≈ Lmat(RS.params(q))

v_r1 = q*v1
v_r2 = UnitQuaternion(q_vec)*v1

#∇rotate(q,v1) is partial h partial q in Eqn.14 in "Planning with Attitude"

v_r22 = q*v1 + RS.∇rotate(q,v1)*RS.∇differential(q)*phi
@test v_r22 ≈ v_r2  atol=0.01
@test ForwardDiff.jacobian(q->UnitQuaternion(q,false)*v1, RS.params(q)) ≈ RS.∇rotate(q,v1)