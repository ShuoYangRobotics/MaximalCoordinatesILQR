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

# Define quaternions using Rotations.jl
q1 = RS.UnitQuaternion(RotY(pi/2))
q2 = RS.UnitQuaternion(RotY(pi/5))
# Get standard vectors representation 
q1_vec = RS.params(q1)
q2_vec = RS.params(q2)
# test L(q) and R(q) 
@test RS.rmult(q1) ≈ Rmat(q1_vec)
@test RS.lmult(q1) ≈ Lmat(q1_vec)
# test multiplication results
@test Lmat(q1_vec)*q2_vec ≈ RS.params(q2 * q1)
@test Rmat(q2_vec)*q1_vec ≈ RS.params(q2 * q1)

# General vector/point A
PA = [0;0;2]
# Rotate π/2 along Y axis
PB = H'*Lmat(q1_vec)*Rmat(q1_vec)'*H*PA
@test PB ≈ q1*PA

# a random quaternion
q = rand(UnitQuaternion)
q_vec = RS.params(q)
# G(q)
@test RS.∇differential(q) ≈ RS.lmult(q)*H