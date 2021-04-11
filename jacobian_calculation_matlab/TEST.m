syms qw qv1 qv2 qv3 real
q = [qw;qv1;qv2;qv3]
Lmat(q)
H = [0 0 0;1 0 0; 0 1 0; 0 0 1]
V = [0 0 0;1 0 0; 0 1 0; 0 0 1]'
syms p1 p2 p3 real
p = [p1;p2;p3]
% test the jacobian of translation joint q*p
g = V*Rmat(q)'*Lmat(q)*H*p
jacobian(g,q)
2*V*Rmat(q)'*Rmat([0;p])
2*V*Rmat(q)'*Rmat([0;p])-jacobian(g,q)

% test the jacobian of rotate joint
syms aw av1 av2 av3 real
qa = [aw;av1;av2;av3]
syms bw bv1 bv2 bv3 real
qb = [bw;bv1;bv2;bv3]
cmat = [0 1 0; 
        1 0 0];
g2 = cmat * V*Lmat(qa)'*qb
jacobian(g2,qa)
jacobian(g2,qb)


syms phi1 phi2 phi3 real

phi = [phi1; phi2; phi3]

q = 1/sqrt(1+norm(phi)^2)*[1;phi]

simplify(jacobian(q,phi)*(phi1^2 + phi2^2 + phi3^2 + 1)^(3/2))
