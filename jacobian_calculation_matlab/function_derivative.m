% f(x_t+1,  x_t, u_t, lambda_t) = 0
% this is very complicated 

H = [0 0 0;eye(3)]
V = H';
cmat = [0 1 0;
        1 0 0];
    

% test the jacobian of rotate joint
syms aw av1 av2 av3 real
qa = [aw;av1;av2;av3]
syms bw bv1 bv2 bv3 real
qb = [bw;bv1;bv2;bv3]
syms w1 w2 w3 dt real
syms vec1 vec2 vec3 vec4 real

w = [w1;w2;w3];
vec = [sqrt(4/dt/dt-w'*w);w]
% vec = [vec1; vec2; vec3; vec4]

f = Lmat(qa)*vec
jacobian(f,qa) % equals to Rmat(vec)
jacobian(f,qa) - Rmat(vec)


fw = jacobian(f,w)
% decompose by inspection
tmp = Lmat(qa)*H
% simplify((fw-tmp)*((4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2)))

fw2 = -qa*w'/((4/dt^2 - w1^2 - w2^2 - w3^2)^(1/2))+Lmat(qa)*H

syms J11 J12 J13 J21 J22 J23 J31 J32 J33 real
J = [J11 J12 J13;
     J21 J22 J23;
     J31 J32 J33];
jw = cross_mat(w)*J*w
jw2 = J*w*sqrt(4/dt/dt-w'*w)+cross_mat(w)*J*w
% 
display('jacobian of jwt1 wrt wt1')
jacobian(jw2, w)

jw2 = -J*w*sqrt(4/dt/dt-w'*w)+cross_mat(w)*J*w 
display('jacobian of jwt wrt wt')
jacobian(jw2, w) 


% derive derivative Eqn 7 8 in my f formulation
syms qtaw qtav1 qtav2 qtav3 real
qta = [qtaw;qtav1;qtav2;qtav3]
syms qtbw qtbv1 qtbv2 qtbv3 real
qtb = [qtbw;qtbv1;qtbv2;qtbv3]
syms pa1 pa2 pa3 pb1 pb2 pb3 real
syms la1 la2 la3 la4 la5    % lambda 
% First define G_qa and G_qb
G_qa = [-2*V*Rmat(qta)'*Rmat([0;pa1;pa2;pa3]);
        -cmat*V*Lmat(qtb)']*Lmat(qa)*H
    
G_qala = G_qa'*[la1;la2;la3;la4;la5]    

display('jacobian of G_qala wrt qa')
jacobian(G_qala,qta)
display('jacobian of G_qala wrt qb')
jacobian(G_qala,qtb)


G_qb = [2*V*Rmat(qtb)'*Rmat([0;pb1;pb2;pb3]);
        cmat*V*Lmat(qta)']*Lmat(qb)*H
    
    
G_qbla = G_qb'*[la1;la2;la3;la4;la5]    

display('jacobian of G_qbla wrt qb')
jacobian(G_qbla,qtb)

display('jacobian of G_qbla wrt qa')
jacobian(G_qbla,qta)