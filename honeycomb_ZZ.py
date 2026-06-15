# 

import sympy as syp
import numpy as np
import numba as nb
from sympy import N as expr2f
from IPython.display import Math, display
from scipy.integrate import dblquad
from scipy.integrate import nquad
from scipy.special import spence
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import Bonds
import SpinWave as SW

## declare symbolic variables
h, J, K, G, Gp, J3 = syp.symbols('h, J, K, G, Gp, J3', real=True); 

th_H, phi_H, g, t1, p1, t2, p2, t3, p3, t4, p4, t5, p5, t6, p6, = syp.symbols('th_H, phi_H, g, t_1, p_1, t_2, p_2, t_3, p_3, t_4, p_4, t_5, p_5, t_6, p_6', real=True);

start = timer();
N = 4; # the zigzag order has 4 site in the magnetic unit cell

## spin directions (spin_theta, spin_phi, orb_theta, orb_phi) on each site
## can use symbols or numerical values
spin = syp.zeros(N,2); 
spin[0,0] = t1; spin[0,1] = p1; 
spin[1,0] = t1; spin[1,1] = p1; 
spin[2,0] = t2; spin[2,1] = p2; 
spin[3,0] = t2; spin[3,1] = p2;

## define Hamiltonian with types of bonds. Each bond type has a 3x1 displacement vector and a 3x3 interaction matrix. make sure to use the same convention as in the Bonds.py file
th_vec = syp.acos(1/syp.sqrt((3)));
abc = syp.Matrix([[syp.sin(th_vec)*syp.cos(-syp.pi/3), syp.sin(th_vec)*syp.sin(-syp.pi/3), syp.cos(th_vec)],
               [syp.sin(th_vec)*syp.cos(syp.pi/3), syp.sin(th_vec)*syp.sin(syp.pi/3), syp.cos(th_vec)],
               [syp.sin(th_vec)*syp.cos(syp.pi), syp.sin(th_vec)*syp.sin((syp.pi)), syp.cos(th_vec)]]);
C3 = syp.Matrix([[-syp.Integer(1)/2, -syp.sqrt(3)/2, 0],
             [syp.sqrt(3)/2, -syp.Integer(1)/2, 0],
             [0, 0, 1]]).T;

Jpp = syp.Integer(1)/3*K + syp.Integer(2)/3*(G-Gp);
Jcp = syp.Integer(1)/3*K - syp.Integer(1)/3*(G-Gp);
Jab = J - Gp +Jcp;
Jc = (J + 2*Gp + Jpp);

Dz = syp.Matrix([[Jab+Jpp, 0, -syp.sqrt(2)*Jcp],
    [0, Jab-Jpp, 0],
    [-syp.sqrt(2)*Jcp,    0, Jc]]);

Dx = syp.expand(C3.T@Dz@C3);
Dy = syp.expand(C3.T@C3.T@Dz@C3@C3);

bond_vecs = [Bonds.delta_x, Bonds.delta_y, Bonds.delta_z, (Bonds.a1+Bonds.a2)*2/3, (Bonds.a1-2*Bonds.a2)*2/3, (Bonds.a2-2*Bonds.a1)*2/3]
bond_Js = [Dx, Dy, Dz, J3*syp.eye(3), J3*syp.eye(3), J3*syp.eye(3)]

## magnetic unit cell vectors
v1 = syp.Matrix([1,0,0]);
v2 = 2*syp.Matrix([syp.Integer(1)/syp.Integer(2),syp.sqrt(3)/2,0]);
v3 = syp.Matrix([0, 0, syp.Integer(1)]);

## position of each site in the magnetic unit cell. 
atom_pos = syp.zeros(N,3);
atom_pos[0,:] = syp.zeros(1,3);
atom_pos[1,:] = Bonds.delta_x.T;
atom_pos[2,:] = Bonds.delta_x.T-Bonds.delta_z.T;
atom_pos[3,:] = 2*Bonds.delta_x.T-Bonds.delta_z.T;

## calculate all the bonds
bonds = Bonds.getHoneycombBonds(atom_pos, v1,v2,v3, bond_vecs);
print("Bonds. time: ", timer()-start);

## add the external magnetic field
hv = h*g*syp.Matrix([syp.sin(th_H)*syp.cos(phi_H), syp.sin(th_H)*syp.sin(phi_H), syp.cos(th_H)])

## get the symbolic Hamiltonian
H4, E4 = SW.spin_Hamiltonian(spin, bonds, bond_Js, hv)
print("Hamiltonian. time: ", timer()-start);

## Numerical interactions
g_num = 1;
h_num = 0.0;
th_H_num = 0.0; # field direction
phi_H_num = 0.0;
J_num = -3.7;
K_num = -16.0;
G_num = 8.0;
Gp_num = 1.0;
J3_num = 1.8;

## calculate spin directions of the model. can change to numerical root-finding for speed
## substitute in the numerical spin directions
Sv = syp.Matrix([syp.sin(t1)*syp.cos(p1), syp.sin(t1)*syp.sin(p1), syp.cos(t1)]);
Ezz = (2*((-Sv.T)@Dz@Sv + Sv.T@Dx@Sv + Sv.T@Dy@Sv) + (6*J3)*Sv.T@syp.eye(3)@Sv)[0];
Ezz = syp.simplify(Ezz.subs([(p1, 0)]));
ZZsoln = syp.solve(syp.diff(Ezz, t1), t1);
def getZZangle_exact(J_num, K_num, G_num, Gp_num, J3_num):
    
    thzz = (expr2f(ZZsoln[0].subs([(J,J_num),(K,K_num),(G,G_num),(Gp,Gp_num),(J3, J3_num)])));
    if thzz < 0 and thzz > -1/2*np.pi:
        thzz += np.pi/2;
    if thzz < -1/2*np.pi and thzz > -np.pi:
        thzz += np.pi;
        
    # r = -G_num/(K_num+Gp_num)
    # thzz = np.pi/2 - 0.5*np.arctan(4*np.sqrt(2)*(1+r)/(7*r-2))

    # print(thzz*180/np.pi)
    return float(thzz);




th_exact = getZZangle_exact(J_num,K_num,G_num,Gp_num,J3_num)
t1_num = th_exact; p1_num = 0.0; 
t2_num = np.pi-th_exact; p2_num = np.pi;
print("zigzag directions. time: ", timer()-start);

H_k_num = H4.subs([(h, h_num), (g, g_num), (th_H, th_H_num), (phi_H, phi_H_num), (J,J_num), (K,K_num), (G,G_num), (Gp,Gp_num),(J3,J3_num),
    (t1, t1_num), (p1, p1_num), (t2, t2_num), (p2, p2_num)])


## diagonalize the Hamiltonian
tau3 = np.eye(H_k_num.shape[0],dtype=np.complex128);
tau3[range(H_k_num.shape[0]//2, H_k_num.shape[0]), range(H_k_num.shape[0]//2, H_k_num.shape[0])] = -1;
H_func = nb.jit(syp.lambdify([SW.k1, SW.k2, SW.k3], expr2f(H_k_num,10), 'numpy'), nopython=True); # if the Hamiltonian is large, numba jit will take a long time.


N_band = H_k_num.shape[0]
@nb.jit(nopython=True)
def  e_func(k_list):
    E_list = np.zeros((k_list.shape[0], N_band))
    for i in range(k_list.shape[0]):
        kx = k_list[i,0]; ky = k_list[i,1]; kz = k_list[i,2];
        H = H_func(kx,ky,kz);
        cholH_dag = np.linalg.cholesky(H);
        cholH = cholH_dag.T.conj();
        E = np.linalg.eigvals(cholH@tau3@cholH_dag);
        E_list[i,:] = np.flip(np.sort(np.real(E)));
    return E_list


print(e_func(np.array([[float(Bonds.M3_point[0]), float(Bonds.M3_point[1]), 0.0]])))
print("numerical Hamiltonian. time: ", timer()-start);

k_path = np.stack(( np.linspace(0, float(Bonds.M3_point[0])*2, 101), np.linspace(0, float(Bonds.M3_point[1])*2, 101), np.linspace(0, 0, 101) )).T
Ek = e_func(k_path)
print("spectrum. time: ", timer()-start);

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(Ek[:,0])
ax.plot(Ek[:,1])

plt.show()
