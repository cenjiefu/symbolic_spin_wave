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

import Bonds
import SpinWave as SW

## declare symbolic variables
h, J, alpha, beta = syp.symbols('h, J, alpha, beta', real=True); 

t1, p1, t2, p2, t3, p3, t4, p4, t5, p5, t6, p6, t7, p7, t8, p8, t9, p9, t10, p10, t11, p11, t12, p12 = syp.symbols('t_1, p_1, t_2, p_2, t_3, p_3, t_4, p_4, t_5, p_5, t_6, p_6, t_7, p_7, t_8, p_8, t_9, p_9, t_10, p_10, t_11, p_11, t_12, p_12', real=True);
h1, g1, h2, g2, h3, g3, h4, g4, h5, g5, h6, g6, h7, g7, h8, g8, h9, g9, h10, g10, h11, g11, h12, g12 = syp.symbols('h_1, g_1, h_2, g_2, h_3, g_3, h_4, g_4, h_5, g_5, h_6, g_6, h_7, g_7, h_8, g_8, h_9, g_9, h_10, g_10, h_11, g_11, h_12, g_12', real=True);

start = timer();
N = 4; # the zigzag order has 4 site in the magnetic unit cell

## spin directions (spin_theta, spin_phi, orb_theta, orb_phi) on each site
## can use symbols or numerical values
spin = syp.zeros(N,4); 
spin[0,0] = t1; spin[0,1] = 0; spin[0,2] = h1; spin[0,3] = 0; 
spin[1,0] = t2; spin[1,1] = 0; spin[1,2] = h2; spin[1,3] = 0;
spin[2,0] = t3; spin[2,1] = 0; spin[2,2] = h3; spin[2,3] = 0; 
spin[3,0] = t4; spin[3,1] = 0; spin[3,2] = h4; spin[3,3] = 0;

## define Hamiltonian with types of bonds. Each bond type has a 3x1 displacement vector and two 3x3 interaction matrices for spin and orbital interactions. 
## For other types of interactions, you can modify the spinorb_Hamiltonian() function as needed.
## make sure to use the same convention as in the Bonds.py file
Dz = syp.Matrix([[-alpha, 0, 0],
        [0, -alpha, 0],
        [0, 0, J-alpha]]);
Mz = syp.Matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]);

Dx = syp.Matrix([[J-alpha, 0, 0],
            [0, -alpha, 0],
            [0, 0, -alpha]]);
Mx = syp.Matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]);

Dy = syp.Matrix([[-alpha, 0, 0],
            [0, J-alpha, 0],
            [0, 0, -alpha]]);
My = syp.Matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]);

bond_vecs = [Bonds.delta_x, Bonds.delta_y, Bonds.delta_z]
bond_Js = [[Dx,Mx], [Dy,My], [Dz,Mz]]

## magnetic unit cell vectors
v1 = syp.Matrix([1,0,0]);
v2 = 2*syp.Matrix([syp.Integer(1)/syp.Integer(2),syp.sqrt(3)/2,0]);
v3 = syp.Matrix([0, 0, syp.Integer(1)]);

## position of each site in the magnetic unit cell. Make sure to use the same convention as in the Bonds.py file
atom_pos = syp.zeros(N,3);
atom_pos[0,:] = syp.zeros(1,3);
atom_pos[1,:] = Bonds.delta_x.T;
atom_pos[2,:] = Bonds.delta_x.T-Bonds.delta_z.T;
atom_pos[3,:] = 2*Bonds.delta_x.T-Bonds.delta_z.T;

## calculate all the bonds
bonds = Bonds.getHoneycombBonds(atom_pos, v1,v2,v3, bond_vecs);
print("Bonds. time: ", timer()-start);


## get the symbolic Hamiltonian
H4, E4 = SW.spinorb_Hamiltonian(spin, bonds, bond_Js,beta)
print("Hamiltonian. time: ", timer()-start);


## substitute in the numerical spin directions
ss1 = np.array([[0, 0, 1],
                [0, 0, -1],
                [0, 0, -1],
                [0, 0, 1]]);

afo1 = np.array([[0, 0, -1],
                [0, 0, 1],
                [0, 0, -1],
                [0, 0, 1]]);

sv = (ss1)
ov = (afo1)

N = 4
angles = syp.zeros(N,4);
for i in range(N):
    angles[i,0] = syp.acos(sv[i,2])
    if sv[i,0]>0:
        angles[i,1] = syp.atan(sv[i,1]/sv[i,0])
    else:
        angles[i,1] = syp.atan(sv[i,1]/sv[i,0])+syp.pi
    angles[i,2] = syp.acos(ov[i,2])
    if ov[i,0]>0:
        angles[i,3] = syp.atan(ov[i,1]/ov[i,0])
    else:
        angles[i,3] = syp.atan(ov[i,1]/ov[i,0])+syp.pi

# angles = np.nan_to_num(angles)
angles = angles.subs(syp.nan, 0)
E_num = E4.subs([(t1,angles[0,0]),(p1,angles[0,1]),(h1,angles[0,2]),(g1,angles[0,3]),
                (t2,angles[1,0]),(p2,angles[1,1]),(h2,angles[1,2]),(g2,angles[1,3]),
                (t3,angles[2,0]),(p3,angles[2,1]),(h3,angles[2,2]),(g3,angles[2,3]),
                (t4,angles[3,0]),(p4,angles[3,1]),(h4,angles[3,2]),(g4,angles[3,3])])

display(E_num/(N/2))

H_num = H4.subs([(t1,angles[0,0]),(p1,angles[0,1]),(h1,angles[0,2]),(g1,angles[0,3]),
                (t2,angles[1,0]),(p2,angles[1,1]),(h2,angles[1,2]),(g2,angles[1,3]),
                (t3,angles[2,0]),(p3,angles[2,1]),(h3,angles[2,2]),(g3,angles[2,3]),
                (t4,angles[3,0]),(p4,angles[3,1]),(h4,angles[3,2]),(g4,angles[3,3])])



## substitute in the numerical interactions
J_num = 2.0;
a_num = 0.5;
beta_num = 0.15;

print(E_num.subs([(J,J_num), (alpha,a_num), (beta, beta_num)])/(N/2))
print("beta_crit = ", np.abs(J_num/4 - a_num/2)/np.abs(J_num/2+a_num/2+3/2))
H_num2 = H_num.subs([(J,J_num), (alpha,a_num), (beta, beta_num)])


## diagonalize the Hamiltonian
tau3 = np.eye(H_num.shape[0],dtype=np.complex128);
tau3[range(H_num.shape[0]//2, H_num.shape[0]), range(H_num.shape[0]//2, H_num.shape[0])] = -1;
H_func = syp.lambdify([SW.k1, SW.k2], expr2f(H_num2,10), 'numpy');
print(H_func(0.1,0.1)[:,0])

N_band = H_num.shape[0]
# @nb.jit(nopython=True)
def  e_func(k_list):
    E_list = np.zeros((k_list.shape[0], N_band))
    for i in range(k_list.shape[0]):
        kx = k_list[i,0]; ky = k_list[i,1];
        H = H_func(kx,ky);
        cholH_dag = np.linalg.cholesky(H);
        cholH = cholH_dag.T.conj();
        E = np.linalg.eigvals(cholH@tau3@cholH_dag);
        E_list[i,:] = np.flip(np.sort(np.real(E)));
    return E_list

print(e_func(np.array([[0.01, 0.01]], dtype=float)))
