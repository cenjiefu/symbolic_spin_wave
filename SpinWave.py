import sympy as syp
import Bonds

Spin, Sa1, Sb1, Sc1, Ta1, Tb1, Tc1, Sa2, Sb2, Sc2, Ta2, Tb2, Tc2 = syp.symbols('S, Sa1, Sb1, Sc1, Ta1, Tb1, Tc1, Sa2, Sb2, Sc2, Ta2, Tb2, Tc2', real=True); 
Sp1, Sm1, Tp1, Tm1, Sp2, Sm2, Tp2, Tm2 = syp.symbols('Sp1, Sm1, Tp1, Tm1, Sp2, Sm2, Tp2, Tm2', real=True);
a1, a1__d, a2, a2__d, a3, a3__d, gam, gam__d = syp.symbols('a1, a1__d, a2, a2__d, a3, a3__d, gam, gam__d', real=True);
b1, b1__d, b2, b2__d, b3, b3__d = syp.symbols('b1, b1__d, b2, b2__d, b3, b3__d', real=True);

A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44 = syp.symbols('A11,A12,A13,A14,A21,A22,A23,A24,A31,A32,A33,A34,A41,A42,A43,A44', real=True);
B11,B12,B13,B14,B21,B22,B23,B24,B31,B32,B33,B34,B41,B42,B43,B44 = syp.symbols('B11,B12,B13,B14,B21,B22,B23,B24,B31,B32,B33,B34,B41,B42,B43,B44', real=True);


k1, k2, k3 = syp.symbols('k1, k2, k3', real=True);
k_vec = syp.Matrix([k1,k2,k3]);

S_local1 = syp.Matrix([Sa1, Sb1, Sc1]); # spin operators in local coordiantes of atom
S_local2 = syp.Matrix([Sa2, Sb2, Sc2]);
T_local1 = syp.Matrix([Ta1, Tb1, Tc1]); # orbital operators in local coordiantes of atom
T_local2 = syp.Matrix([Ta2, Tb2, Tc2]);

def expr_contains(expr_tup,x,y):
    indices = [];
    if type(expr_tup) is tuple:
        for i in range(len(expr_tup)):
            coeff, x_power = expr_tup[i].as_coeff_exponent(x)
            coeff, y_power = (expr_tup[i]/x).as_coeff_exponent(y)
            if x_power >=1 and y_power >=1:
                indices.append(i);
    else:
        coeff, x_power = expr_tup.as_coeff_exponent(x)
        coeff, y_power = (expr_tup/x).as_coeff_exponent(y)
        if x_power >=1 and y_power >=1:
            return True;
            
    return indices;

## rotation matrix such that S = R@S', S' is in local coordinate of spins
def get_R(theta,phi):
    # first rotation about b axis by theta then about c axis by phi
    R = syp.Matrix([[syp.cos(theta)*syp.cos(phi), -syp.sin(phi), syp.sin(theta)*syp.cos(phi)],
               [syp.cos(theta)*syp.sin(phi), syp.cos(phi) ,syp.sin(theta)*syp.sin(phi)],
               [-syp.sin(theta), syp.Integer(0), syp.cos(theta)]]);
    return R;

## perform the generalized Holstein-Primakoff transformation on the spin-orbital Hamiltonian
def generalized_HPT_transform(spin,b,bond_Js,beta):
    
    at1 = b[0];
    at2 = b[1];
    R1s = get_R(spin[at1,0], spin[at1,1]);
    R1t = get_R(spin[at1,2], spin[at1,3]);
    R2s = get_R(spin[at2,0], spin[at2,1]);
    R2t = get_R(spin[at2,2], spin[at2,3]);
    D = R2s.T@bond_Js[b[2]][0]@R1s
    M = R2t.T@bond_Js[b[2]][1]@R1t
    
    terms = ((S_local2.T@D@S_local1)[0]+beta)*((T_local2.T@M@T_local1)[0]-beta);

    expr = syp.expand(terms);

    expr = syp.expand(expr).subs([(Sa1, (Sp1 + Sm1)/2), (Sb1, -syp.I*(Sp1 - Sm1)/2), (Sa2, (Sp2 + Sm2)/2), (Sb2, -syp.I*(Sp2 - Sm2)/2),
                              (Ta1, (Tp1 + Tm1)/2), (Tb1, -syp.I*(Tp1 - Tm1)/2), (Ta2, (Tp2 + Tm2)/2), (Tb2, -syp.I*(Tp2 - Tm2)/2)]);
    

    expr = syp.expand(expr).subs([(Sp1*Tc1, (A12-A34)/2), (Sm1*Tc1, (A21-A43)/2), (Sp2*Tc2, (B12-B34)/2), (Sm2*Tc2, (B21-B43)/2),
                             (Sc1*Tp1, (A13-A24)/2), (Sc1*Tm1, (A31-A42)/2), (Sc2*Tp2, (B13-B24)/2), (Sc2*Tm2, (B31-B42)/2),
                             (Sp1*Tp1, A14), (Sm1*Tm1,  A41), (Sp1*Tm1, A32), (Sm1*Tp1, A23), (Sp2*Tp2, B14), (Sm2*Tm2, B41), (Sp2*Tm2, B32), (Sm2*Tp2, B23),
                             (Sc1*Tc1, (A11-A22-A33+A44)/4), (Sc2*Tc2, (B11-B22-B33+B44)/4)])
    

    expr = syp.expand(expr).subs([(Sp1, A12+A34), (Sm1, A21+A43), (Sc1, (A11-A22+A33-A44)/2), (Sp2, B12+B34), (Sm2, B21+B43), (Sc2, (B11-B22+B33-B44)/2),
                             (Tp1, A13+A24), (Tm1, A31+A42), (Tc1, (A11+A22-A33-A44)/2), (Tp2, B13+B24), (Tm2, B31+B42), (Tc2, (B11+B22-B33-B44)/2)]);
    
    expr = syp.expand(expr.subs([(A32, A31*A12), (A23, A21*A13), (A42, A41*A12), (A24, A21*A14), (A43, A41*A13), (A34, A31*A14),
                      (B32, B31*B12), (B23, B21*B13), (B42, B41*B12), (B24, B21*B14), (B43, B41*B13), (B34, B31*B14),
                      (A22, A21*A12), (A33, A31*A13), (A44, A41*A14), (B22, B21*B12), (B33, B31*B13), (B44, B41*B14),
                      (A11, Spin - a1__d*a1 - a2__d*a2 - a3__d*a3), (B11, Spin - b1__d*b1 - b2__d*b2 - b3__d*b3),
                      (A21, a1__d), (A12, a1), (A31, a2__d), (A13, a2), (A41, a3__d), (A14, a3),
                      (B21, b1__d), (B12, b1), (B31, b2__d), (B13, b2), (B41, b3__d), (B14, b3)]));
        
    return syp.expand(expr);



## extract the spin wave Hamiltonian from the HPT transformation
## spin: spin directions (spin_theta, spin_phi, orb_theta, orb_phi) of all sites
## bonds: all bonds
## bond_Js: list of two 3x3 interaction matrices (spin and orbital) for each type of bonds
## return: symbolic spin wave Hamiltonian and classical ground state energy
def spinorb_Hamiltonian(spin,bonds,bond_Js,beta):

    N = spin.shape[0];
    H = syp.zeros(2*N*3);
    E_cla = 0;
    
    for b in bonds:

        expr = generalized_HPT_transform(spin,b,bond_Js,beta);
        # Fourier transform phase factor
        phase = syp.Integer(1);
        if b[3]!=0:
            phase = syp.exp(-syp.I*k_vec.T*b[3])[0];

        E_cla += sum([expr.args[i] for i in expr_contains(expr.args, Spin,Spin)]);

        expr2 = syp.Integer(0);
        for e in expr.args:
            coeff, a1_power = e.as_coeff_exponent(a1)
            coeff, a2_power = e.as_coeff_exponent(a2)
            coeff, a3_power = e.as_coeff_exponent(a3)
            coeff, b1_power = e.as_coeff_exponent(b1)
            coeff, b2_power = e.as_coeff_exponent(b2)
            coeff, b3_power = e.as_coeff_exponent(b3)

            coeff, a1d_power = e.as_coeff_exponent(a1__d)
            coeff, a2d_power = e.as_coeff_exponent(a2__d)
            coeff, a3d_power = e.as_coeff_exponent(a3__d)
            coeff, b1d_power = e.as_coeff_exponent(b1__d)
            coeff, b2d_power = e.as_coeff_exponent(b2__d)
            coeff, b3d_power = e.as_coeff_exponent(b3__d)

            if (a1_power+a2_power+a3_power+b1_power+b2_power+b3_power + a1d_power+a2d_power+a3d_power+b1d_power+b2d_power+b3d_power) == 2:
                expr2 += e
        
        
        at1 = b[0];
        at2 = b[1];
        a1_list = [a1__d, a2__d, a3__d];
        a2_list = [b1, b2, b3];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = syp.conjugate(phase);
                H[at1*3+i,at2*3+j] += H_term*gamma;
                H[at2*3+N*3+j, at1*3+N*3+i] += (H_term*gamma).subs([(k1,-k1), (k2,-k2)]);

                ## the conjugate a*b__d terms
                H[at2*3+j,at1*3+i] += syp.conjugate(H_term*gamma);
                H[at1*3+N*3+i,at2*3+N*3+j] += syp.conjugate(H_term*gamma).subs([(k1,-k1), (k2,-k2)]);
                    
                    
        a1_list = [a1__d, a2__d, a3__d];
        a2_list = [a1, a2, a3];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = 1;
                H[at1*3+i,at1*3+j] += H_term;
                H[at1*3+N*3+j,at1*3+N*3+i] += H_term;
                ## the conjugate terms are already included in the ij permuations

                H[at2*3+i,at2*3+j] += H_term;
                H[at2*3+N*3+j,at2*3+N*3+i] += H_term;
        
                    
        a1_list = [a1__d, a2__d, a3__d];
        a2_list = [b1__d, b2__d, b3__d];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = syp.conjugate(phase);
                H[at1*3+i,at2*3+N*3+j] += H_term*gamma;
                H[at2*3+j, at1*3+N*3+i] += (H_term*gamma).subs([(k1,-k1), (k2,-k2)]);
                
                ## the conjugate a*b terms terms
                H[at2*3+N*3+j,at1*3+i] += syp.conjugate(H_term*gamma);
                H[at1*3+N*3+i, at2*3+j] += syp.conjugate(H_term*gamma).subs([(k1,-k1), (k2,-k2)]);



    H = H.subs(Spin, 1)
    E_cla = E_cla.subs(Spin, 1)

    return H, E_cla



## perform the Holstein-Primakoff transformation on the spin Hamiltonian
def HPT_transform(spin,b,bond_Js):
    
    at1 = b[0];
    at2 = b[1];
    R1s = get_R(spin[at1,0], spin[at1,1]);
    R2s = get_R(spin[at2,0], spin[at2,1]);

    D = R2s.T@bond_Js[b[2]]@R1s
    terms = (S_local2.T@D@S_local1)[0];

    expr = syp.expand(terms);

    expr = syp.expand(expr).subs([(Sa1, (Sp1 + Sm1)/2), (Sb1, -syp.I*(Sp1 - Sm1)/2), (Sa2, (Sp2 + Sm2)/2), (Sb2, -syp.I*(Sp2 - Sm2)/2)]);
    
    expr = syp.expand(expr).subs([(Sp1, syp.sqrt(2*Spin)*a1), (Sp2, syp.sqrt(2*Spin)*b1), (Sm1, syp.sqrt(2*Spin)*a1__d), (Sm2, syp.sqrt(2*Spin)*b1__d), (Sc1, Spin - a1__d*a1), (Sc2, Spin - b1__d*b1)]);
        
    return syp.expand(expr);


## extract the spin wave Hamiltonian from the HPT transformation
## spin: spin directions (spin_theta, spin_phi) of all sites
## bonds: all bonds
## bond_Js: 3x3 interaction matrices for each type of bonds
## return: symbolic spin wave Hamiltonian and classical ground state energy
def spin_Hamiltonian(spin,bonds,bond_Js,hv):

    N = spin.shape[0];
    H = syp.zeros(2*N);
    E_cla = syp.Integer(0);
    
    for b in bonds:

        expr = HPT_transform(spin,b,bond_Js);
        # Fourier transform phase factor
        phase = syp.Integer(1);
        if b[3]!=0:
            phase = syp.exp(-syp.I*k_vec.T*b[3])[0];

        E_cla += sum([expr.args[i] for i in expr_contains(expr.args, Spin,Spin)]);

        expr2 = syp.Integer(0);
        for e in expr.args:
            coeff, a1_power = e.as_coeff_exponent(a1)
            coeff, b1_power = e.as_coeff_exponent(b1)

            coeff, a1d_power = e.as_coeff_exponent(a1__d)
            coeff, b1d_power = e.as_coeff_exponent(b1__d)

            if (a1_power+b1_power + a1d_power+b1d_power) == 2:
                expr2 += e
        
        
        at1 = b[0];
        at2 = b[1];
        a1_list = [a1__d];
        a2_list = [b1];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = syp.conjugate(phase);
                H[at1+i,at2+j] += H_term*gamma;
                H[at2+N+j, at1+N+i] += (H_term*gamma).subs([(k1,-k1), (k2,-k2)]);

                H[at2+j,at1+i] += syp.conjugate(H_term*gamma);
                H[at1+N+i,at2+N+j] += syp.conjugate(H_term*gamma).subs([(k1,-k1), (k2,-k2)]);
                    
                    
        a1_list = [a1__d];
        a2_list = [a1];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = 1;
                H[at1+i,at1+j] += H_term;
                H[at1+N+j,at1+N+i] += H_term;

                H[at2+i,at2+j] += H_term;
                H[at2+N+j,at2+N+i] += H_term;
        
                    
        a1_list = [a1__d];
        a2_list = [b1__d];
        for i in range(len(a1_list)):
            for j in range(len(a2_list)):
                H_term = sum([expr2.args[k]/(a1_list[i]*a2_list[j]) for k in expr_contains(expr2.args, a1_list[i],a2_list[j])]);
                if isinstance(H_term, (int, float)):
                    H_term = syp.Integer(1)*H_term
                gamma = syp.conjugate(phase);
                H[at1+i,at2+N+j] += H_term*gamma;
                H[at2+j, at1+N+i] += (H_term*gamma).subs([(k1,-k1), (k2,-k2)]);
                
                H[at2+N+j,at1+i] += syp.conjugate(H_term*gamma);
                H[at1+N+i, at2+j] += syp.conjugate(H_term*gamma).subs([(k1,-k1), (k2,-k2)]);


        ### if the Hamiltonian has single-ion anisotropy term S^2, then need to include the a*a and a__d*a__d cases
        

    for i in range(N):
        spin_dir = syp.Matrix([syp.sin(spin[i,0])*syp.cos(spin[i,1]), syp.sin(spin[i,0])*syp.sin(spin[i,1]), syp.cos(spin[i,0])])
        H[i,i] += (hv.T@spin_dir)[0];
        H[i+N,i+N] += (hv.T@spin_dir)[0];
        E_cla += -(hv.T@spin_dir)[0]*Spin;

    H = H.subs(Spin, syp.Integer(1)/2)
    E_cla = E_cla.subs(Spin, syp.Integer(1)/2)



    return H, E_cla