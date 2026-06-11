
import sympy as syp

X_BOND = 'X'; 
Y_BOND = 'Y'; 
Z_BOND = 'Z';
J3_BOND = 'J3';

# primitive unit cell vectors of the honeycomb lattice
a1 = syp.Matrix([-syp.Integer(1)/2,syp.sqrt(3)/2,0]);
a2 = syp.Matrix([syp.Integer(1)/2,syp.sqrt(3)/2,0]);
a3 = syp.Matrix([0, 0, syp.Integer(1)]);

b1 = 2*syp.pi*syp.Matrix([-1,1/syp.sqrt(3),0]);
b2 = 2*syp.pi*syp.Matrix([1,1/syp.sqrt(3),0]);
K2_point = (2*b1+b2)/3;
K1_point = (b1-b2)/3;
K3_point = (b1+2*b2)/3;
M1_point = (b1)/2;
M2_point = (b1+b2)/2;
M3_point = b2/2;
X1_point = (b1-b2)/3/2;
X2_point = (2*b1+b2)/3/2;
X3_point = (b1+2*b2)/3/2;

delta_z = syp.Matrix([0,-1/syp.sqrt(3),0]);
delta_x = a2 + delta_z;
delta_y = a1 + delta_z;


#### atom_pos: Nx3 sympy matrix of atom positions
#### v1,v2,v3: magnetic unit cell vector
#### return: list of bonds. each bond is a list of [site_i_index, site_j_index, BOND_TYPE, cell_shift_vector]
#### you can modify to include more bonds (further neigbhour, interlayer, etc.) or to apply to other lattices
def getHoneycombBonds(atom_pos, v1, v2, v3):

    bonds = [];

    # position shift between unit cells
    uc_shift = [v1, -v1, v2, -v2, v3, -v3,
                v1+v2, v1-v2, -v1+v2, -v1-v2, v1+v3, v1-v3, -v1+v3, -v1-v3, v2+v3, v2-v3, -v2+v3, -v2-v3,
                v1+v2+v3, v1+v2-v3, v1-v2+v3, v1-v2-v3, -v1+v2+v3, -v1+v2-v3, -v1-v2+v3, -v1-v2-v3];
    
    #  add bonds within one unit cell with zero unit cell shift
    d3 = syp.Matrix([0,0,v3[2]/3]);
    for i in range(atom_pos.shape[0]):
        for j in range(atom_pos.shape[0]):
            d = syp.simplify(atom_pos[j,:] - atom_pos[i,:]).T;

            if d.equals(delta_z):
                bonds.append([i,j,Z_BOND,syp.zeros(3,1)]);
            if d.equals(delta_x):
                bonds.append([i,j,X_BOND,syp.zeros(3,1)]);
            if d.equals(delta_y):
                bonds.append([i,j,Y_BOND,syp.zeros(3,1)]);
            if d.equals((a1+a2)*2/3) or d.equals((a1-2*a2)*2/3) or d.equals((a2-2*a1)*2/3):
                bonds.append([i,j,J3_BOND,zeros(3,1)]);

                
    # add bonds between neigbouring unit cells with the corresponding unit cell shifts
    for shift in uc_shift:
        for i in range(atom_pos.shape[0]):
            for j in range(atom_pos.shape[0]):
                d = syp.simplify((atom_pos[j,:] - atom_pos[i,:]).T + shift);
                
                if d.equals(delta_z):
                    bonds.append([i,j,Z_BOND,shift]);
                if d.equals(delta_x):
                    bonds.append([i,j,X_BOND,shift]);
                if d.equals(delta_y):
                    bonds.append([i,j,Y_BOND,shift]);
                if d.equals((a1+a2)*2/3) or d.equals((a1-2*a2)*2/3) or d.equals((a2-2*a1)*2/3):
                    bonds.append([i,j,J3_BOND,shift]);

                           
    return bonds;