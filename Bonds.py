
import sympy as syp

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


#### atom_pos:  Nx3 sympy matrix of atom positions
#### v1,v2,v3:  magnetic unit cell vector
#### bond_vecs: displacement vectors for each type of bonds
#### return: list of bonds. each bond is a list of [site_i_index, site_j_index, BOND_TYPE, cell_shift_vector]
#### you can modify to include more bonds (further neigbhour, interlayer, etc.) or to apply to other lattices
def getHoneycombBonds(atom_pos, v1, v2, v3, bond_vecs):

    if not all( (bv.shape == (3,1)) and (isinstance(bv, syp.matrices.dense.MutableDenseMatrix) or isinstance(bv, syp.matrices.immutable.ImmutableDenseMatrix)) for bv in bond_vecs):
        raise Exception("each bond_vector must be 3x1 matrix") 


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

            for k, vec in enumerate(bond_vecs):
                if d.equals(vec):
                    bonds.append([i,j,k,syp.zeros(3,1)]);

                
    # add bonds between neigbouring unit cells with the corresponding unit cell shifts
    for shift in uc_shift:
        for i in range(atom_pos.shape[0]):
            for j in range(atom_pos.shape[0]):
                d = syp.simplify((atom_pos[j,:] - atom_pos[i,:]).T + shift);
                
                for k, vec in enumerate(bond_vecs):
                    if d.equals(vec):
                        bonds.append([i,j,k,shift]);

                           
    return bonds;