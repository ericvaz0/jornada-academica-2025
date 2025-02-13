# %%
from sympy import *
init_session()
# %%

#Let SL(2,k) be the group of 2x2 matrices with determinant 1 over a subfield k < C of the complex numbers.
#Its Lia algebra sl(2,k) consists of the 2x2 matrices with trace 0.
#Let sl(2,k) receive the basis {e_1,e_2,e_3}, where:

e_1 = Matrix([[1, 0],
              [0, -1]])

e_2 = Matrix([[0, 1],
              [1, 0]])

e_3 = Matrix([[0, 1],
              [-1, 0]])

#The Adjoint representation of SL(2,k) is defined by Ad_g(x) = g*x*g^(-1), for all g in SL(2,k) and x in sl(2,k).
#The vector space sl(2,k) is endowed with the bilinear product <x,y> = tr(xy)/2. With respect to the basis {e_1,e_2,e_3},
#the inner product is given by the matrix diag(1,1,-1).
#Below we show that Ad(g) preserves the inner product.

a,b,c,d = symbols('a b c d')
g = Matrix([[a, b],
            [c, d]])

ge_1 = (g * e_1 * g.inv())
ge_2 = (g * e_2 * g.inv())
ge_3 = (g * e_3 * g.inv())

#The elements ge_1, ge_2 and ge_3 are the images of the basis elements e_1, e_2 and e_3 under the Adjoint representation.
#They must be put in the form a*e_1 + b*e_2 + c*e_3, for some a,b,c in k.

g_00 = ge_1[0, 0]
g_10 = (ge_1[0, 1] + ge_1[1, 0])/2
g_20 = (ge_1[0, 1] - ge_1[1, 0])/2

g_01 = ge_2[0, 0]
g_11 = (ge_2[0, 1] + ge_2[1, 0])/2
g_21 = (ge_2[0, 1] - ge_2[1, 0])/2

g_02 = ge_3[0, 0]
g_12 = (ge_3[0, 1] + ge_3[1, 0])/2
g_22 = (ge_3[0, 1] - ge_3[1, 0])/2
# %%

#Thus the linear transformation Ad_g is given by the matrix:
Ad_g =  Matrix([[g_00, g_01, g_02],
                [g_10, g_11, g_12],
                [g_20, g_21, g_22]])


product_1 = Matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0,-1]])

#Indeed, Ad_g preserves the product:
simplify((Ad_g * product_1 * Ad_g.T)) == product_1

# %%
#Furthermore, the determinant of Ad_g is 1:
det(Ad_g)

#Thus we obtain a map p : SL(2,k) -> SO(2,1), where SO(2,1) is the group of 2x2 matrices with determinant 1 that preserve an product of signature (2,1).
#Since the kernel of p is the center of SL(2,k), we have that ker(p) = {I,-I}.
#We therefore obtain a homomorphism P : PSL(2,k) -> SO(2,1), where PSL(2,k) = SL(2,k)/{I,-I} is the projective special linear group.

simplify(Ad_g).subs(det(g), 1)
# %%

u,v = symbols('u v')
X = Matrix([u,v])
Y = Matrix([u**2 - v**2, 2*u*v, u**2 + v**2])

g_uv = g @ X
u_ = g_uv[0]
v_ = g_uv[1]

out_1 = Ad_g @ Y
out_2 = Matrix([u_**2 - v_**2, 2*u_*v_, u_**2 + v_**2])
simplify(out_1)
# %%
out_2

# %%
