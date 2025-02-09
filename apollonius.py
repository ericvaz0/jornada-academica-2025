import numpy as np
import scipy.linalg as la
import plotly.graph_objects as go

#DEFINITIONS:
#A generalized circle is represented by a 2x2 Hermitian matrix [[a, b], [c, d]] where the circle is given by the equation
#a|z|^2 + bz + c\bar{z} + d = 0, where z is a complex number and \bar{z} is the complex conjugate of z.
#Note that this equation represents a true circle exactly when the matrix is positive definite, i.e., when a > 0 and ad - |b|^2 > 0.
#Otherwise, the equation represents a line, which is a circle in the Riemann sphere passing through the point at infinity.
#This matrix representation is not unique, and two hermitian matrices represent the same circle if, and only if, they are proportional (over the real numbers).
#It is convenient to represent a true circle by a matrix with a = 1, and a line with |b| = 1.

#A Mobius transformation is a transformation of the form f(z) = (az + b)/(cz + d), where a, b, c, d are complex numbers and ad - bc != 0.
#Equivalentely, a Mobius transformation can be represented by a 2x2 matrix [[a, b], [c, d]] acting on the complex projective line.
#Mobius transformations map generalized circles to generalized circles, and mutually tangent generalized circles remain tangent after the transformation.
#Two lines are said to be tangent if they are parallel.


#FUNCTIONS:
#All of the plotting functions use the Plotly library and take a figure object (fig) as an argument.

#Plots a circle with a given radius and center. If plot_center is True, the center is also plotted.
def plot_circle(radius, center_x, center_y, fig, plot_center=True):
    # Create the circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius*np.cos(theta) + center_x
    y = radius*np.sin(theta) + center_y

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='blue'))

    # Add the center
    if(plot_center):
        fig.add_trace(go.Scatter(x=[center_x], y=[center_y], mode='markers', marker_color='red'))

#Plots a line given by the equation ax + by + c = 0.
def plot_line(a, b, c, fig):
    if(np.abs(b) < 1e-10):
        x = np.array([-c/a, -c/a])
        y = np.array([-5, 5])
    else:
        x = np.array([-5, 5])
        y = (-a*x-c)/b
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='blue'))

#Plots a point in the complex plane.
def plot_point(z, fig):
    fig.add_trace(go.Scatter(x=[np.real(z)], y=[np.imag(z)], mode='markers', marker_color='red'))

#Plots a generalized circle. If force_line is True, the circle is plotted as a line even if it is a true circle.
def plot_generalized_circle(circle, fig, force_line=False, plot_center=True):
    if(np.abs(circle[0,0]) >= 1e-10 and not force_line):
        center = -circle[1,0]/circle[0,0]
        radius = np.sqrt(np.abs(center)**2 - np.real(circle[1,1]/circle[0,0]))
        plot_circle(radius, np.real(center), np.imag(center), fig, plot_center)
    else:
        plot_line(2*np.real(circle[0,1]), -2*np.imag(circle[0,1]), np.real(circle[1,1]), fig)

    if(force_line):
        print(np.abs(circle[0,0]))

#Applies (the inverse of) a Mobius transformation to a generalized circle and puts the result in the canonical form.
#If force_line is True, the circle it put into the canonical form of a line (|b| = 1), even if |a| > 0.
def mobius_transform_circle(circle, matrix, force_line=False):
    circle_new = (matrix.T @ circle) @ np.conjugate(matrix)
    if(np.abs(circle_new[0,0]) >= 1e-10 and not force_line):
        circle_new = circle_new/circle_new[0,0]
    elif np.abs(circle_new[0,1]) >= 1e-10:
        circle_new = circle_new/np.abs(circle_new[0,1])
    else:
        print("Warning: Degenerate circle")
    return circle_new

#Applies a Mobius transformation to a point.
def mobius_transform_point(z, matrix):
    return (matrix[0,0]*z+matrix[0,1])/(matrix[1,0]*z+matrix[1,1])

#Finds the point of tangency of two circles. If the circles are tangent at infinity, the function returns np.inf.
def point_of_tangency(circle_1, circle_2):
    if(np.abs(circle_1[0,0]) < 1e-10):
        line = circle_1
        circle = circle_2
        if(circle_2[0,0] < 1e-10):
            return np.inf
    elif(circle_2[0,0] < 1e-10):
        line = circle_2
        circle = circle_1
    else:
        line = circle_1 - circle_2
        circle = circle_2

    circle = circle/circle[0,0]
    line = line/np.abs(line[0,1])
    matrix = np.array([[1/line[0,1], -circle[1,0]], [0, 1]], dtype=complex)
    new_line = mobius_transform_circle(line, matrix)
    intersect = mobius_transform_point(-new_line[1,1]/2, matrix)

    return intersect

#Finds the circle that passes through three points (if the points are colinear, this will be a line).
def circle_from_points(z_1, z_2, z_3):
    matrix = np.array([[np.abs(z_1)**2, 2*np.real(z_1), -2*np.imag(z_1), 1],
                       [np.abs(z_2)**2, 2*np.real(z_2), -2*np.imag(z_2), 1],
                       [np.abs(z_3)**2, 2*np.real(z_3), -2*np.imag(z_3), 1]])
    sol = la.null_space(matrix).flatten()
    circle = np.array([[sol[0], sol[1] + 1j*sol[2]], [sol[1] - 1j*sol[2], sol[3]]], dtype=complex)
    if(np.abs(circle[0,0]) >= 1e-10):
        circle = circle/circle[0,0]
    elif np.abs(circle[1,0]) >= 1e-10:
        circle = circle/np.abs(circle[1,0])
    else:
        print("Warning: Degenerate circle")
    return circle

#Returns a circle with a given center and radius.
def circle_from_center_and_radius(center, radius):
    return np.array([[1, -np.conj(center)], [-center, np.abs(center)**2 - radius**2]], dtype=complex)

#Returns the line defined by the equation ax + by + c = 0.
def circle_from_line(a,b,c):
    return np.array([[0, a/2 - (b/2)*1j], [a/2 + (b/2)*1j, c]], dtype=complex)

#Finds the unique two circles that are tangent to three mutually tangent circles.
def descartes_circles(circle_1, circle_2, circle_3):
    kiss_point = point_of_tangency(circle_1, circle_2)

    matrix_1 = np.array([[kiss_point, 1], [1, 0]], dtype=complex)

    line_1 = mobius_transform_circle(circle_1, matrix_1, force_line=True)
    line_2 = mobius_transform_circle(circle_2, matrix_1, force_line=True)
    circle = mobius_transform_circle(circle_3, matrix_1)

    matrix_2 = np.array([[1/line_1[0,1], -circle[1,0]], [0, 1]], dtype=complex)

    circle = mobius_transform_circle(circle, matrix_2)
    line_1 = mobius_transform_circle(line_1, matrix_2, force_line=True)
    line_2 = mobius_transform_circle(line_2, matrix_2, force_line=True)

    radius = np.sqrt(np.abs(circle[1,1]))

    new_circle_1 = circle_from_center_and_radius(2*radius*1j, radius)
    new_circle_2 = circle_from_center_and_radius(-2*radius*1j, radius)

    inverse_matrix = np.linalg.inv(matrix_1 @ matrix_2)
    return mobius_transform_circle(new_circle_1, inverse_matrix), mobius_transform_circle(new_circle_2, inverse_matrix)


#EXAMPLE:
z_1 = -5.71-0.27j
z_2 = -3.05-1.65j
z_3 = -3.97+2.65j
r_1 = 1
r_2 = 2
r_3 = 2.4

circle_1 = np.array([[1, -np.conj(z_1)], [-z_1, np.abs(z_1)**2 - r_1**2]], dtype=complex)
circle_2 = np.array([[1, -np.conj(z_2)], [-z_2, np.abs(z_2)**2 - r_2**2]], dtype=complex)
circle_3 = np.array([[1, -np.conj(z_3)], [-z_3, np.abs(z_3)**2 - r_3**2]], dtype=complex)

new_circle_1, new_circle_2 = descartes_circles(circle_1, circle_2, circle_3)
fig = go.Figure()
fig.update_layout(xaxis_range=[-10, 10], yaxis_range=[-10, 10])
plot_generalized_circle(circle_1, fig)
plot_generalized_circle(circle_2, fig)
plot_generalized_circle(circle_3, fig)
plot_generalized_circle(new_circle_1, fig, plot_center=False)
plot_generalized_circle(new_circle_2, fig, plot_center=False)
fig.show()