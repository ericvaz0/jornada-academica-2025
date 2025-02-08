import numpy as np
import scipy.linalg as la
import plotly.graph_objects as go

def plot_circle(a, b, c):
    # Create the circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = a*np.cos(theta) + b
    y = a*np.sin(theta) + c

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='blue'))

    # Add the center
    fig.add_trace(go.Scatter(x=[b], y=[c], mode='markers', marker_color='red'))

    # Show the plot
    fig.show()

def mobius_transform_form(circle, a, b, c, d):
    matrix = np.array([[a,b], [c,d]], dtype=complex)
    return (matrix.T @ circle) @ np.conjugate(matrix)

def mobius_transform_point(z, a, b, c, d):
    return (a*z+b)/(c*z+d)

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

    new_line = mobius_transform_form(line, 1/line[0,1], -circle[1,0], 0, 1)
    intersect = mobius_transform_point(-new_line[1,1]/2, 1/line[0,1], -circle[1,0], 0, 1)

    return intersect

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
    return circle

def dual_descates_configuration(circle_1, circle_2, circle_3, circle_4):
    intersect_12 = point_of_tangency(circle_1, circle_2)
    intersect_13 = point_of_tangency(circle_1, circle_3)
    intersect_14 = point_of_tangency(circle_1, circle_4)
    intersect_23 = point_of_tangency(circle_2, circle_3)
    intersect_24 = point_of_tangency(circle_2, circle_4)
    intersect_34 = point_of_tangency(circle_3, circle_4)

    dual_123 = circle_from_points(intersect_12, intersect_13, intersect_23)
    dual_124 = circle_from_points(intersect_12, intersect_14, intersect_24)
    dual_134 = circle_from_points(intersect_13, intersect_14, intersect_34)
    dual_234 = circle_from_points(intersect_23, intersect_24, intersect_34)

    return(dual_123, dual_124, dual_134, dual_234)

z_1 = -6.35+2.4j
z_2 = -3.17-1.87j
z_3 = -3.27+1.2j

print(circle_from_points(z_1, z_2, z_3))