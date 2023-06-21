import numpy as np


### Carthesian and Polar Coordinates

def polar2cart(phi,theta,r,inclination):

    f1 = lambda x: np.cos(x) if inclination else np.sin(x)
    f2 = lambda x: np.sin(x) if inclination else np.cos(x)

    x = r * f1(phi) * np.cos(theta)
    y = r * f1(phi) * np.sin(theta)
    z = r * f2(phi)
    return np.vstack([x,y,z])


def phi_i(x,y,z,inclination):
    if inclination:
        return np.arcsin(z / np.sqrt(x**2+y**2+z**2)) # or arccos

    else:
        return np.arccos(z / np.sqrt(x**2+y**2+z**2)) # or arccos

def theta_i(x,y,z,zero_to_360=True):
    phi = np.arctan2(y,x)
    if zero_to_360: phi[phi<0] = 2*np.pi+phi[phi<0]
    return phi

def cart2polar(xyz_t,inclination,zero_to_360):
    x = xyz_t[0]
    y = xyz_t[1]
    z = xyz_t[2]
    return phi_i(x,y,z,inclination),theta_i(x,y,z,zero_to_360)



## Sampling functions 

import math
from scipy.spatial.transform import Rotation as R
def vertex(x, y, z): 
    """ Return vertex coordinates fixed to the unit sphere """ 
    length = np.sqrt(x**2 + y**2 + z**2) 
    return [i / length for i in (x,y,z)] 

def middle_point(verts,middle_point_cache,point_1, point_2): 
    """ Find a middle point and project to the unit sphere """ 
    # We check if we have already cut this edge first 
    # to avoid duplicated verts 
    smaller_index = min(point_1, point_2) 
    greater_index = max(point_1, point_2) 
    key = '{0}-{1}'.format(smaller_index, greater_index) 
    if key in middle_point_cache: return middle_point_cache[key] 
    # If it's not in cache, then we can cut it 
    vert_1 = verts[point_1] 
    vert_2 = verts[point_2] 
    middle = [sum(i)/2 for i in zip(vert_1, vert_2)] 
    verts.append(vertex(*middle)) 
    index = len(verts) - 1 
    middle_point_cache[key] = index 
    return index

def icosphere(subdiv):
    # verts for icosahedron
    r = (1.0 + np.sqrt(5.0)) / 2.0;
    verts = np.array([[-1.0, r, 0.0],[ 1.0, r, 0.0],[-1.0, -r, 0.0],
                      [1.0, -r, 0.0],[0.0, -1.0, r],[0.0, 1.0, r],
                      [0.0, -1.0, -r],[0.0, 1.0, -r],[r, 0.0, -1.0],
                      [r, 0.0, 1.0],[ -r, 0.0, -1.0],[-r, 0.0, 1.0]]);
    # rescale the size to radius of 0.5
    verts /= np.linalg.norm(verts[0])
    # adjust the orientation
    r = R.from_quat([[0.19322862,-0.68019314,-0.19322862,0.68019314]])
    verts = r.apply(verts)
    verts = list(verts)

    faces = [[0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],
             [0, 10, 11],[1, 5, 9],[5, 11, 4],[11, 10, 2],
             [10, 7, 6],[7, 1, 8],[3, 9, 4],[3, 4, 2],
             [3, 2, 6],[3, 6, 8],[3, 8, 9],[5, 4, 9],
             [2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1],];
    
    for i in range(subdiv):
        middle_point_cache = {}
        faces_subdiv = []
        for tri in faces: 
            v1  = middle_point(verts,middle_point_cache,tri[0], tri[1])
            v2  = middle_point(verts,middle_point_cache,tri[1], tri[2])
            v3  = middle_point(verts,middle_point_cache,tri[2], tri[0])
            faces_subdiv.append([tri[0], v1, v3]) 
            faces_subdiv.append([tri[1], v2, v1]) 
            faces_subdiv.append([tri[2], v3, v2]) 
            faces_subdiv.append([v1, v2, v3]) 
        faces = faces_subdiv
                
    return np.array(verts), faces

def ico(n=None,delta_lamb=0,delta_phi=0,scanner_shadow_angle=0):
    subdiv = n
    coords, faces = icosphere(subdiv=subdiv)
    points = np.mean(coords[np.array(faces)],axis=1).T
    x,y,z = points

    angles = cart2polar(points,inclination=False,zero_to_360=True)
    angles = np.array([(i,j) for i,j in zip(angles[0],angles[1])])
    return points, angles.T#[(p,l) for l,p in zip(lamb,phi)]

def cubemap(n=None,delta_lamb=0,delta_phi=0,scanner_shadow_angle=0):
    angles = np.stack([(np.pi/2,0),(np.pi/2,np.pi*.5),(np.pi/2,np.pi),
                  (np.pi/2,1.5*np.pi),(0,0),(np.pi,0)]).T

    faces = {'front':(0,np.pi/2),'right':(np.pi*.5,np.pi/2),'back':(np.pi,np.pi/2),
                  'left':(1.5*np.pi,np.pi/2),'top':(0,0),'bottom':(0,np.pi)}
    
    

    points = polar2cart(angles[0],angles[1],1,False)
        
 
    return points,angles