import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cv2 as cv 
import helper
import math
import scipy.ndimage.filters as filt
import random
from scipy.stats import skew

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    #Normalizing
    pts1 = pts1/M
    pts2 = pts2/M

    N = pts1.shape[0]  # N= 110

    # Forming matrix A
    A = np.zeros([N, 9])
    for i in range(N):
        A[i,0] = pts2[i][0]*pts1[i][0]
        A[i,1] = pts2[i][0]*pts1[i][1]
        A[i,2] = pts2[i][0]
        A[i,3] = pts2[i][1]*pts1[i][0]
        A[i,4] = pts2[i][1]*pts1[i][1]
        A[i,5] = pts2[i][1]
        A[i,6] = pts1[i][0]
        A[i,7] = pts1[i][1]
        A[i,8] = 1


    u,s,vT = np.linalg.svd(A)
    v = vT.T
    last_col = v[:,-1]
    F = last_col.reshape(3,3)
    # print (np.linalg.matrix_rank(F))

    # with this function rank is converted to 2 instead of 3
    # F = helper._singularize(F)
    F = helper.refineF(F,pts1,pts2)
    # print (np.linalg.matrix_rank(F))


    #Unnormalizing
    T = np.array([[1/M, 0, 0], [0, 1/M, 0],[0,0,1]])
    a1 = T.T
    F = np.matmul(a1, np.matmul(F, T))

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):

    #Normalizing
    pts1 = pts1/M
    pts2 = pts2/M

    # List of F matrices
    Farray = []

    # Forming matrix A
    A = np.zeros([7, 9])
    for i in range(7):
        A[i,0] = pts2[i][0]*pts1[i][0]
        A[i,1] = pts2[i][0]*pts1[i][1]
        A[i,2] = pts2[i][0]
        A[i,3] = pts2[i][1]*pts1[i][0]
        A[i,4] = pts2[i][1]*pts1[i][1]
        A[i,5] = pts2[i][1]
        A[i,6] = pts1[i][0]
        A[i,7] = pts1[i][1]
        A[i,8] = 1

    # print (A.shape)

    u,s,vT = np.linalg.svd(A)
    v = vT.T
    v1 = v[:,-1]
    v2 = v[:,-2]

    F1 = v1.reshape(3,3) 
    F2 = v2.reshape(3,3)


    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 =  fun(0)
    a1 = 2/3*(fun(1)-fun(-1))-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3 = (fun(1)-fun(-1))/2 - a1

    coeff = [a3, a2, a1, a0]
    # print (coeff)
    a = np.roots(coeff)
    number_roots = 3

    for i in range(3):
        if isinstance(a[i], complex):
            number_roots = 1

    T = np.array([[1/M, 0, 0], [0, 1/M, 0],[0,0,1]])
    a1 = T.T

    for k in range(number_roots):
        temp = a[k] * F1 + (1 - a[k]) * F2
        Farray.append(temp)    
        Farray[k] = helper.refineF(Farray[k],pts1,pts2)
    # #Unnormalizing
        Farray[k] = np.real(np.matmul(np.matmul(a1, Farray[k]),T))

    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.matmul(np.matmul(K2.T, F), K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    pt1 = []
    pt2 = []
    for i in range(len(pts1)):
        pt1.append([pts1[i, 0], pts1[i,1], 1])
        pt2.append([pts2[i, 0], pts2[i,1], 1])
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    N = len(pts2)
    u1, v1, u2, v2 = pts1[:,0], pts1[:,1], pts2[:,0], pts2[:,1]
    P = np.zeros([len(pts1), 4])
    error = np.zeros([len(pts1), 4])
    err_sum = 0
    for i in range(N):
        D1 = u1[i] * C1[2,:] - C1[0,:]
        D2 = v1[i] * C1[2,:] - C1[1,:]
        D3 = u2[i] * C2[2,:] - C2[0,:]
        D4 = v2[i] * C2[2,:] - C2[1,:]
        A = np.array([D1, D2, D3, D4])
        u, s, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        X = X/X[3]
        P[i, :] = X
        pt1_p = np.matmul(C1, X.T)
        pt1_p = pt1_p/pt1_p[2]
        pt2_p = np.matmul(C2, X.T)
        pt2_p = pt2_p/pt2_p[2]
        e1 = pt1[i, :] - pt1_p
        e2 = pt2[i, :] - pt2_p
        e = np.linalg.norm(e1)+ np.linalg.norm(e2)
        err_sum = err_sum +e

    P = P[:, 0:3] 
    err = err_sum

    return P,err



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

    #Gaussian mask
def gkern(kernlen=22, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    mat = np.zeros((kernlen, kernlen))
    mat[kernlen//2, kernlen//2] = 1
    return filt.gaussian_filter(mat, nsig)

def epipolarCorrespondence(im1, im2, F, x1, y1):

    # mask around x1 and  y1
    max_error = np.inf
    st = 11
    sig = 3

    H_P_start = np.zeros([3,1])
    H_P_end = np.zeros([3,1])
    H_P_start[0] = x1
    H_P_start[1] = y1
    H_P_start[2] = 1
    ep_line = np.matmul(F, H_P_start)
    # line --> ax + by + c = 0
    a = ep_line[0]
    b = ep_line[1]
    c = ep_line[2]

    # To take care of the edge case
    if (y1 < 474 and y1 > 6):
        mask1 = im1[int(y1-st):int(y1+st), int(x1-st):int(x1+st)]

    gauss_k_t = gkern(2*st,sig)
    g_kernel = np.dstack((gauss_k_t, gauss_k_t, gauss_k_t))

    error_list = []
    # Assuming object has not moved much between 2 images
    for y2 in range(y1-30,y1+30):  
        x2 = int((-c - b*y2)/a)

        if (x2-st > 0 and (x2+st < im2.shape[1])) and (y2-st>0 and (y2+st<=im2.shape[0])):
            mask2 = im2[int(y2-st):int(y2+st), int(x2-st):int(x2+st)]

            distance = mask1 - mask2
            weight_dis = np.multiply(g_kernel, distance)
            err = np.linalg.norm(weight_dis)
            error_list.append(err)
            
            # Finding the min error and updating it
            if err < max_error:
                max_error = err
                x2_final = x2
                y2_final = y2

    return x2_final, y2_final

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    bestF= 0
    max_in= 0
    iter= 10
    tol= 0.001
    inliers= 0
    N = pts1.shape[0]
    ones = np.ones([N,1])
    pts1_in= pts1
    pts2_in= pts2
    pts1_xx=  np.concatenate((pts1, ones), axis=1)
    pts2_xx=  np.concatenate((pts2, ones), axis=1)

    for k in range(0,iter):
        pts1= []
        pts2= []
        pt_indices= [np.random.randint(0, N-1) for p in range(0, 7)]
        for i in range(0,7):
            pts1.append(pts1_in[pt_indices[i],:])
            pts2.append(pts2_in[pt_indices[i],:])
        pts1_m= np.vstack(pts1)
        pts2_m= np.vstack(pts2)
        F7= sevenpoint(pts1_m,pts2_m, M)
        for j in range(0,len(F7)):
            inlier_pt1 = []
            inlier_pt2 = []
            inliers= 0
            for k in range(0,pts1_xx.shape[0]):
                error_x= np.abs(np.matmul(np.matmul(np.transpose(pts2_xx[k]), F7[j]), pts1_xx[k])) 
                if (error_x < tol):
                    inliers= inliers +1
                    inlier_pt1.append(pts1_xx[k])
                    inlier_pt2.append(pts2_xx[k])
            if(inliers > max_in):
                max_in= inliers
                final= F7[j]
                f_in_pt1 = inlier_pt1
                f_in_pt2 = inlier_pt2
    f_in_pt2 = np.asarray(f_in_pt2)
    f_in_pt1 = np.asarray(f_in_pt1)
    f_in_pt2 = f_in_pt2[:,0:2]
    f_in_pt1 = f_in_pt1[:,0:2]
    print (max_in)
    return final, f_in_pt1, f_in_pt2


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # print(cv.Rodrigues(r)[0]) # To scheck if the function is working correctly
    theta = np.linalg.norm(r, 2)
    unit = r/theta
    unit = unit.reshape(3,1)
    R = np.eye(3,3)*np.cos(theta) + (1-np.cos(theta))*(unit.dot(unit.T)) + skew(unit)*(np.sin(theta))
    # print(R)
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):

    A  =  (R - R.T)/2
    ro = np.asarray([A[2,1], A[0,2], A[1,0]]).T
    s = np.linalg.norm(ro, 2)
    c = (R[0,0] + R[1,1] + R[2,2]-1)/2
    theta = np.arctan2(s,c)

    if s==0 and c==1:
        return np.asarray([0,0,0])

    elif s==0 and c==-1:
        u = ro/s
        v= (R + np.eye(3)).reshape(9,1)
        u = v/np.linalg.norm(v,2)
        r = (u*np.pi)
        if np.linalg.norm(r,2)==np.pi and ( (r[0] ==0 and r[1] ==0 and r[2]<0) or ( r[0]==0 and r[1]<0 ) or (r[0]<0) ):
            return -r
        else:
            return r
    elif np.sin(theta) != 0:
        u = ro/s
        return  u*theta
    else:
        print('False')
        return None


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    n = x.shape[0]
    num = n//3
    P_concat = x[:-6]
    P = P_concat.reshape(p1.shape[0], 3)
    t = x[-3:].reshape(3,1)
    r = x[-6:-3]
    R = rodrigues(r)
    M2 = np.hstack(R, t)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    P[i, :] = X
    pt1_p = np.matmul(C1, X.T)
    p1_hat = pt1_p/pt1_p[2]
    pt2_p = np.matmul(C2, X.T) 
    p2_hat = pt2_p/pt2_p[2] #nX2

    residuals = numpy.concatenate([(p1-p1 hat).reshape([-1]),(p2-p2 hat).reshape([-1])])

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''



def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    R = M2_init[:,0:3]
    t = M2_init[:,3:]

    r = invRodrigues(R)
    x = P_init.flatten()
    x = np.hstack((x, r.flatten()))
    x = np.hstack((x, t.flatten()))
    # print (x.shape)

    func = lambda x: ((rodriguesResidual(K1, M1,p1, K2, p2, x))**2).sum()

    yy = scipy.optimize.minimize(func, x)
    xnew = yy.x



    
    t = x[-3:].reshape(3,1)
    r = x[-6:-3]
    R = rodrigues(r)
    M2 = np.hstack(R, t)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    return M2,P