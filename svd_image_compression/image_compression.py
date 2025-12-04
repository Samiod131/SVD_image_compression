import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import sys


def reverse_reduced_SVD(M, cut=0.7, max=1000, normalize=True, init_norm=True, norm_ord=1, limit_max=False):
    '''
    This is a reduced SVD function. But it does the cut the opposite way.
    Cuts the highest values of the svd as to reach the cut variable value.
    Max is the limit of svd values cut, not kept!
    Built for image compression analysis
    cut is the norm value cut for lower svd values;
    limit_max activates an upper limit to the spectrum's size;
    normalize activates normalization of final svd spectrum;
    norm_ord choose the vector normalization order;
    init_norm make use of relative norm for unormalized tensor's decomposition.

    '''

    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    intensity = np.linalg.norm(S, ord=norm_ord)
    # relative norm calculated for cut evaluation
    if init_norm == True:
        norm_S = S/np.linalg.norm(S, ord=norm_ord)
    else:
        norm_S = S
    norm_sum = 0
    i = 0
    # Evaluating final SVD value kept (index), for size limit fixed or not
    if limit_max == True:
        while norm_sum <= cut and i <= max-1 and i <= S.size-1:
            norm_sum += norm_S[i]
            i += 1
    else:
        while norm_sum <= cut and i <= S.size-1:
            norm_sum += norm_S[i]
            i += 1
    # Final renormalization of SVD vaues kept or not, returning the correct
    # matrices sizes
    if normalize == True:
        return U[:, i:], (S[i:]/np.linalg.norm(S[i:], ord=norm_ord)*intensity), Vh[i:, :], i, S/S[0]
    else:
        return U[:, i:], S[i:], Vh[i:, :], i, S/S[0]


def reduced_SVD(M, cut=0.3, max=1000, normalize=False, init_norm=True, norm_ord=1, limit_max=True):
    '''
    This is a reduced SVD function.
    cut is the norm value cut for lower svd values;
    limit_max activates an upper limit to the spectrum's size;
    normalize activates normalization of final svd spectrum;
    norm_ord choose the vector normalization order;
    init_norm make use of relative norm for unormalized tensor's decomposition.

    '''

    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    intensity = np.linalg.norm(S, ord=norm_ord)
    # relative norm calculated for cut evaluation
    if init_norm == True:
        norm_S = S/np.linalg.norm(S, ord=norm_ord)
    else:
        norm_S = S
    norm_sum = 0
    i = 0
    # Evaluating final SVD value kept (index), for size limit fixed or not
    if limit_max == True:
        while norm_sum <= (1-cut) and i <= max-1 and i <= S.size-1:
            norm_sum += norm_S[i]
            i += 1
    else:
        while norm_sum <= (1-cut) and i <= S.size-1:
            norm_sum += norm_S[i]
            i += 1
    # Final renormalization of SVD values kept or not, returning the correct
    # matrices sizes
    if normalize == True:
        return U[:, :i], (S[:i]/np.linalg.norm(S[:i], ord=norm_ord)*intensity), Vh[:i, :], i, S/S[0]
    else:
        return U[:, :i], S[:i], Vh[:i, :], i, S/S[0]


#name = 'vangogh.png'
#cut = 0.2
name = input("Enter full in folder PNG file name: ")
cut = float(
    input("Enter relative matrix norm to be cut out ([0,1], suggestion=0.1): "))

img = imread(name)
shape = img.shape
img = img.reshape(shape[0], shape[1], -1)
or_float_num = img.size
shape = img.shape

saved_svd = []
full_svd = []
float_num = 0
U_float_num = []
S_float_num = []
V_float_num = []
for j in range(img.shape[2]):
    print('Compressing color '+str(j+1)+' out of '+str(img.shape[2]))
    U, S, Vh, i, fS = reverse_reduced_SVD(img[:, :, j], cut=cut, limit_max=False)
    #U, S, Vh, i, fS = reduced_SVD(img[:, :, j], cut=cut, limit_max=False)
    U_float_num.append(U.size)
    S_float_num.append(S.size)
    V_float_num.append(Vh.size)
    float_num += (U.size+S.size+Vh.size)
    saved_svd.append(S.size)
    full_svd.append(fS)
    a = np.dot(U, np.diag(S))
    img[:, :, j] = np.dot(a, Vh)

# We make sure no values are out of range
img = np.clip(img, 0, 1)
print('Compression ratio: '+str("{:.2f}".format(or_float_num/float_num)))
print('original size:'+str(or_float_num))
print('new size:'+str(float_num))
print('U sizes:'+str(U_float_num))
print('S sizes:'+str(S_float_num))
print('Vt sizes:'+str(V_float_num))


# Daa analysis, images an plots saving
fig = plt.figure()
ax = plt.subplot()

if shape[2] == 1:
    img = img.reshape(shape[0], shape[1])
    plt.imsave('comp_'+name, img, cmap='gray')
    ax.plot(full_svd[0], 'k', label='Black and white')

else:
    plt.imsave('comp_'+name, img)
    plt.plot(full_svd[0], 'r', label='Red')
    plt.plot(full_svd[1], 'g', label='Green')
    plt.plot(full_svd[2], 'b', label='Blue')


ax.legend()
plt.xlabel(r'$i$')
plt.ylabel(r'$\frac{S[i]}{S[0]}$', fontsize=14)
plt.title('SVD spectrum for '+name)

#plt.text(0.65*full_svd[0].size,0.7,'Compression ratio: '+str("{:.2f}".format(or_float_num/float_num)))
plt.tight_layout()
fig.savefig(name+'_svd.png')
