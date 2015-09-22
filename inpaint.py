"""
Class and functions for image in-painting via total variation minimization
"""

import warnings
import sys
from numpy import clip, empty,  Inf, mod, sum, vstack, zeros
from numpy.linalg import norm



def l22tv(image):
    """
    Computes the L2-norm-squared total variation and a subgradient of the image
    """

    pass



def l2tv(image):

    """
    Computes the L2-norm total variation and a subgradient of an image

    References
    ==========
    Adapted from Stanford EE 364B Convex Optimization II final

    """
    m, n = image.shape

    # Pixel value differences across columns
    col_diff = image[: -1, 1 :] - image[: -1, : -1]

    # Pixel value differences across rows
    row_diff = image[1 :, : -1] - image[: -1, : -1]

    # Compute the L2-norm total variation of the image
    diff_norms = norm(vstack((col_diff.T.flatten(), row_diff.T.flatten())).T, ord=2, axis=1)
    val = sum(diff_norms) / ((m - 1) * (n - 1))

    # Compute a subgradient. When non-differentiable, set to 0
    # by dividing by infinity.
    subgrad = zeros((m, n))
    norms_mat = diff_norms.reshape(n - 1, m - 1).T
    norms_mat[norms_mat == 0] = Inf
    subgrad[: -1, : -1] = - col_diff / norms_mat
    subgrad[: -1, 1 :] = subgrad[: -1, 1 :] + col_diff / norms_mat
    subgrad[: -1, : -1] = subgrad[: -1, : -1] - row_diff / norms_mat
    subgrad[1 :, : -1] = subgrad[1 :, : -1] + row_diff / norms_mat

    return val, subgrad



def make_mask(shape, known_coords):

    mask = zeros(shape)
    mask[known_coords] = 1
    return mask.astype(bool) # necessary? probably slower...



class Inpaint(object):

    """
    Inpaints an image with unknown pixels

    Parameters
    ==========
    alpha : float, optional, default = 200
        Numerator parameter in square-summable-but-not-summable (SSBNS...
        or better non-lame name?) step size

    beta : float, optional, default = 1
        Denominator parameter in SSBNS step size

    max_iter : int, optional, default = 1000
        Maximum number of iterations

    method : str in {'l2tv'}, optional, default = 'l2tv'
        Method for inpainting. Currently only projected subgradient for L2-norm
        total variation minimization supported.

    store : boolean, optional, default = False
        Whether to store and return each iterate

    tol : float, optional, default = 1e-3
        Stopping tolerance on L2 norm of the difference between iterates

    verbose : boolean, optional, default = False
        Whether to print objective per iteration


    References
    ==========
    Something Bertsekas wrote on SSBNS. Need to find

    """

    def __init__(self, alpha=200, beta=1, max_iter=1000, method='l2tv', store=False,
                tol=1e-4, verbose=False):

        if method not in ('l2tv'):
            raise ValueError('Invalid method: got %r instead of one of %r' %
                            (method, ('l2tv')))
        self.method = method
        
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.store = store
        self.tol = tol
        self.verbose = verbose



    def _l2tv(self, image, mask):

        # Projected subgradient method for L2-norm total variation (L2TV) minimization.

        # Initialize iterate.
        painted = image
        painted_best = image
        objective = empty(self.max_iter)
        obj_best = Inf
        if self.store:
            iterates = self.max_iter * [None]
        else:
            iterates = None

        for n_iter in range(self.max_iter):

            # Compute L2TV objective and subgradient.
            obj, subgrad = l2tv(painted)
            objective[n_iter] = obj
            if self.store:
                iterates[n_iter] = painted

            # Update iterate with best objective so far.
            if obj < obj_best:
                obj_best = obj
                painted_best = painted

            if self.verbose:
                if mod(n_iter, 100) == 0:
                    print 'Iter: %i. Objective: %f. Best objective: %f.' %(n_iter, obj, obj_best)
                    sys.stdout.flush()

            # Update iterate by stepping in negative subgradient direction.

            # TODO: Try searching for step size that produces sufficient decrease
            # ("Armijo rule along the projection arc" in Bertsekas (1999),
            # using shortcut condition in Lin (2007) Eq. (17)).

            painted_prev = painted
            painted = painted - (self.alpha / (self.beta + n_iter)) * subgrad

            # Projection onto feasible set, or set all known pixel values in
            # non-fancy speak.
            painted[mask] = image[mask]
            clip(painted, 0, 256, painted)

            # Check for convergence.
            if norm(painted - painted_prev) / norm(painted) < self.tol:
                break

        if self.verbose:
            print 'Iter: %i. Final objective %f. Best objective %f.' % (n_iter, obj, obj_best)
            sys.stdout.flush()

        objective = objective[: n_iter + 1]
        if self.store:
            iterates = iterates[: n_iter + 1]
        painted = painted_best

        return painted, objective, iterates



    def transform(self, image, known_coords):

        # TODO: Check for scaling in [0, 1] vs. [0, 255] and set
        # np.clip() parameter in _l2tv() correspondingly

        # Implement L2-norm-squared total variation for kicks? Should converge faster
        # since quadratic, same optimum.

        """
        Inpaints image given unknown pixel coordinates

        Parameters
        ==========
        image : array, shape (m, n, n_channel)
            Input image with unknown pixel values. n_channel is the number
            of color channels, i.e. n_channel = 3 if RGB, n_channel = 4 if RGBA, etc.

        known_coords : tuple (array-like, array-like)
            x- and y-coordinates of known pixel values

        Returns
        =======
        painted : array, shape (m, n)
            Inpainted image

        objective : array, shape (n_iter)
            For each channel, value of objective at each iteration

        iterates : list of n_iter arrays of shape (m, n)
            If self.store == True, a list of each iterate. Else, None.

        """

        image = image.astype(float)
        shape = image.shape

        if len(shape) == 2:

            mask = make_mask(shape, known_coords)
            if self.method == "l2tv":
                painted, objective, iterates = self._l2tv(image, mask)

            # Cast back to ints.
            painted = painted.astype(int)
            if self.store:
                iterates = [iterate.astype(int) for iterate in iterates]

        elif len(shape) == 3:

            raise ValueError("Color images not supported: naively in-painting " \
                "each channel separately sucks bad. Need to figure out algorithm to do them jointly...")

            m, n, n_channel = shape
            painted = empty(shape)
            objective = n_channel * [None]
            iterates = n_channel * [None]

            # Convert known coordinates into binary mask
            # for easier indexing. Was much faster in MATLAB,
            # need to make sure also faster here.
            mask = make_mask((m, n), known_coords)

            for chan in range(n_channel):

                if self.verbose:
                    print "\nIn-painting channel %i." % chan 

                if self.method == "l2tv":
                    painted[:, :, chan], objective[chan], iterates[chan] = self._l2tv(image[:, :, chan], mask)

        else:
            raise ValueError('Invalid input dimensions: image has %i dimensions, ' + 
                            'but needs to have 2 or 3.' % len(shape))

        return painted, objective, iterates


