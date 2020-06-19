import scipy.linalg
import scipy.spatial

import numpy as np

import itertools
import math

eps = np.finfo(float).eps * 10

def gram_schmidt(vs):
    """ Finds an orthonormal set that spans the same space as the given
        vectors do. """
    ret = [np.array(vs[0]) / np.linalg.norm(vs[0], 2)]
    for i in range(1, len(vs)):
        v = np.array(vs[i])
        for w in ret:
            v -= w.dot(v) * w
        if np.linalg.norm(v, 2) > eps:
            ret.append(v / np.linalg.norm(v, 2))
    return tuple(ret)

def complete_orthonormal_set(vs):
    """ Completes an orthonormal set to an orthonormal basis """
    n = len(vs[0])
    ret = []
    for i in range(n):
        w = np.zeros(n)
        w[i] = 1.
        for v in itertools.chain(vs, ret):
            w -= v.dot(w) * v
        if np.linalg.norm(w, 2) > eps:
            ret.append(w / np.linalg.norm(w, 2))
    return tuple(ret)


class Set(object):
    """ Represents a finitely generated convex sets by (n, ps),
        where n is the dimension of the ambient space and ps is a set
        of generating points. """

    def __init__(self, n, ps):
        self.n = n
        self.ps = tuple(map(tuple,ps))
        self._triang = None

    def _get_triangulation(self):
        if self._triang is None:
            self._triang = scipy.spatial.Delaunay(np.array(self.ps))
        return self._triang

    def __contains__(self, p):
        t = self._get_triangulation()
        return t.find_simplex(p) >= 0

    def __repr__(self):
        return f"Set {self.ps}"

    def prod(self, Y):
        """ Returns the cartesian product self x Y with left and
            right projectors. """
        n = self.n + Y.n
        P = Set(n, [p + q for p in self.ps for q in Y.ps])

        vs = [(1.,) + (0.,)*self.n]
        for i in range(self.n):
            v = [0.] * (self.n + 1)
            v[i+1 ] =1.
            vs.append(tuple(v))
        for i in range(Y.n):
            vs.append((0,) * (self.n+1))

        proj1 = Map(np.column_stack(vs), P, self)

        vs = [(1.,) + (0.,)*Y.n]
        for i in range(self.n):
            vs.append((0,) * (Y.n+1))
        for i in range(Y.n):
            v = [0.] * (Y.n + 1)
            v[i+1] =1.
            vs.append(tuple(v))

        proj2 = Map(np.column_stack(vs), P, self)

        return P, proj1, proj2

    def id(self):
        """ Return identity map """
        return Map(np.identity(self.n+1), self, self)

    def __eq__(self, other):
        if self is other:
            return True
        if self.n != other.n:
            return False
        for p in self.ps:
            if p not in other:
                return False
        for p in other.ps:
            if p not in self:
                return False
        return True

class Map(object):
    """ Represents an (affine) map between finitely generated convex
        sets X=(n, ps) and Y=(m, qs) by a (m+1) x (n+1) matrix A.
        
        The convex set X is embedded into n+1 dim space by by
        mapping (p_0, p_1, …, p_n) to (1, p_0, …, p_n).

        Thus the intended affine map f is given by:

            (1) ‖ f(x) = A (1) ‖ x

        Hence (1, 0, …, 0) is mapped to (1) ‖ f(0, …,  0) by A
        and so (0) ‖ e_i is mapped to (0) ‖ (f(e_i) - f(0)).
        Thus the row of A is (1, 0, …, 0).  """
    def __init__(self, A, src, trg):
        self.A = A
        self.src = src
        self.trg = trg

    def factorize(self):
        """ Factorize the map as an injection after a surjection. """
        # projection from codomain to range
        proj = scipy.linalg.orth(self.A).T

        # Compute the images of the extreme points of the src under A
        # within the range.
        qs = [proj.dot(self.A.dot((1,)+p)) for p in self.src.ps]

        # qs spans a hyperplane.  To compute its normal, we first compute
        # all differences with an arbitrary fixed point in the hyperplane.
        # The unique line orthogonal to them is spanned by the normal, so
        # turn the differences into an orthonormal set with Gram--Schmidt
        # and then complete into an orthonormal basis to find the normal.
        qs2 = gram_schmidt([q - qs[0] for q in qs[1:]])
        normal = complete_orthonormal_set(qs2)[0]
        normal *= normal.dot(qs[0]) # find correct factor

        # We transform our hyperplane so that its normal is (1, 0, ..., 0)
        # as expected by the Set class.
        O = np.column_stack((normal,) + qs2).T

        # Now we have our Set in the middle.  We just need to remove possible
        # convex dependancy.
        mps = [O.dot(q)[1:] for q in qs]
        if len(normal)-1 > 1:
            mps = [mps[i] for i in scipy.spatial.ConvexHull(mps).vertices]
        elif len(normal)-1 == 1:
            lb, ub = min(mps), max(mps)
            mps = [lb]
            if ub != lb: mps.append(ub)
        else:
            raise NotImplementedError

        mid = Set(len(normal)-1, mps)

        surj = Map(np.dot(O, np.dot(proj,  self.A)), self.src, mid)
        inj = Map(np.dot(proj.T, O.T), mid, self.trg)

        return (surj, inj)

    def __call__(self, p):
        """ Computes the image of p under self """
        return tuple(self.A.dot(((1.,) + tuple(p)))[1:])

    def __repr__(self):
        return f"{self.A}\nfrom {self.src}\nto {self.trg}>"

    def check(self):
        """ Checks whether this is a proper map, for instance whether the
            alleged extreme points of the domain map into the codomain. """
        for p in self.src.ps:
            if self(p) not in self.trg:
                raise ValueError

        # Check whether the first row of A is (1, 0, …, 0).
        if not math.isclose(self.A[0][0], 1.):
            raise ValueError
        for i in range(1, self.src.n):
            if not abs(self.A[0][i]) < eps:
                raise ValueError

    def tuple(self, g):
        """ Returns the <self, g> : X -→ Y x Z, where g: X -→ Z. """
        if self.src != g.src:
            raise ValueError("g should have the same domain as self")
        trg, p1, p2 = self.trg.prod(g.trg)
        vs = [(1.,) + tuple(self.A.T[0][1:]) + tuple(g.A.T[0][1:])]
        for i in range(self.src.n):
            vs.append((0.,) + tuple(self.A.T[i+1][1:]) +
                    tuple(g.A.T[i+1][1:]))

        return Map(np.column_stack(vs), self.src, trg)

    def prod(self, g):
        """ Return self x g : X x A -→ Y x B where g: A -→ B. """
        src, p1, p2 = self.src.prod(g.src)
        trg, q1, q2 = self.trg.prod(g.trg)
        return self.after(p1).tuple(g.after(p2))

    def after(self, g):
        """ Composition of maps. """
        return Map(np.dot(self.A, g.A), g.src, self.trg)

    def __eq__(self, other):
        if self is other:
            return True
        return (self.src == other.src and self.trg == other.trg and
                np.allclose(self.A, other.A))


def simplex(n):
    """ Returns a simplex of dimension n """
    # We use the extreme points:
    #  (0, 0, 0, …, 0 )
    #  (1, 0, 0, …, 0 )
    #  (0, 1, 0, …, 0 )
    #       (…)
    #  (0, 0, 0, …, 1 )
    if n == 0: raise ValueError
    ps = [np.zeros(n-1) for i in range(n)]
    for i in range(1, n): ps[i][i-1] = 1.
    return Set(n-1, ps)

def free_map(n, X, ps):
    """ Returns the map from the n-simplex to X, where the ith point
        is mapped to ps[i]. """
    src = simplex(n)

    # (1, 0, ..., 0) is mapped to (1) ‖ ps[0]
    q0 = (1.,) + tuple(ps[0])

    # (1, 1, 0, ... 0) is mapped to (1) ‖ ps[1] so (0, 1, 0, ...) is mapped
    # to (0) ‖ (ps[1] - ps[0]).

    qs = [q0] + [(0.,) + tuple(np.array(ps[i]) - np.array(ps[0]))
                for i in range(1, n)]

    return Map(np.column_stack(qs), src, X)

def F(f, k):
    """ Computes action of the functor F(X) = [0,1] x X^k on the map f. """
    ret = f
    for i in range(k-1):
        ret = ret.prod(f)
    return simplex(2).id().prod(ret)

def solve(xs, ys):
    """ Find a matrix A such that A xs[i] = ys[i] for all i or return None
        if no such matrix exists. """
    if len(xs) != len(ys):
        raise ValueError("There should be as many xs as ys")
    pm = []
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        for x2, y2 in pm:
            c = x2.dot(x)
            x -= c * x2
            y -= c * y2
        norm = np.linalg.norm(x, 2)
        if norm < eps:
            if np.linalg.norm(y, 2) > eps:
                return None
            continue
        x /= norm
        y /= norm
        pm.append((x,y))
    conull = tuple(x for x, y in pm)
    null = complete_orthonormal_set(conull)
    A = np.column_stack(conull + null)
    B = np.column_stack(tuple(y for x, y in pm) + tuple(np.zeros(len(ys[0])) for x in null))
    return np.dot(B, A.T)


tau1 = free_map(4, simplex(4),
    ((.5, .5, 0.),
     (.5, 0., 0.),
     (0., .5, 0.),
     (0., 0, 0.)))
tau2 = free_map(4, simplex(4),
    ((.5, .5, 0.),
     (.5, .5, 0.),
     (0., .5, 0.),
     (0., 0, 0.)))
gamma = free_map(4, simplex(2),
    ((0.1,), (0.2,), (0.2,), (0.2,)))

X = tau1.src
X2, _, _ = X.prod(X)
FX, p1, p2 = simplex(2).prod(X2)
alpha = gamma.tuple(tau1.tuple(tau2)) # α = <γ,τ₁,τ₂>
d = p1.after(alpha) # d = d₁ ≡ γ

print(f"Alpha")
print(f" src {len(alpha.src.ps)} in {alpha.src.n}")
print(f" trg {len(alpha.trg.ps)} in {alpha.trg.n}")

epi = None

step = 0
while True:
    step += 1
    print(f"Step {step}")
    oldEpi = epi
    epi, mono = d.factorize()
    d = F(epi, 2).after(alpha)
    print(f" src {len(d.src.ps)} in {d.src.n}")
    print(f" trg {len(d.trg.ps)} in {d.trg.n}")
    print(d.A)

    if oldEpi is not None:
        delta = solve(oldEpi.A.T, d.A.T)
        if delta is not None:
            print("Finished!")
            print(delta)
            deltaMap = Map(delta, oldEpi.trg, d.trg)
            deltaMap.check()
            print(deltaMap.after(oldEpi) == d)
            break

    if step == 5:
        print(d == alpha)
        break
