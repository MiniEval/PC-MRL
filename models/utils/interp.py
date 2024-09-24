import multiprocessing as mp

from bezier import Curve
import numpy as np


def auto_handles(prev, curr, next, vector=False):
    currs = np.reshape(curr, (-1, 2))
    if prev is not None:
        prevs = np.reshape(prev, (-1, 2))
    else:
        prevs = None
    if next is not None:
        nexts = np.reshape(next, (-1, 2))
    else:
        nexts = None

    p2 = currs[:]

    if prevs is None:
        p3 = nexts[:]
        p1 = p2 * 2.0 - p3
    else:
        p1 = prevs[:]

    if nexts is None:
        p3 = p2 * 2.0 - p1
    else:
        p3 = nexts[:]

    dvec_a = p2 - p1
    dvec_b = p3 - p2

    len_a = np.expand_dims(np.minimum(dvec_a[:, 0], 1.0), axis=-1)
    len_b = np.expand_dims(np.minimum(dvec_b[:, 0], 1.0), axis=-1)

    tvec = (dvec_b / len_b) + (dvec_a / len_a)

    l = 6.0

    # l = tvec[:, 0] * 2.5614
    # if len_a > 5.0 * len_b:
    #     len_a = 5.0 * len_b
    # if len_b > 5.0 * len_a:
    #     len_b = 5.0 * len_a

    len_a /= l
    len_b /= l

    if vector:
        prev_handles = dvec_a * -1.0/3.0 + p2
        next_handles = dvec_b * 1.0/3.0 + p2
    else:
        prev_handles = tvec * -len_a + p2
        next_handles = tvec * len_b + p2

    return prev_handles, next_handles


class Interp:
    def interp(self, length):
        pass


class Lerp(Interp):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def interp(self, length):
        diff = self.k2 - self.k1

        steps = np.array([(float(t) / length) for t in range(length + 1)])
        for i in range(len(self.k1.shape)):
            steps = np.expand_dims(steps, axis=-1)

        interval = np.expand_dims(self.k1, 0).repeat(length + 1, axis=0)
        interval += (np.expand_dims(diff, 0).repeat(length + 1, axis=0) * steps)

        return interval


class Bezier(Interp):
    def __init__(self, p0, p1, p2, p3):
        self.p0 = np.reshape(p0, (-1, 2))
        self.p1 = np.reshape(p1, (-1, 2))
        self.p2 = np.reshape(p2, (-1, 2))
        self.p3 = np.reshape(p3, (-1, 2))

        self.beziers = []

        for i in range(self.p0.shape[0]):
            nodes = np.stack((self.p0[i], self.p1[i], self.p2[i], self.p3[i]), axis=-1)
            nodes[0] -= nodes[0, 0]
            if nodes[0, -1] == 0.0:
                print("Invalid Bezier values")
                exit(1)
            else:
                nodes[0] /= nodes[0, -1]
            self.beziers.append(Curve(np.asfortranarray(nodes), degree=3))

    def interp(self, length):
        steps = np.array([(float(t) / length) for t in range(length)])

        # Multi-thread solution
        with mp.Pool(mp.cpu_count()) as pool:
            args = []
            for index in np.ndindex(length, len(self.beziers)):
                if index[0] == 0:
                    continue
                args.append(index)
            interval = pool.starmap(self.bez_evaluate, [(steps, i, j) for (i, j) in args])
            interval = np.reshape(np.array(interval), (length - 1, len(self.beziers)))

        vals = np.empty((length + 1, self.p0.shape[0]))
        vals[0] = self.p0[:, 1]
        vals[-1] = self.p3[:, 1]
        vals[1:-1] = interval

        return vals

    def bez_evaluate(self, steps, i, j):
        step_bezier = Curve(np.asfortranarray([[steps[i], steps[i]], [-100.0, 100.0]]), degree=1)
        intersection = self.beziers[j].intersect(step_bezier)
        x_array = np.asfortranarray(intersection[:, 0])
        co = self.beziers[j].evaluate_multi(x_array)
        return co[1, 0].item()
