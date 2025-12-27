from quarismamodules.test import Testing

try:
    pass
except ImportError:
    print("Numpy (http://numpy.scipy.org) not found.")
    print("This test requires numpy!")
    Testing.skip()


class TestHrtmanWatsonDistribution(Testing.quarismaTest):
    def testHrtmanWatsonDistribution(self):
        """n = 12
        t = 3.0
        start = -7.2
        end = 3.1

        size_roots = 26
        roots = vector["double"](size_roots)
        w1 = vector["double"](size_roots)
        w2 = vector["double"](size_roots)
        gaussianQuadrature.gauss_kronrod(size_roots, roots, w1, w2)

        a = np.linspace(start, end, n)
        r = numpy_to_quarisma(a)
        b = np.zeros(n)
        result = numpy_to_quarisma(b)
        hartmanWatsonDistribution.distribution(
            result, t, r, roots, w1, hartman_watson_distribution_enum.MIXTURE
        )"""
        # print(b)


if __name__ == "__main__":
    Testing.main([(TestHrtmanWatsonDistribution, "test")])
