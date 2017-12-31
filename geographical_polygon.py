import numpy as np


def transform_coordinates(latitude, longitude):
    """
    Transforms geographical coordinates to spherical coordinates in radians.

    :param latitude: float
    :param longitude: float
    :return: float [0, pi/2), float (-pi, pi]
    """
    scale = np.pi/180.
    return 0.5*(0.5*np.pi + latitude*scale), longitude*scale


def coordinates_to_complex(latitude, longitude):
    """
    Maps geographical coordinates (all apart the north pole) to complex plane.

    :param latitude: float
    :param longitude: float
    :return: complex or np.array of complex numbers
    """
    alpha, argument = transform_coordinates(latitude, longitude)
    dist = np.arctan(alpha)
    return np.exp(1.j*argument)*dist


class ClosedChain:
    """
    Class for representing polygons with chain specified by its nodes.
    Takes huge advantage of representing point on plane as complex numbers.

    Usage:


    curve = [(1, 1), (2, 1), (2, 2), (1, 2)]
    polygonal = ClosedChain(*curve)
    pt1 = (1.5, 1.5)
    pt4 = (3, 3)

    print(polygonal.is_inside(pt1)) # True
    print(polygonal.is_inside(pt2)) # False
    print(polygonal.is_inside([pt1, pt2])) # [True False]

    """
    def __init__(self, *nodes):
        """
        :param nodes: (float, float) / complex numbers / array-like of  (float, float) / array-like of complex numbers
        """
        try:
            nodes = np.array([complex(*p) for p in nodes], dtype=np.complex128)
        except TypeError:
            nodes = np.array([p for p in nodes], dtype=np.complex128)
        self.nodes = np.squeeze(nodes)
        self.nodes = self.nodes[:-1] if self.nodes[0] == self.nodes[-1] else self.nodes
        self.n = len(self.nodes)
        reals = self.nodes.real
        self.min_real, self.max_real = min(reals), max(reals)
        imags = self.nodes.imag
        self.min_imag, self.max_imag = min(imags), max(imags)

    def winding_number(self, *points):
        """
        Calculates winding number of the chain for given point, i.e.  the total number of times that curve
        travels counterclockwise around the point (counted counter-clockwise).

        :param points: (float, float) / complex numbers / array-like of  (float, float) / array-like of complex numbers
        :return: np.int32 or np.array(dtype=np.int32)
        """
        try:
            points = np.array([complex(*p) for p in points], dtype=np.complex128)
        except TypeError:
            points = np.array([p for p in points], dtype=np.complex128)
        points = np.squeeze(points)

        pi2 = 2. * np.pi
        total_length = 1 if not points.shape else len(points)
        in_range = self._in_box(points)
        in_range = np.array(in_range)
        points = np.matrix(points[in_range]).T
        in_range_count = len(points)

        polygon = np.zeros((in_range_count, self.n+1), dtype=np.complex128)
        tiled_nodes = np.tile(self.nodes, (in_range_count, 1))
        tiled_points = np.tile(points, (1, self.n))
        polygon[:, :-1] = tiled_nodes - tiled_points
        polygon[:, -1] = polygon[:, 0]
        results = np.zeros(total_length)

        angles = np.angle(polygon)
        diff_angles = np.diff(angles, axis=1)
        skipped = np.abs(diff_angles) > np.pi
        signs_of_skipped = np.sign(diff_angles[skipped])
        diff_angles[skipped] = (pi2 - np.abs(diff_angles[skipped])) * (-1.*signs_of_skipped)
        winding_numbers = np.sum(diff_angles, axis=1) / pi2

        try:
            results[in_range] = winding_numbers
        except IndexError:
            try:
                results = winding_numbers[0]
            except IndexError:
                    results = 0
        return np.int32(results)

    def is_inside(self, *points):
        """
        Checks if point lays inside the chain.

        :param points: (float, float) / complex numbers / array-like of  (float, float) / array-like of complex numbers
        :return: bool or np.array(dtype=bool)
        """
        return np.abs(self.winding_number(*points)) > 0.5

    def _in_box(self, z):
        """
        Checks if point lays in range of values of the chain. Boosts up speed of calculating winding number.

        :param z: complex number or np.array of complex numbers
        :return: bool or np.array(dtype=bool)
        """
        real, imag = z.real, z.imag
        imag_range = np.logical_and(imag <= self.max_imag, imag >= self.min_imag)
        real_range = np.logical_and(real <= self.max_real, real >= self.min_real)
        return np.logical_and(imag_range, real_range)
