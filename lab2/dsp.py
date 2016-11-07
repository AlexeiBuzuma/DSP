import sys
from typing import List, Tuple, Dict
from random import choice
from math import sqrt, atan
from glob import glob
from collections import Counter, namedtuple, defaultdict
from statistics import median
import numpy as np
from scipy.spatial import distance
sys.setrecursionlimit(10000)

SOURCES = sorted(glob('src/*'))


def height(image: np.ndarray) -> int:
    return image.shape[0]


def width(image: np.ndarray) -> int:
    return image.shape[1]


def brightness(image: np.ndarray, change: int) -> np.ndarray:
    h = height(image)
    w = width(image)
    corrected = image.copy()
    for x in range(w):
        for y in range(h):
            for i, v in enumerate(corrected[y][x]):
                if v + change > 255:
                    corrected[y][x][i] = 255
                elif v + change < 0:
                    corrected[y][x][i] = 0
                else:
                    corrected[y][x][i] += change
    return corrected


def labelling(image: np.ndarray) -> np.ndarray:
    def fill(x, y):
        if (labels[y][x] == 0) and (image[y][x] == 255):
            labels[y][x] = L
            if x > 0:
                fill(x-1, y)
            if x < w - 1:
                fill(x+1, y)
            if y > 0:
                fill(x, y-1)
            if y < h - 1:
                fill(x, y+1)

    h = height(image)
    w = width(image)
    labels = np.zeros((h, w), dtype=np.int32)
    L = 1
    for y in range(h):
        for x in range(w):
            fill(x, y)
            L += 1
    return labels


def grayscale(image: np.ndarray) -> np.ndarray:
    h = height(image)
    w = width(image)
    gs = np.zeros((h, w), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            gs[y][x] = round(sum(image[y][x])/3)
    return gs


def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
    h = height(image)
    w = width(image)
    binimage = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            if sum(image[y][x])/3 > threshold:
                binimage[y][x] = 255
    return binimage


def squares(labels: np.ndarray) -> Counter:
    return Counter(labels.flat)


def is_inner(labels: np.ndarray, x: int, y: int) -> bool:
    label = labels[y][x]
    neighbours = {(row_index, line_index) for row_index in range(x-1, x+2)
                  for line_index in range(y-1, y+2)}
    neighbours -= {(x, y)}
    return all(labels[y][x] == label for x, y in neighbours)


def perimeters(labels: np.ndarray, squares_cnt: Counter) -> Dict[int, int]:
    inners_cnt = Counter()
    for y in range(1, height(labels)-1):
        for x in range(1, width(labels)-1):
            label = labels[y][x]
            inners_cnt.update({label: is_inner(labels, x, y)})
    return {key: value-inners_cnt[key] for key, value in squares_cnt.items()}


def compactness(square: float, perimeter: float) -> float:
    if square:
        return perimeter**2/square
    else:
        return 0


def compactnesses(sqs: Counter, perims: Counter) -> Dict[int, float]:
    return {int(label): compactness(sqs[label], perims[label])
            for label in sqs}


def mass_center(labels: np.ndarray, figure_class: int) -> Tuple[float, float]:
    square = squares(labels)[figure_class]
    x_center = 0
    y_center = 0
    for x in range(width(labels)):
        for y in range(height(labels)):
            belongs_to_figure = labels[y][x] == figure_class
            x_center += x*belongs_to_figure
            y_center += y*belongs_to_figure
    return (x_center/square, y_center/square)


def dcm(i: int, j: int, labels: np.ndarray, figure_class: int,
        mass_center: Tuple[float, float]) -> float:
    m = 0
    for x in range(width(labels)):
        for y in range(height(labels)):
            belongs_to_figure = labels[y][x] == figure_class
            xc = mass_center[0]
            yc = mass_center[1]
            m += ((x - xc)**i)*((y - yc)**j)*belongs_to_figure
    return m


def dcm11(labels: np.ndarray, figure_class: int, mass_center) -> float:
    return dcm(1, 1, labels, figure_class, mass_center)


def dcm02(labels: np.ndarray, figure_class: int, mass_center) -> float:
    return dcm(0, 2, labels, figure_class, mass_center)


def dcm20(labels: np.ndarray, figure_class: int, mass_center) -> float:
    return dcm(2, 0, labels, figure_class, mass_center)


def elongation(m02: float, m11: float, m20: float) -> float:
    t = sqrt((m20 - m02)**2 + 4*m11**2)
    t2 = (m20 + m02 - t)
    if t2:
        return (m20 + m02 + t)/t2
    else:
        return 0


def elongations(labels: np.ndarray) -> Dict[int, float]:
    res = {}
    ls = set(labels.flat)
    for label in ls:
        mc = mass_center(labels, label)
        m02 = dcm02(labels, label, mc)
        m11 = dcm11(labels, label, mc)
        m20 = dcm20(labels, label, mc)
        res[label] = elongation(m02, m11, m20)
    return res


def orientation(labels: np.ndarray, figure_class: int) -> float:
    m20 = dcm20(labels, figure_class)
    m02 = dcm02(labels, figure_class)
    m11 = dcm11(labels, figure_class)
    return atan(2*m11/(m20-m02))/2


ParamVector = namedtuple('ParamVector',
                         'square perimeter compactness elongation')


euclidean = distance.euclidean


def init_medians(vectors: List[ParamVector], k: int) -> List[ParamVector]:
    medians = []
    while k > 0:
        vector = choice(vectors)
        if vector in medians:
            continue
        else:
            medians.append(vector)
            k -= 1
    return medians


def recalculate_medians(clusters: defaultdict,
                        medians: List[ParamVector]) -> List[ParamVector]:
    for index, vectors in clusters.items():
        square = median((v.square for v in vectors))
        perimeter = median((v.perimeter for v in vectors))
        compactness = median((v.compactness for v in vectors))
        elongation = median((v.elongation for v in vectors))
        m = ParamVector(square=square, perimeter=perimeter,
                        compactness=compactness, elongation=elongation)
        medians[index] = m
    return medians


def create_clusters(medians: List[ParamVector],
                    vectors: List[ParamVector]) -> defaultdict:
    clusters = defaultdict(set)
    for vector in vectors:
        d = [(i, euclidean(m, vector)) for i, m in enumerate(medians)]
        min_index, _ = min(d, key=lambda x: x[1])
        clusters[min_index].add(vector)
    return clusters


def k_medians(vectors: List[ParamVector], k: int) -> defaultdict:
    medians = init_medians(vectors, k)
    clusters = create_clusters(medians, vectors)
    for _ in range(9):
        medians = recalculate_medians(clusters, medians)
        clusters = create_clusters(medians, vectors)
    return clusters

# вектора из параметров (площадь, периметр, компактность, вытянутость)
# выбираем вектора рандомно
# посчитать Евклидово расстояние до остальных векторов,
#  и ближайшие отнести к кластеру
# пересчитать центры - посчитать медианы каждого из параметров
#  векторов кластера
# снова отнести вектора к кластерам
# так раз 4-5
