#!/usr/bin/env python2
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image', help="Path to imagine that will be processed.")
    parser.add_argument('threshold', help="", type=int)
    parser.add_argument('clusters', help="Number of clusters", type=int)

    return parser.parse_args()


def main():
    args = _parse_args()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from argparse import ArgumentParser
from os.path import split
import numpy as np
import cv2
from dsp import (binarize, labelling, squares, perimeters,
                 compactnesses, elongations, ParamVector, k_medians)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('threshold', type=int)
    parser.add_argument('num_clusters', type=int)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    print('Binarization')
    binimage = binarize(image, args.threshold)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(2):
        binimage = cv2.morphologyEx(binimage, cv2.MORPH_CLOSE, kernel)
    for _ in range(2):
        binimage = cv2.morphologyEx(binimage, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(split(args.image)[1], binimage)
    print('Labellling')
    labels = labelling(binimage)
    print('Squares')
    sqs = squares(labels)
    print('Perimeters')
    perims = perimeters(labels, sqs)
    print('Compactnesses')
    comps = compactnesses(sqs, perims)
    print('Elongations')
    elongs = elongations(labels)
    print('Vectors')
    vectors = [ParamVector(sqs[l], perims[l], comps[l], elongs[l])
               for l in sqs]
    print('Clusters')
    clusters = k_medians(vectors, args.num_clusters)
    for c, vs in clusters.items():
        print(c, len(vs))


