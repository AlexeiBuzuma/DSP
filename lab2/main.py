#!/usr/bin/env python2
import argparse
import cv2
import numpy as np
import os
from dsp import (binarize, labelling, squares, perimeters,
                 compactnesses, elongations, ParamVector, k_medians)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest="image", help="Path to imagine that will be processed.", default="src/2.jpg")
    parser.add_argument('-t', dest="threshold", help="threshold", type=int, default=220)
    parser.add_argument('--number', dest='num_clusters', help="Number of clusters", type=int, default=2)

    return parser.parse_args()


def main():
    args = _parse_args()

    # Imagine Binarization
    print("Start imagine binarization with threshold: '{0}'... ".format(args.threshold))
    image = cv2.imread(args.image)
    binimage = binarize(image, args.threshold)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(2):
        binimage = cv2.morphologyEx(binimage, cv2.MORPH_CLOSE, kernel)
    for _ in range(2):
        binimage = cv2.morphologyEx(binimage, cv2.MORPH_OPEN, kernel)

    path_to_save = os.path.split(args.image)[1]
    cv2.imwrite(path_to_save, binimage)
    print("Binarization successful finished. Saved to '{}'\n".format(path_to_save))

    # Labelling
    print("Start labelling...")
    labels = labelling(binimage)
    print("Labels: {}".format(labels))
    print("Labelling successfully finished.\n")

    # Squaring
    print("Start calculating squares...")
    sqs = squares(labels)
    print("Calculating squares successfully finished.\n")

    # Calculating perimeters
    print("Start calculating perimeters...")
    perims = perimeters(labels, sqs)
    print("Calculating perimeters successfully finished.\n")

    # Calculating compactness
    print("Start calculating compactness...")
    comps = compactnesses(sqs, perims)
    print("Calculating compactness successfully finished.\n")

    print("Start calculating elongations...")
    elongs = elongations(labels)
    print("Calculating elongations successfully finished.\n")

    print("Start creating vectors with parameters...")
    vectors = [ParamVector(sqs[l], perims[l], comps[l], elongs[l]) for l in sqs]
    print("{} vectors successfully created.\n".format(len(vectors)))

    print('Clusters')
    print("Start searching clusters...")
    clusters = k_medians(vectors, args.num_clusters)
    print("Found '{}' clusters:".format(len(clusters)))

    for c, vs in clusters.items():
        print(c, len(vs))


if __name__ == '__main__':
    main()
