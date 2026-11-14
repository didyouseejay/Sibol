"""Classical pipeline: hand-crafted features + one-vs-rest linear SVM, written in NumPy.

Modules:
    preprocessing  resize to 128 x 128 (direct or centre crop), grayscale, train-only standardisation
    hog            Histogram of Oriented Gradients from scratch (8,100 features); flip permutation
    color          HSV hue x saturation histograms from scratch (320 features); flip permutation
    linear_svm     multiclass one-vs-rest hinge-loss SVM from scratch
    features       split-level extraction of HOG / colour with a validated on-disk cache
    experiment     Milestone 2b recipes (sqrt, colour, flip, centre crop) and validation grid search
"""
