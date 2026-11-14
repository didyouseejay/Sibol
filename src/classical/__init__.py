"""Classical pipeline: HOG features + one-vs-rest linear SVM, written in NumPy.

Modules:
    preprocessing  resize to 128 x 128, grayscale, train-only feature standardisation
    hog            Histogram of Oriented Gradients from scratch (8,100 features)
    linear_svm     multiclass one-vs-rest hinge-loss SVM from scratch
    features       split-level extraction with a validated on-disk cache
"""
