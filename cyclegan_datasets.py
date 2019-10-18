"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'apple2orange_train': 2014,
    'apple2orange_test': 514
#    'apple2orange_trainA': 995,
#    'apple2orange_rainB': 1019,
#    'apple2orange_testA': 266,
#    'apple2orange_testB': 248
#'horse2zebra_train': 1334,
#'horse2zebra_test': 140
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'apple2orange_train': '.jpg',
    'apple2orange_test': '.jpg',
#'apple2orange_trainB': '.jpg',
#    'apple2orange_testB': '.jpg'
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'apple2orange_train': './cyclegan-1/input/apple2orange/apple2orange_train.csv',
    'apple2orange_test': './cyclegan-1/input/apple2orange/apple2orange_test.csv',
# 'apple2orange_trainB': './cyclegan-1/input/apple2orange/apple2orange_trainB.csv',
#'apple2orange_testB': './cyclegan-1/input/apple2orange/apple2orange_testB.csv',
}
