import os

from models import predict_constructor_model
from utils import download_photos


def f1guesser(download, num_photos, years, build_models):
    if download:
        download_photos(years[0], years[1], num_photos)

    predict_constructor_model(years[0], years[1], build_models)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    f1guesser(
        download=False,
        num_photos=200,
        years=(1990, 2020),
        build_models=['constructor', 'chassis']
    )
