from models import predict_constructor_model
from utils import download_photos


def f1guesser(download, years, build_models):
    if download:
        download_photos(years[0], years[1])
    for model in build_models:
        if model == 'constructor':
            predict_constructor_model(years[0], years[1])


if __name__ == '__main__':
    f1guesser(
        download=False,
        years=(2010, 2020),
        build_models=['constructor']
    )
