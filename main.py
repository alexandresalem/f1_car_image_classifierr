from utils import download_photos


def f1guesser(download, years, build_models):
    if download:
        download_photos(years[0], years[1])


if __name__ == '__main__':
    f1guesser(
        download=True,
        years=(2010, 2020),
        build_models=['team', 'chassis']
    )
