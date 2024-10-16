from INaturalistDownloader import INaturalistDownloader

inaturalist_animals = {
    "hazel_dormouse": 45856,
    "wood_mouse": 45562,
    "european_fat_dormouse": 74383,
    "european_garden_dormouse4": 45863,
    "common_shrew": 46414,
    "domestic_cat": 118552,
    "red_fox": 42069,
    "grey_squirrel": 46017
}

inaturalist_trees = {
    "alder_tree": 966205,
    "beech_tree": 54227,
    "english_oak": 56133
}


def main():
    for species_name, species_id in inaturalist_animals.items():
        download_animals(species_id, species_name)

    for species_name, species_id in inaturalist_trees.items():
        download_trees(species_id, species_name)


def download_animals(species_id, species_name):
    downloader = INaturalistDownloader(species_id=species_id, species_name=species_name, alive_only=True)
    downloader.download_images()

def download_trees(species_id, species_name):
    # For trees, alive_query must be removed from the URL for it to work
    downloader = INaturalistDownloader(species_id=species_id, species_name=species_name)
    downloader.download_images()

if __name__ == '__main__':
    main()
