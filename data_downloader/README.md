## Dormouse Data Downloader

### Overview
The script `download_data.py` automatically downloads images from [INaturalist](https://www.inaturalist.org/), a platform for sharing biodiversity observations.

### Instructions
1. Follow the README in the project root folder to install dependencies.
2. Run `download_data.py` to start the download.
```bash
python data_downloader/download_data.py
```
3. The images are saved in: `data_downloader/images`.

### Download Custom Species
To use this script to download custom species images, edit the `species_data` dictionary in `download_data.py` with the desired INaturalist taxon IDs and names. 
The taxon IDs can be found on the [INaturalist website](https://www.inaturalist.org/) by searching for a species, visiting the species page and looking in the URL. For instance, Hazel Dormouse: https://www.inaturalist.org/observations?taxon_id=45856

### Note
Please follow INaturalist's terms of service and usage policies.
