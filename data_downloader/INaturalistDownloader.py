import requests
import os


class INaturalistDownloader:
    def __init__(self, species_id, alive_only=False, species_name=None):
        self.species_id = species_id
        self.species_name = species_name
        self.alive_only = alive_only
        self.URL = self._build_URL()
        self.output_folder = self._create_output_folder()
        self.photo_count = 0

    def download_images(self):
        page = 1
        while True:
            page_URL = "{}&page={}".format(self.URL, page)
            response = requests.get(page_URL)
            results = response.json().get('results', None)

            if results:
                self._download_images_from_results(results)
                print(f"Page {page} complete.")
                print(f"Photo download count: {self.photo_count}.")
                page += 1
            else:
                # Reached the end of pages, exit loop
                break

        print("Download finished.")

    def _download_images_from_results(self, results):
        for result in results:
            for photo in result['photos']:
                # Download the photo
                image_url = photo['url'].replace('square', 'original')
                image_response = requests.get(image_url)

                # Save the photo to the output folder
                image_name = f"{str(photo['id'])}.jpg"
                output_filepath = os.path.join(self.output_folder, image_name)
                with open(output_filepath, 'wb') as f:
                    f.write(image_response.content)
                print(f"Downloaded image: {output_filepath}")
                self.photo_count += 1

    def _build_URL(self):
        base_URL = "https://api.inaturalist.org/v1/observations"
        alive_query = "&term_id=17&term_value_id=18" if self.alive_only else ""
        return f"{base_URL}?taxon_id={self.species_id}&quality_grade=research{alive_query}&photo_license=CC0,CC-BY,CC-BY-NC"

    def _create_output_folder(self):
        folder = f"images/{self.species_name}" if self.species_name else f"images/{self.species_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

