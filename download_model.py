import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Get confirmation token (needed for large files)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    total = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f, tqdm(
        desc=destination, total=total, unit='iB', unit_scale=True
    ) as bar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

# ✅ Create the folder if it doesn't exist
os.makedirs("Wav2Lip/checkpoints", exist_ok=True)

# ✅ Download the model
file_id = "1jaKU4V5cWuuOpVzhzLX5mVLjwpDPP0Iq"
destination = "Wav2Lip/checkpoints/wav2lip_gan.pth"
download_file_from_google_drive(file_id, destination)
