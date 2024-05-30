import os
import requests
from tqdm import tqdm

base_url = 'https://api.giphy.com/v1/gifs/search'
api_key = os.environ['API_KEY']  # Can be easily found by browsing giphy.com and watching the Network tab in devtools.
limit = 100


def downlad_dataset(category='waterfall', dataset_dir='dataset'):
    # Get all relevant gifs url from giphy
    gifs = []
    page = 0
    while True:
        res = requests.get(f'{base_url}?offset={page * limit}&limit={limit}&type=gifs&q={category}&api_key={api_key}')
        tmp = [x['url'] for x in res.json()['data']]
        if len(tmp) == 0:
            break

        gifs += tmp
        page += 1

    # Download all the gifs to a dataset directory
    for gif in tqdm(gifs):
        gif_id = gif.split('-')[-1] if '-' in gif else gif.split('/')[-1]
        if os.path.exists(f'{dataset_dir}/{gif_id}.mp4'):
            continue
        try:
            url = f'https://i.giphy.com/{gif_id}.mp4'
            res = requests.get(url, allow_redirects=True)
            with open(f'{dataset_dir}/{gif_id}.mp4', mode='wb') as file:
                file.write(res.content)
        except:
            print(f'Failed fetching gif {gif_id}')
