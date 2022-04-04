import os
from urllib.request import urlretrieve

url_name_dict = {
    "movieLens_vit_features.pkl" : "https://docs.google.com/uc?export=download&id=1ASwXAyz2YUBXj_6g1ziBduXpPStfX1xj"
}



for filename, url in url_name_dict.items():
    path = filename
    urlretrieve(url,path)