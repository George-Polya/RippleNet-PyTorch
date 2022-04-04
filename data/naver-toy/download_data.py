import os
from urllib.request import urlretrieve

url_name_dict = {
    "naver_vit_features.pkl" : "https://docs.google.com/uc?export=download&id=1J9J-NwnyqvdkItCTjXpSMqQLKNxPFyX6"
}

for filename, url in url_name_dict.items():
    path = filename
    urlretrieve(url,path)