import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = "https://www.airlinequality.com/airline-reviews/british-airways/"
total_pages = 380

def get_reviews(base_url, pages):
    reviews = []
    for page in range(1, pages + 1):
        url = f"{base_url}/page/{page}/"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # İncelemeleri içeren HTML elemanlarını seç
            review_blocks = soup.find_all('div', class_='text_content', itemprop='reviewBody')

            for block in review_blocks:
                review_text = block.text.strip()
                reviews.append(review_text)

            print(f"Page {page} scraped successfully.")
            time.sleep(1)  # İstekler arasında bir saniye bekleyin (isteklerin engellenmemesi için)
        else:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
            break

    return reviews


# Tüm sayfaları gezip incelemeleri çek
all_reviews = get_reviews(base_url, total_pages)

# Verileri bir DataFrame'e kaydet
if all_reviews:
    df = pd.DataFrame(all_reviews, columns=["Review"])
    # Veriyi "data" klasörüne kaydet
    df.to_csv('data/skytrax_reviews.csv', index=False)
    print("Data saved successfully!")
else:
    print("No reviews found.")
