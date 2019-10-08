import pandas as pd

df = pd.read_csv('data/listings.csv', low_memory=False)

# reduce columns
drop_columns = ['id', 'listing_url','scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description',
                'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules',
                'experiences_offered', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id',
                'host_url', 'host_name', 'host_location', 'host_about', 'host_acceptance_rate', 'host_thumbnail_url',
                'host_picture_url', 'street', 'neighbourhood', 'neighbourhood_group_cleansed', 'state', 'market',
                'smart_location', 'country_code', 'country', 'square_feet', 'weekly_price', 'monthly_price',
                'calendar_last_scraped', 'license', 'jurisdiction_names']
reduced_columns_listings = df.drop(drop_columns, axis=1)

# Extract specific areas
specific_areas_listings = reduced_columns_listings[(reduced_columns_listings['neighbourhood_cleansed'] == 'Shibuya Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Shinjuku Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Toshima Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Taito Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Minato Ku')]

# change char to int in 'price'
specific_areas_listings = specific_areas_listings.replace({'price': {'\$': ''}}, regex=True)
print(specific_areas_listings)
