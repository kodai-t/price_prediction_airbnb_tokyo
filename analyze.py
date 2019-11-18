import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers

# Read the file
df = pd.read_csv('data/listings.csv', low_memory=False)

# Remove useless columns
drop_columns = ['id', 'listing_url','scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description',
                'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules',
                'experiences_offered', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id',
                'host_url', 'host_name', 'host_location', 'host_about', 'host_acceptance_rate', 'host_thumbnail_url',
                'host_picture_url', 'street', 'neighbourhood', 'neighbourhood_group_cleansed', 'state', 'market',
                'smart_location', 'country_code', 'country', 'square_feet', 'weekly_price', 'monthly_price',
                'calendar_last_scraped', 'license', 'jurisdiction_names', 'zipcode', 'first_review', 'last_review',
                'requires_license', 'require_guest_profile_picture', 'require_guest_phone_verification',
                # later I should adjust these below columns to int
                'host_since', 'host_response_time', 'host_response_rate', 'host_neighbourhood', 'host_verifications',
                'city', 'amenities', 'calendar_updated', 'cancellation_policy',
                # Remove some data based on correlation
                'host_total_listings_count']
data = df.drop(drop_columns, axis=1)
data = data.rename(columns={'neighbourhood_cleansed': 'area'})

# Extract specific areas
data = data[(data['area'] == 'Shibuya Ku') | (data['area'] == 'Shinjuku Ku') | (data['area'] == 'Toshima Ku') |
            (data['area'] == 'Taito Ku') | (data['area'] == 'Minato Ku')]

# A list has NaN, the list will be dropped
data = data.dropna(axis=0, how='any')

# Change char to int in 'price'
data = data.replace({'price': {'\$': '', ',': ''}, 'security_deposit': {'\$': '', ',': ''},
                     'cleaning_fee': {'\$': '', ',': ''}, 'extra_people': {'\$': '', ',': ''}}, regex=True)

# Put 0 or 1 based on boolean data
bool_map = {'f': 0, 't': 1}
data['instant_bookable'].replace(bool_map, inplace=True)
data['is_business_travel_ready'].replace(bool_map, inplace=True)
data['host_has_profile_pic'].replace(bool_map, inplace=True)
data['host_identity_verified'].replace(bool_map, inplace=True)
data['host_is_superhost'].replace(bool_map, inplace=True)
data['is_location_exact'].replace(bool_map, inplace=True)
data['has_availability'].replace(bool_map, inplace=True)


# One-hot encoding
data = pd.get_dummies(data, columns=['area', 'property_type', 'room_type', 'bed_type'])


# get correlation
data = data.astype(dtype='float')
data_corr = data.corr()

# Extract columns which has high correlation with 'price'
high_corr_columns = []
for column, corr in data_corr['price'].items():
    # You can change threshold
    if abs(corr) > 0.05:
        high_corr_columns.append(column)

print(high_corr_columns)
