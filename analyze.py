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

# Extract specific areas
data = data[(data['neighbourhood_cleansed'] == 'Shibuya Ku') | (data['neighbourhood_cleansed'] == 'Shinjuku Ku') |
            (data['neighbourhood_cleansed'] == 'Toshima Ku') | (data['neighbourhood_cleansed'] == 'Taito Ku') |
            (data['neighbourhood_cleansed'] == 'Minato Ku')]

# A list has NaN, the list will be dropped
data = data.dropna(axis=0, how='any')

# Change char to int in 'price'
data = data.replace({'price': {'\$': '', ',': '', '.00': ''}, 'security_deposit': {'\$': '', ',': '', '.00': ''},
                      'cleaning_fee': {'\$': '', ',': '', '.00': ''}, 'extra_people': {'\$': '', ',': '', '.00': ''}},
                    regex=True)

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
feature_columns = []

neighbourhood_cleansed_vocabulary_list = data['neighbourhood_cleansed'].unique()
neighbourhood_cleansed = feature_column.categorical_column_with_vocabulary_list('neighbourhood_cleansed', neighbourhood_cleansed_vocabulary_list)
neighbourhood_cleansed_one_hot = feature_column.indicator_column(neighbourhood_cleansed_vocabulary_list)
feature_columns.append(neighbourhood_cleansed_one_hot)

property_type_vocabulary_list = data['property_type'].unique()
property_type = feature_column.categorical_column_with_vocabulary_list('property_type', property_type_vocabulary_list)
property_type_one_hot = feature_column.indicator_column(property_type)
feature_columns.append(property_type_one_hot)

room_type_vocabulary_list = data['room_type'].unique()
room_type = feature_column.categorical_column_with_vocabulary_list('room_type', room_type_vocabulary_list)
room_type_one_hot = feature_column.indicator_column(room_type)
feature_columns.append(room_type_one_hot)

bed_type_vocabulary_list = data['bed_type'].unique()
bed_type = feature_column.categorical_column_with_vocabulary_list('bed_type', bed_type_vocabulary_list)
bed_type_one_hot = feature_column.indicator_column(bed_type)
feature_columns.append(bed_type_one_hot)

