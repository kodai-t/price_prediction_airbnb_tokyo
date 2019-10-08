import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('data/listings.csv', low_memory=False)

# reduce columns
drop_columns = ['id', 'listing_url','scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description',
                'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules',
                'experiences_offered', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id',
                'host_url', 'host_name', 'host_location', 'host_about', 'host_acceptance_rate', 'host_thumbnail_url',
                'host_picture_url', 'street', 'neighbourhood', 'neighbourhood_group_cleansed', 'state', 'market',
                'smart_location', 'country_code', 'country', 'square_feet', 'weekly_price', 'monthly_price',
                'calendar_last_scraped', 'license', 'jurisdiction_names', 'zipcode', 'first_review', 'last_review',
                'requires_license', 'require_guest_profile_picture', 'require_guest_phone_verification',
                # later I should adjust these below columns to int
                'host_since', 'host_response_time', 'host_response_rate']
reduced_columns_listings = df.drop(drop_columns, axis=1)

# Extract specific areas
specific_areas_listings = reduced_columns_listings[(reduced_columns_listings['neighbourhood_cleansed'] == 'Shibuya Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Shinjuku Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Toshima Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Taito Ku') |
                                                   (reduced_columns_listings['neighbourhood_cleansed'] == 'Minato Ku')]

# A list has NaN, the list will be dropped
specific_areas_listings = specific_areas_listings.dropna(axis=0, how='any')

# Change char to int in 'price'
processed_listings = specific_areas_listings.replace({'price': {'\$': ''}})

# Mapping
cancellation_policy_map = {'modelate': 0, 'strict_14_with_grace_period': 1}
processed_listings['cancellation_policy'].map(cancellation_policy_map)

bool_map = {'f': 0, 't': 1}
processed_listings['instant_bookable'].map(bool_map)
# processed_listings['is_business_travel_ready'].map(bool_map)
# processed_listings['host_is_superhost'].map(bool_map)
# processed_listings['host_has_profile_pic'].map(bool_map)
# processed_listings['host_identity_verified'].map(bool_map)
# processed_listings['is_location_exact'].map(bool_map)
# processed_listings['has_availability'].map(bool_map)

print(processed_listings['instant_bookable'])
quit()
print(processed_listings.where(processed_listings == 't'))

train_set, test_set = train_test_split(processed_listings, test_size=0.25)
sns.heatmap(train_set.astype(float))

