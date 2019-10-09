import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

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
                'host_since', 'host_response_time', 'host_response_rate', 'host_neighbourhood', 'host_verifications',
                'city', 'amenities', 'calendar_updated', 'cancellation_policy',
                # Remove some data based on correlation
                'host_total_listings_count']
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
processed_listings = specific_areas_listings.replace({'price': {'\$': '', ',': '', '.00': ''},
                                                      'security_deposit': {'\$': '', ',': '', '.00': ''},
                                                      'cleaning_fee': {'\$': '', ',': '', '.00': ''},
                                                      'extra_people': {'\$': '', ',': '', '.00': ''}
                                                      }, regex=True)

# Mapping
# cancellation_policy_map = {'modelate': 0, 'strict_14_with_grace_period': 1}
# processed_listings['cancellation_policy'].map(cancellation_policy_map)

bool_map = {'f': 0, 't': 1}
processed_listings['instant_bookable'].replace(bool_map, inplace=True)
processed_listings['is_business_travel_ready'].replace(bool_map, inplace=True)
processed_listings['host_has_profile_pic'].replace(bool_map, inplace=True)
processed_listings['host_identity_verified'].replace(bool_map, inplace=True)
processed_listings['host_is_superhost'].replace(bool_map, inplace=True)
processed_listings['is_location_exact'].replace(bool_map, inplace=True)
processed_listings['has_availability'].replace(bool_map, inplace=True)

area_map = {'Shinjuku Ku': 0, 'Taito Ku': 1, 'Toshima Ku': 2, 'Shibuya Ku': 3, 'Minato Ku': 4}
processed_listings['neighbourhood_cleansed'].replace(area_map, inplace=True)

property_n_room_type_map = {'Apartment': 0, 'House': 1, 'Serviced apartment': 2, 'Condominium': 3, 'Villa': 4, 'Hostel': 5,
                     'Aparthotel': 6, 'Ryokan (Japan)': 7, 'Hotel': 8, 'Boutique hotel': 9, 'Guest suite': 10,
                     'Guesthouse': 11, 'Cabin': 12, 'Other': 13, 'Townhouse': 14, 'Loft': 15, 'Dorm': 16,
                     'Tiny house': 17, 'Bungalow': 18, 'Hut': 19, 'Camper/RV': 20, 'Bed and breakfast': 21,
                     'Entire home/apt': 22, 'Private room': 23, 'Shared room': 24}
processed_listings['property_type'].replace(property_n_room_type_map, inplace=True)
processed_listings['room_type'].replace(property_n_room_type_map, inplace=True)

bed_type_map = {'Real Bed': 0, 'Futon': 1, 'Pull-out Sofa': 2}
processed_listings['bed_type'].replace(bed_type_map, inplace=True)

# Make training set and test set
processed_listings = processed_listings.apply(pd.to_numeric)
train_set, test_set = train_test_split(processed_listings, test_size=0.25)

# Visualize correlation by using heatmap
# colormap = plt.cm.RdBu
# heat_map = sns.heatmap(train_set.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white')
# plt.show()

print(train_set.corr()['price'])

