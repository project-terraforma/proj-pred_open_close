import geopandas as gpd
import pandas as pd

def extract_primary_name(names):
    try:
        if isinstance(names, dict):
            return names.get('primary', '') or ''
        return ''
    except:
        return ''

cities = {
    'los_angeles': 'la_places.parquet',
    'san_francisco': 'sf_places.parquet',
    'santa_cruz': 'sc_places.parquet',
}

frames = []
for city, filepath in cities.items():
    print(f"Loading {city}...")
    df = gpd.read_parquet(filepath)
    df['city'] = city
    df['primary_name'] = df['names'].apply(extract_primary_name)

    # Filter closed by operating_status OR name signal
    name_upper = df['primary_name'].str.upper()
    closed_mask = (
        (df['operating_status'] == 'closed') |
        name_upper.str.contains(r'\bCLOSED\b|PERMANENTLY CLOSED|TEMPORARILY CLOSED', regex=True, na=False) |
        name_upper.str.contains(r'\bVACANT\b|\bFORMERLY\b', regex=True, na=False)
    )

    closed_df = df[closed_mask].copy()
    print(f"  Total: {len(df)}, Closed: {len(closed_df)}")
    frames.append(closed_df)

# Combine all cities
combined = pd.concat(frames, ignore_index=True)
print(f"\nTotal closed records across all cities: {len(combined)}")
print(combined['city'].value_counts())

# Save
combined.to_parquet("closed_places_all_cities.parquet", index=False)
print("\nSaved: closed_places_all_cities.parquet")
