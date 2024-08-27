import pandas as pd
import json


df = pd.read_csv('new_values/result_economic_corrected.csv')
# Example NER results (simplified for demonstration)
ner_results = json.load(open('NER/year_month_entities.json'))

# Convert NER results to a DataFrame with entity counts by type
def ner_to_features(ner_results):
    feature_data = []
    for year, months in ner_results.items():
        for month, entities in months.items():
            # Initialize period data with Year and Month
            period_data = {'Year': int(year), 'Month': int(month)}
            
            # Dynamically update period data based on entities encountered
            for entity, count in entities.items():
                # Extract entity type from the entity string
                entity_type = entity.split(" - ")[-1]
                # If the entity type is not already in period_data, initialize it
                if entity_type not in period_data:
                    period_data[entity_type] = 0
                # Increment the count for this entity type
                period_data[entity_type] += count
            
            # Add the period's data to the list
            feature_data.append(period_data)
    return pd.DataFrame(feature_data)

# Convert NER results to features
ner_features_df = ner_to_features(ner_results)

df = pd.merge(df, ner_features_df, on=['Year', 'Month'], how='left')

df.to_csv('new_values/result_economic_corrected_ner.csv', index=False)


# Now `df` includes the NER-derived features and can be used for XGBoost analysis