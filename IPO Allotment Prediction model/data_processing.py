# data_processing.py

import pandas as pd
import numpy as np
from fuzzywuzzy import process

def load_data(ipo_path, cleaned_ipo_path):
    """Loads the two datasets."""
    try:
        ipo = pd.read_csv(ipo_path, header=1)
        cleaned_ipo = pd.read_csv(cleaned_ipo_path)
        return ipo, cleaned_ipo
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def fuzzy_merge(df1, df2, key1, key2, threshold=80):
    """
    Merges two dataframes based on a fuzzy match of string columns.
    """
    matches = []#Initializes an empty list
    for name in df1[key1]:  #Starts a loop that iterates through every value (IPO name) in the key1 column of df1.
        try:
            result = process.extractOne(name, df2[key2]) #Performs the fuzzy match. process.extractOne finds the single best match for the current name from the list of names in df2[key2]. 
            if result and result[1] >= threshold:#Checks if a match was found (result is not None) AND if the matching score (result[1], the second element of the tuple) is greater than or equal to the specified threshold (80).
                matches.append(result[0])#If the match is good, the actual matched name (result[0]) is appended to the matches list.
            else:
                matches.append(None)#indicating no suitable fuzzy match was found.
        except:
            matches.append(None)
    df1['matched_name'] = matches
    return df1.merge(df2, left_on='matched_name', right_on=key2, how='inner') #Performs the final merge. It merges df1 and df2 using a standard pandas inner join.

def create_features(df): #takes a single DataFrame (df) and adds new features and the target variable.
    """Creates features for the model."""
    # Target Variable (Binary)
    df['Target'] = ((df['Listing Gains(%)'].fillna(0) > 0) | (df['Returns'].fillna(0) > 0)).astype(int)

    # Log Transform Skewed Features
    if 'Issue Size  (in crores)' in df.columns:
        df['Log_Issue_Size'] = np.log1p(df['Issue Size  (in crores)'])

    # Investor Category (Categorical)
    df['Investor_Category'] = pd.cut(
        df['RII'],
        bins=[-np.inf, 50, 100, np.inf],
        labels=['Retail', 'HNI', 'Institutional']
    )
    
    # --- Add Placeholders for New, More Powerful Features ---
    # These columns are added so you can populate them with real data later.
    # The model will not improve until this data is added.
    df['gmp_percentage'] = 0.0  # Grey Market Premium as a percentage of issue price
    df['revenue_crores'] = 0.0  # Company's latest annual revenue
    df['profit_margin'] = 0.0    # Company's net profit margin

    return df

def main():
    """Main function to run the data processing pipeline."""
    ipo_path = "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/ipo.csv"
    cleaned_ipo_path = "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/cleaned_ipo_data 2022-25.csv"
    
    ipo_df, cleaned_ipo_df = load_data(ipo_path, cleaned_ipo_path)
    if ipo_df is None:
        return

    print("Merging datasets...")
    combined_df = fuzzy_merge(ipo_df, cleaned_ipo_df, 'IPO Name', 'Name')
    print("Merge complete.")

    print("Creating features...")
    featured_df = create_features(combined_df)
    print("Feature creation complete.")

    # Select final columns
    final_columns = [
        'Log_Issue_Size', 'Investor_Category', 'Target',
        'gmp_percentage', 'revenue_crores', 'profit_margin', 'Inverse_Subscription'
    ]
    # Ensure all columns exist before selecting
    final_columns = [col for col in final_columns if col in featured_df.columns]
    final_df = featured_df[final_columns]

    # Save the processed data
    output_path = "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/featured_ipo_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved processed data to {output_path}")

if __name__ == "__main__":
    main()
