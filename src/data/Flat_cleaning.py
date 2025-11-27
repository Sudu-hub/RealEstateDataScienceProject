import numpy as np
import pandas as pd
import re
import os

def load_data(df):
    df.drop(columns=['link','property_id'], inplace=True)
    df.rename(columns={'area':'price_per_sqft'},inplace=True)
    df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    df = df[df['price'] != 'Price on Request']
    return df

def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)
        
def clean_function(df):
    df['price'] = df['price'].str.split(' ').apply(treat_price)
    df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
    df = df[~df['bedRoom'].isnull()]
    return df

def change_data_type(df):
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
    df['additionalRoom'].fillna('not available',inplace=True)
    df['additionalRoom'] = df['additionalRoom'].str.lower()
    df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')
    df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))
    df.insert(loc=1,column='property_type',value='flat')
    return df
    
def save_data(data_path,df):
    os.mkdir(data_path)
    df.to_csv(os.path.join(data_path, 'flat_cleaned.csv'))

def main():
    df = pd.read_csv('./data/raw/flats.csv')
    df = load_data(df)
    df = clean_function(df)
    df = change_data_type(df)
    data_path = os.path.join('data','flat_clean')
    save_data(data_path, df)

if __name__ == '__main__':
    main()