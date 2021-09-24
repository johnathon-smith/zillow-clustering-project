import numpy as np
import pandas as pd
import env
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

#Plot the individual distributions
def get_dists(df):
    for col in df.columns:
        sns.histplot(x = col, data = df)
        plt.title(col)
        plt.show()

#The following function will remove columns and rows based on the proportion of missing data
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    #function that will drop rows or columns based on the percent of values that are missing:\
    #handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def get_zillow():
    #Write sql query for zillow data
    zillow_query = """
        SELECT prop.*,
        pred.logerror,
        pred.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM   properties_2017 prop
        INNER JOIN (SELECT parcelid,
                    Max(transactiondate) transactiondate
                    FROM   predictions_2017
                    GROUP  BY parcelid) pred
                USING (parcelid)
                            JOIN predictions_2017 as pred USING (parcelid, transactiondate)
        LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
        LEFT JOIN storytype story USING (storytypeid)
        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE  prop.latitude IS NOT NULL
        AND prop.longitude IS NOT NULL
    """

    #Write zillow url to access the codeup database
    zillow_url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

    file_name = 'zillow.csv'

    if os.path.isfile(file_name):
        return pd.read_csv(file_name)

    else:
        df = pd.read_sql(zillow_query, zillow_url)
        df.to_csv(file_name, index = False)
        
        return df

#The following function will acquire and prepare a general zillow dataframe
def wrangle_zillow():
    #Acquire initial data
    zillow = get_zillow()

    #Begin preparing
    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    zillow = zillow[zillow.propertylandusetypeid.isin(single_use)]

    #Now remove things that don't make sense and/or are impossible/illegal.
    #If something doesn't sound like the average 'single family residential' property, drop it.
    zillow = zillow[(zillow.bedroomcnt > 0) & (zillow.bathroomcnt > 0)]
    zillow = zillow[zillow.calculatedfinishedsquarefeet <= 8000]

    #There are minimum size laws for single unit homes. 
    #Although these change from state to state and county to county,
    #A good rule of thumb is 120sqft per bedroom.
    zillow = zillow[zillow.calculatedfinishedsquarefeet >= (120 * zillow.bedroomcnt)]

    #If the tax amount owed is outrageous, then its probably either wrong or an outlier.
    zillow = zillow[zillow.taxamount <= 20_000]

    #Also check for properties that are priced significantly higher than normal.
    zillow = zillow[zillow.taxvaluedollarcnt < 5_000_000]

    #Now handle missing values.
    zillow = handle_missing_values(zillow)

    # Add column for counties
    zillow['county'] = zillow['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

    # drop unnecessary columns
    dropcols = ['parcelid',
        'id',
        'calculatedbathnbr',
        'finishedsquarefeet12',
        'fullbathcnt',
        'heatingorsystemtypeid',
        'propertycountylandusecode',
        'propertylandusetypeid',
        'propertyzoningdesc',
        'censustractandblock',
        'propertylandusedesc',
        'rawcensustractandblock',
        'unitcnt',
        'transactiondate',
        'assessmentyear']

    zillow = zillow.drop(columns = dropcols)

    # assume that since this is Southern CA, null means 'None' for heating system
    zillow.heatingorsystemdesc.fillna('None', inplace = True)

    # replace nulls with median values for select columns
    zillow.lotsizesquarefeet.fillna(7313, inplace = True)
    zillow.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Just to be sure we caught all nulls, drop them here
    zillow = zillow.dropna()

    #Create a few new features and remove features that are related.
    zillow['age'] = 2017 - zillow.yearbuilt

    #acres: lotsizesquarefeet/43560
    zillow['acres'] = zillow.lotsizesquarefeet / 43_560

    #tax_rate: taxamount/taxvaluedollarcnt fields (total, land & structure). 
    #We can then remove taxamount and taxvaluedollarcnt, 
    #and will keep taxrate, tructuretaxvaluedollarcnt, and landtaxvalue.
    zillow['tax_rate'] = zillow.taxamount / zillow.taxvaluedollarcnt

    # dollar per square foot-structure
    zillow['structure_dollar_per_sqft'] = zillow.structuretaxvaluedollarcnt/zillow.calculatedfinishedsquarefeet

    # dollar per square foot - land
    zillow['land_dollar_per_sqft'] = zillow.landtaxvaluedollarcnt/zillow.lotsizesquarefeet

    # ratio of bathrooms to bedrooms
    zillow['bath_bed_ratio'] = zillow.bathroomcnt/zillow.bedroomcnt

    #Now drop all of the related variables
    zillow = zillow.drop(columns = ['taxamount',
                                    'taxvaluedollarcnt',
                                    'yearbuilt',
                                    'fips',
                                    'bedroomcnt',
                                    'structuretaxvaluedollarcnt',
                                    'landtaxvaluedollarcnt',
                                    'lotsizesquarefeet',
                                    'regionidzip',
                                    'regionidcounty',
                                    'regionidcity'])

    #Rename columns
    zillow.rename(columns = 
                {'bathroomcnt':'bathroom_count',
                'buildingqualitytypeid':'quality_type',
                'calculatedfinishedsquarefeet':'home_square_feet',
                'roomcnt':'room_count',
                'heatingorsystemdesc':'heating_system_desc'}, inplace = True)

    return zillow

#The following function will split a df into train, validate, and test splits.
def train_validate_test_split(df, seed = 123):
    '''
    This function takes in a dataframe and an integer for setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''

    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

#The following function will create dummy variables and split a df into train, validate, test sets for modeling
def get_dummy_vars_and_split(df):
    #Get cols to create dummies for
    cat_cols = df.select_dtypes('object').columns
    
    df_dummies = pd.get_dummies(df[cat_cols], dummy_na=False, drop_first=True)
    df = pd.concat([df, df_dummies], axis = 1).drop(columns = cat_cols)

    train, validate, test = train_validate_test_split(df)

    return train, validate, test