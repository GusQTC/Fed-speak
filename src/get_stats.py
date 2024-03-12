from fredapi import Fred
import pandas as pd

FRED_API_KEY= "55ddb1dfa49988e7cd6ecaa8095a5143"

fred = Fred(api_key=FRED_API_KEY)

interest_rate = fred.get_series('FEDFUNDS', observation_start='01/01/2019', )

GDP = fred.get_series('GDP', observation_start='01/01/2019')

CPI = fred.get_series('CPIAUCSL', observation_start='01/01/2019')

Employment = fred.get_series('PAYEMS', observation_start='01/01/2019')


data = pd.DataFrame({'Interest Rate': interest_rate, 'GDP': GDP, 'CPI': CPI, 'Employment': Employment})

with open('economic_data.csv', 'w') as f:
    data.to_csv(f)