from typing import Tuple

import pandas as pd

from sklearn import  ensemble


def thousend(value):
    value = value * 1000
    value = value.astype(int)
    return value

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:

    temperature = temperature[temperature['serialNumber'] == serial_number_for_prediction]
    temperature = temperature.loc[:,['value']]
    temperature.columns = ['temp']

    target_temperature = target_temperature.loc[:,['value']]
    target_temperature.columns = ['target_temp']

    valve_level = valve_level.loc[:,['value']]
    valve_level.columns = ['valve']

    df_combined = pd.concat([temperature, target_temperature,valve_level])
    df_combined = df_combined.resample(pd.Timedelta(minutes=15),label='right').mean().fillna(method='ffill')
    df_combined.dropna(inplace=True)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20.34)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value=20.34)

    x_train = df_combined[['temp','valve']].to_numpy()[1:-1]
    y_train= df_combined['temp_gt'].to_numpy()[1:-1]

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(x_train,y_train)


    df_test = df_combined.tail(1)

    x_test = df_test[['temp', 'valve']].to_numpy()

    temp_predicted= reg_rf.predict(x_test)

    y_train = df_combined['valve_gt'].to_numpy()[1:-1]

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(x_train, y_train)

    df_test = df_combined.tail(3)

    x_test = [df_test[['temp', 'valve']].mean().to_numpy()]
    valve_predicted = reg_rf.predict(x_test)

    result = (temp_predicted,valve_predicted)

    return result
