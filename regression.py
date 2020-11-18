import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    ...
    text_data = response.text
    areas = text_data.split('\n')[0]
    prices = text_data.split('\n')[1]
    areas = list(map(float, areas.split(',')[1:]))
    prices = list(map(float, prices.split(',')[1:]))
    data = [[areas[i], prices[i]] for i in range(len(areas))]
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(areas, prices)
    print(slope)
    print(intercept)
    print(r_value)
    print(p_value)
    print(std_err)
    res = [ area[i]*slope + intercept for i in range(len(area)) ]
    return res


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
