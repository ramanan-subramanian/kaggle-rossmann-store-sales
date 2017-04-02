"""
Kaggle = 0.10334
"""
import numpy as np
import pandas
import pickle
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot
def process_data(data):
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), "Open"] = 1

    category = {"0": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    data.StoreType.replace(category, inplace=True)
    data.Assortment.replace(category, inplace=True)
    data.StateHoliday.replace(category, inplace=True)

    data.DayOfWeek = data.Date.dt.dayofweek
    data["Year"] = data.Date.dt.year
    data["Month"] = data.Date.dt.month
    data["Day"] = data.Date.dt.day
    data["WeekOfYear"] = data.Date.dt.weekofyear
    data = data.drop(["Date"], axis=1)

    data.CompetitionDistance = np.log1p(data.CompetitionDistance)

    data["CompetitionSince"] = \
        12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    data = data.drop(["CompetitionOpenSinceMonth",
                      "CompetitionOpenSinceYear"], axis=1)

    data["PromoSince"] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data["PromoSince"] = data.PromoSince.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, "PromoSince"] = 0
    data = data.drop(["Year", "Promo2SinceWeek", "Promo2SinceYear"], axis=1)

    month_string = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov",
                    12: "Dec"}
    data["MonthString"] = data.Month.map(month_string)
    data.loc[data.PromoInterval == 0, "PromoInterval"] = ""
    data["IsPromoMonth"] = 0
    for interval in data.PromoInterval.unique():
        if interval != "":
            for month in interval.split(","):
                data.loc[(data.MonthString == month) &
                         (data.PromoInterval == interval), "IsPromoMonth"] = 1
    data = data.drop(["MonthString", "PromoInterval"], axis=1)

    states = {"HE": 0, "TH": 1, "NW": 2, "BE": 3, "SN": 4, "SH": 5, "HB,NI": 6,
              "BY": 7, "BW": 8, "RP": 9, "ST": 10, "HH": 11}
    data.State.replace(states, inplace=True)

    return data


def process_average_sales(data, average_sales):
    data["AverageSales"] = 0
    for i in range(1, 1116):
        mean = data[data.Store == i].Sales.mean()
        mean = np.log1p(mean)
        data.loc[data.Store == i, "AverageSales"] = mean
        average_sales.append(mean)
    return data


def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat)/y) ** 2))


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


dtype = {"Store": "int", "DayOfWeek": "int", "Date": "str", "Customers": "int",
         "Open": "int", "Promo": "int", "StateHoliday": "str",
         "SchoolHoliday": "int", "Sales": "int"}

print("Load and process datasets.")
store = pandas.read_csv("data/store.csv")
state = pandas.read_csv("data/state.csv")

train = pandas.read_csv("data/train.csv", parse_dates=[2], dtype=dtype)
train = train.drop(["Customers"], axis=1)
train = train[train["Open"] != 0]
train = train[train["Sales"] > 0]
train = train.merge(store, on="Store")
train = train.merge(state, on="Store")
train = process_data(train)
train = train.drop(["Open"], axis=1)

test = pandas.read_csv("data/test.csv", parse_dates=[3])
test.fillna(1, inplace=True)
test = test.merge(store, on="Store")
test = test.merge(state, on="Store")
test = process_data(test)
test = test.drop(["Open"], axis=1)

average_sales = []
train = process_average_sales(train, average_sales)
test["AverageSales"] = 0
for i in range(1, 1116):
    test.loc[test.Store == i, "AverageSales"] = average_sales[i-1]

X_test = test.drop(["Id"], axis=1)
dtest = xgboost.DMatrix(X_test)

print("Loading model.")
model = pickle.load(open("model.dat", "rb"))

plot_importance(model)
pyplot.show()

print("Making predictions on the test set.")
predictions = model.predict(dtest)

print("Saving predictions to csv file for submission.")
result = pandas.DataFrame({"Id": test["Id"], "Sales": np.expm1(predictions)})
result.to_csv("Submission.csv", index=False)

print("Done")
