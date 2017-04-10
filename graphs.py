import pandas
from matplotlib import pyplot as plt


df = pandas.read_csv("data/analysis.csv")

# StateHoliday
fig, ax = plt.subplots()
dfg = df.groupby(['StateHoliday']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='StateHoliday',
       ylabel='Sales',
       title='Sales by StateHoliday')
plt.show()

# Promo2
fig, ax = plt.subplots()
dfg = df.groupby(['Promo2']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='Promo2',
       ylabel='Sales',
       title='Sales by Promo2')
plt.show()

# IsPromoMonth
fig, ax = plt.subplots()
dfg = df.groupby(['IsPromoMonth']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='IsPromoMonth',
       ylabel='Sales',
       title='Sales by IsPromoMonth')
plt.show()

# SchoolHoliday
fig, ax = plt.subplots()
dfg = df.groupby(['SchoolHoliday']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='SchoolHoliday',
       ylabel='Sales',
       title='Sales by SchoolHoliday')
plt.show()

# Promo
fig, ax = plt.subplots()
dfg = df.groupby(['Promo']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='Promo',
       ylabel='Sales',
       title='Sales by Promo')
plt.show()

# Assortment
fig, ax = plt.subplots()
dfg = df.groupby(['Assortment']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='Assortment',
       ylabel='Sales',
       title='Sales by Assortment')
plt.show()

# StoreType
fig, ax = plt.subplots()
dfg = df.groupby(['StoreType']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='StoreType',
       ylabel='Sales',
       title='Sales by StoreType')
plt.show()

# State
fig, ax = plt.subplots()
dfg = df.groupby(['State']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='State',
       ylabel='Sales',
       title='Sales by State')
plt.show()

# DayOfWeek
fig, ax = plt.subplots()
dfg = df.groupby(['DayOfWeek']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='DayOfWeek',
       ylabel='Sales',
       title='Sales by DayOfWeek')
ax.set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.show()

# Month
fig, ax = plt.subplots()
dfg = df.groupby(['Month']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='Month',
       ylabel='Sales',
       title='Sales by Month')
plt.show()

# CompetitionDistance
fig, ax = plt.subplots()
dfg = df.groupby(['CompetitionDistance']).agg({'Sales': 'mean'})
dfg.plot(ax=ax)
ax.set(xlabel='CompetitionDistance',
       ylabel='Sales',
       title='Sales by CompetitionDistance')
plt.show()

# WeekOfYear
fig, ax = plt.subplots()
dfg = df.groupby(['WeekOfYear']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='WeekOfYear',
       ylabel='Sales',
       title='Sales by WeekOfYear')
plt.show()

# PromoSince
fig, ax = plt.subplots()
dfg = df.groupby(['PromoSince']).agg({'Sales': 'mean'})
dfg.plot(ax=ax)
ax.set(xlabel='PromoSince',
       ylabel='Sales',
       title='Sales by PromoSince')
plt.show()

# Day
fig, ax = plt.subplots()
dfg = df.groupby(['Day']).agg({'Sales': 'mean'})
dfg.plot.bar(ax=ax)
ax.set(xlabel='Day',
       ylabel='Sales',
       title='Sales by Day')
plt.show()

# CompetitionSince
fig, ax = plt.subplots()
dfg = df.groupby(['CompetitionSince']).agg({'Sales': 'mean'})
dfg.plot.hist(ax=ax)
ax.set(xlabel='CompetitionSince',
       ylabel='Sales',
       title='Sales by CompetitionSince')
plt.show()

# StateHoliday, SchoolHoliday
fig, ax = plt.subplots()
dfg = df.groupby(['StateHoliday', 'SchoolHoliday']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='SchoolHoliday', columns='StateHoliday', values='Sales').plot.bar(ax=ax)
ax.set(xlabel='SchoolHoliday',
       ylabel='Sales',
       title='Sales by SchoolHoldiay and StateHoliday')
plt.show()

# State, StateHoliday
fig, ax = plt.subplots()
dfg = df.groupby(['State', 'StateHoliday']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='State', columns='StateHoliday', values='Sales').plot.bar(ax=ax)
ax.set(xlabel='State',
       ylabel='Sales',
       title='Sales by State and StateHoliday')
plt.show()

# Promo, Promo2
fig, ax = plt.subplots()
dfg = df.groupby(['Promo', 'Promo2']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='Promo', columns='Promo2', values='Sales').plot.bar(ax=ax)
ax.set(xlabel='Promo',
       ylabel='Sales',
       title='Sales by Promo and Promo2')
plt.show()

# Promo, IsPromoMonth
fig, ax = plt.subplots()
dfg = df.groupby(['Promo', 'IsPromoMonth']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='Promo', columns='IsPromoMonth', values='Sales').plot.bar(ax=ax)
ax.set(xlabel='Promo',
       ylabel='Sales',
       title='Sales by Promo and IsPromoMonth')
plt.show()

# PromoSince, Promo
fig, ax = plt.subplots()
dfg = df.groupby(['PromoSince', 'Promo']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='PromoSince', columns='Promo', values='Sales').plot(ax=ax)
ax.set(xlabel='PromoSince',
       ylabel='Sales',
       title='Sales by PromoSince and Promo')
plt.show()

# PromoSince, Promo2
fig, ax = plt.subplots()
dfg = df.groupby(['PromoSince', 'Promo2']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='PromoSince', columns='Promo2', values='Sales').plot(ax=ax)
ax.set(xlabel='PromoSince',
       ylabel='Sales',
       title='Sales by PromoSince and Promo2')
plt.show()

# PromoSince, IsPromoMonth
fig, ax = plt.subplots()
dfg = df.groupby(['PromoSince', 'IsPromoMonth']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='PromoSince', columns='IsPromoMonth', values='Sales').plot(ax=ax)
ax.set(xlabel='PromoSince',
       ylabel='Sales',
       title='Sales by PromoSince and IsPromoMonth')
plt.show()

# StoreType, Assortment
fig, ax = plt.subplots()
dfg = df.groupby(['StoreType', 'Assortment']).agg({'Sales': 'mean'})
dfg.reset_index().pivot(index='StoreType', columns='Assortment', values='Sales').plot.bar(ax=ax)
ax.set(xlabel='Store Type',
       ylabel='Sales',
       title='Sales by Store Type and Assortment')
plt.show()
