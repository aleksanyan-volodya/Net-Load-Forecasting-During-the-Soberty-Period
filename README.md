# Overview

Electricity demand forecasting is crucial to reduce the operational costs of power providers, especially when relying on intermittent sustainable energy sources.

Both demand and supply characteristics evolve over time. On the demand side, unexpected events as well as longer-term changes in consumption habits affect demand patterns. Recently, massive savings have been observed in Europe, following an unprecedented rise of energy prices.

On the production side, the increasing penetration of intermittent power generation significantly changes the forecasting needs.

This challenge addresses this problem. You will need to develop adaptive forecasting tools to adapt to these changes. 

# Data Description
The data correspond to french electricity load from 20212 to january 2021 at a daily resolution (mean per day) in MW. We provide a train set from january 2012 to august 2019 and a test set from september 2019 to january 2021. The goal of the challenge is to forecast the target "Load".
File descriptions:

    train.csv - the training set from january 2012 to april 2020
    test.csv - the test set from april 2020 to january 2021.
    sampleSubmission.csv - a sample submission file in the correct format

Data fields

    Date Date , format YYYY-mm-dd,
    Net_demand - the net demand in MW
    Net_demand.1 - lag one day of net demand
    Net_demand.7 - lag 7 days of net demand
    Solar_power - the solar production in MW
    Solar_power.1 - lag one day of solar production
    Solar_power.7 - lag 7 days of solar production
    Wind_power - the wind production in MW
    Wind_power.1 - lag one day of wind production
    Wind_power.7 - lag 7 days of wind production
    Load - the electricity load in MW
    Load.1 - lag one day of electricity load
    Load.7 - lag 7 days of electricity load
    Temp - mean temperature. over 39 stations of France in celsius degrees
    Temp_s95 - smooth temperature with 0.95 smoothing parameter
    Temp_s99 - smooth temperature with 0.99 smoothing parameter
    Temp_s95_min - daily min of the smoothed temperature with 0.95 smoothing parameter
    toy - time. of year from 0 to 1 each year
    WeekDays - day of the week, 5: Saturday, 6: Sunday, 0: Monday...
    BH_before - if the day is before a bank holidays
    BH - bank holidays
    BH_after - if the day is after a bank holidays
    DLS - daylight savings
    Summer_break - summer holidays period
    Winter_break - winter holidays period
    Holiday -school holidays period
    Holiday_zone_a -school holidays period for zone a
    Holiday_zone_b -school holidays period for zone b
    Holiday_zone_c -school holidays period for zone c
    BH_Holiday -bank holidays during holidays

## Week 1:
...


## Week 2:
    Nous allons utiliser pinball loss. 
    On va chercher l'argmin par rapport à q de la médiande de loss. E[loss(Y-q)]