# Iowa House-Prices
*Price prediction model using regression with regularization and Machine learning techniques.*

#### Regression Summary
***
This model, written in Python, predicts house prices.  The regression model was tuned using **cross-validation** (CV) with RMSE and R^2.  Cross validation returned **Root Mean Squared Error** (RMSE) for each fold.  The RMSE can be inspected and the standard deviation of the RMSEs taken to assess model fit quality and model stability.

**R^2**, the coefficient of determination, explains what portion of house prices are explained by the features (the independent variables).

Good models need a reality-check to make sure the model generalizes to unseen data, which we call **holdout** data.  After fitting/scoring the model with CV, I also scored model predictions on the holdout data.

The best model was Ridge, which was unsurprising as model dimensionality was high compared to available observations.  Ridge reduced the complexity of the model by regularizing the features.  It does this by penalizing some features with low weights, decreasing model complexity.

#### Feature Engineering

A great deal of thought is required to provide good features to regression models.  Skewed data, outliers, data on different scales, unencoded data, missing data, and collinear data are part of the feature engineering challenge.

It turns out that variables with a Guassian, or bell curve shaped distribution, work better in regression models.  This includes not only the features.  It applies to our prediction target house sales price also.

Below, the house price is normalized to Guassian.

<ins>**Normalizing the prediction target, house price:**</ins>

![Alt text](images/Target_engineering_price.PNG)

Missing data was cleaned on a case-by-case basis after thorough exploration.  For example, missing KitchenQual was replaced with the most common quality, 'TA'.  KitchenQual was presented as a categorical variable, but it is actually ordinal--the categories have a meaningful order.  It was updated as follows:

* **Before:** `[Ex, Gd, TA, Fa, Po, None]`
* **After:**  `[5, 4, 3, 2, 1]`

Outlier house sizes were dropped.

<ins>**Outlier chart:**</ins>

![Alt text](images/outliers-TOT_SF.PNG)


