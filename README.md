# Iowa House-Prices
*Price prediction model using regression with regularization and Machine learning techniques.*

###Regression Summary
***
This model, written in Python, predicts house prices.  The regression model was tuned using **cross-validation** (CV) with RMSE and R^2.  Cross validation returned **Root Mean Squared Error** (RMSE) for each fold.  The RMSsE can be inspected and the standard deviation of the RMSEs taken to assess model fit quality and model stability.

**R^2**, the coefficient of determination, explains what portion of house prices are explained by the features (the independent variables).

Good models need a reality-check to make sure the model generalizes to unseen data, which we call **holdout** data.  After fitting/scoring the model with CV, I also scored model predictions on the holdout data.

The best model was Ridge, which was unsurprising as model dimensionality was high compared to available observations.  Ridge reduced the complexity of the model by regularizing the features.  It does this by penalizing some features with low weights, decreasing model complexity.

###Feature Engineering

<ins>**Normalizing the prediction target, house price:**</ins>


![Alt text](images/Target_engineering_price.PNG)
