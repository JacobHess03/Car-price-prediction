Vehicle Price Prediction

This repository contains the code and resources for the Vehicle Price Prediction competition. The goal of this competition is to accurately predict vehicle prices based on various characteristics.
Evaluation Criterion

The primary evaluation criterion for this competition is the Mean Absolute Error (MAE). MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It is calculated as the mean of the absolute differences between the predicted values and the actual values.

    MAE=n1​i=1∑n​∣yi​−y^​i​∣

Where:

    n is the number of data points
    yi​ is the actual price
    y^​i​ is the predicted price

Submission Format

Your submission file should be a CSV with two columns:

    Vehicle ID: Corresponds to the id column in test.csv.
    Predicted Price: Your model's predicted price for each vehicle.

Data

The dataset used for this competition was collected from the auto.am website. Each vehicle entry includes 8 characteristic columns that can be used for prediction:

    Year: Manufacturing year of the vehicle.
    Motor type: Type of engine (e.g., gasoline, diesel).
    Mileage: Distance covered by the vehicle.
    Wheel: Wheel drive type (e.g., front, rear, all-wheel).
    Color: Exterior color of the vehicle.
    Car_type: Body type of the vehicle (e.g., sedan, SUV).
    Status: Condition or registration status of the vehicle.
    Motor volume: Engine displacement in liters.

Car Models

The dataset primarily features the following five car models:

    Toyota Camry
    Mercedes-Benz C-Class
    Hyundai Elantra
    Nissan Rogue
    Kia Forte
