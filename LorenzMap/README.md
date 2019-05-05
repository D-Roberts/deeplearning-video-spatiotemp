### End to end time series prediction gluon code for the article:

'Lorenz Trajectories Prediction: Travel Through Time'.
https://arxiv.org/abs/1903.07768

Plotting two of the three trajectories (z vs x) gives rise to the Lorenz butterfly.

![Lorenz_butterfly](assets/Lorenz_butterfly.png)

### Training and inference

Conditional Wavenet architecture runs in 24 seconds end to end (CPU) and achieves on average the test RMSE reported in https://arxiv.org/abs/1703.04691): 

###### default setting
```
python main.py
``` 

### Learning for Conditional Wavenet x series one step ahead prediction:

![losses_cw](assets/losses_cw.png)

### Predictions vs ground truth for x trajectory:

![preds_cwn](assets/preds_cwn.png)
