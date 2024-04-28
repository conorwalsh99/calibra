# calibra

Calibra is a python package for evaluating the calibration of machine learning models. It serves to help researchers and practitioners to evaluate the reliability of their models, by comparing expected and observed rates of occurence.

The calibration of a model is a measure of how reliable its predictions are. If a model predicts an outcome has an 80% probability of occurring, one would expect that outcome to occur approximately 80% of the time. Calibra can help users identify whether or not this is the case, by quantifying just how much the true rate of occurrence deviates from the expected rate of occurrence. This quantity can be measured in several different ways. We use the class-wise Expected Calibration Error (ECE) as described by [Kull et. al [1]](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html).

In certain settings, modellers may be more interested in estimating the probability of certain outcomes, rather than simply predicting the most likely outcome, such as weather forecasting, diagnosis of disease, and sports betting. Under these circumstances, evaluating a model by its calibration may be more useful than evaluating it by accuracy alone.
    
Calibra provides a simple interface to allow users to calculate the model's class-wise expected calibration error, for any number of classes. It is highly compatible with scikit-learn and matplotlib.
To see how to use calibra, check out notebooks/examples.ipynb for a short tutorial. 


# Installation
## Dependencies
- Python (>=3.6)
- Pandas
- NumPy 
- Matplotlib

## User installation

To install calibra using pip, simply run:

`pip install calibra`

# Usage
The main use of calibra is the evaluation of the calibration of machine learning models. Users can easily calculate thge class-wise ECE of a set of predictions, as shown below.

```
from calibra.errors import classwise_ece

y_pred_proba = [0.76, 0.85, 0.43, 0.95, 0.33, 0.12, 0.56, 0.45, 0.89, 0.82]
y_true = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]

expected_calibration_error = classwise_ece(y_pred_proba, y_true) 
``` 

The class-wise ECE is given by the following equation: 

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7Bk%7D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%5Cfrac%7B%7CB_%7Bj%2Ci%7D%7C%7D%7Bn%7D%7Cy_i%28B_%7Bj%2Ci%7D%29-%5Cbar%7Bp%7D_%7Bi%7D%28B_%7Bj%2Ci%7D%29%7C) 

for the k-class problem with m bins. For each class, we weight each bin by the proportion of the dataset it contains, and get the absolute error between the expected and actual rate of occurrence (of instances of the given class) within each bin. We then sum these weighted deviations across all bins in the given class to get the error for that class, and get the average of these errors across all classes. 

By default the error is calculated for 20 bins of equal width, where the i_th bin is the interval [i/m, (i+1)/m), i.e. each interval includes its start boundary but not its end boundary, except for the final bin. 
These default values can be adjusted by changing the 'num_bins' and 'method' parameters.

Users can also visualise the calibration of their models by plotting calibration curves:

```
# y_pred_proba, y_true different to those in previous example

import matplotlib.pyplot as plt
from calibra.plotting import CalibrationCurve

calibration_curve = CalibrationCurve(y_pred_proba, y_true)
calibration_curve.plot(label='Random Forest Classifier')
plt.title('Calibration Curve')
plt.legend()
```

![output](random_forest_calibration_curve_example.png)

4. Features

Calibra users can:
1. Examine the distribution of their model's predictions by grouping these predictions into a specified number of bins
2. Calculate the class-wise Expected Calibration Error (ECE) of a set of predictions, to quantify the reliability of the model
3. Plot a calibration curve, to visualise the reliability of the model



5. Documentation
Link to the full project documentation if available. This could be hosted on sites like ReadTheDocs or a GitHub wiki.

6. Examples/Demo
Include more detailed examples or a link to a Jupyter notebook that demonstrates the module’s capabilities in action. This helps users see practical applications of your module.

7. Contributing
If you’re open to contributions from others, explain how they can contribute. Provide guidelines on submitting pull requests, reporting bugs, or suggesting enhancements.

8. License
Specify the license under which the module is released, so users know how they are permitted to use it and what restrictions apply.

9. Authors and Acknowledgment
List the authors or maintainers of the module. You might also want to acknowledge contributors or organizations that helped in the development.

10. Citations
If your module uses or refers to algorithms or data from papers, books, or other software, include a citations section to credit the original authors properly.

[1] [Kull, M., Perello Nieto, M., Kängsepp, M., Silva Filho, T., Song, H. and Flach, P., 2019. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. Advances in neural information processing systems, 32.](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)

11. Contact Information
Provide contact information or ways to reach the maintainer for support or further questions.
Each of these sections helps make your README comprehensive, ensuring users understand how to install, utilize, and contribute to your module effectively.







# TODO
1. Add documentation (clean up README.md, provide output for example classwise_ece, maybe plot curve for same example for consistency? Or ensure it refers to new example. coverage [and other] badges to Github)
2. Package (add poetry or requirements.txt for dependency management, ensure works under specific versions, bump version and republish to pypi)
GITHUB:
3. Add pre-commit hooks (unit tests pass, black style used etc.)
4. Think about open source collaboration strategy (pre-commit hooks, admin rights, PR strategies)
5. Publicise (Linkedin post into machine learning community, medium article)

6. BUG: when show_density=True, we are not calling plt.plot(**kwargs). Therefore we cannot dispay the labels directly in this case. 

