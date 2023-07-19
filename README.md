# Bayes Classifier
Conducting **one-vs-all** classification on the '[penguins.csv](https://github.com/fardinabbasi/Bayes_Classifier/blob/main/penguins.csv)' dataset using both the **Naive Bayes classifier** implemented **from scratch** and the one provided by **Scikit-Learn's built-in** functions.

In the Bayes classifier for L number of features, $N^L$ data points are required to determine the **decision boundary**, which can be both **costly** and **time-consuming**. On the contrary, **Naive Bayes** assumes that features are independent and **identically distributed (i.i.d)** to each other, thereby reducing the number of required data points to $N*L$. Although this approach may slightly reduce overall precision, it proves to be highly useful in scenarios where data gathering can be a challenging task.
### From Scratch
| Result | Adelie vs. All | Gentoo vs. All | Chinstrap vs. All |
| --- | --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/aa1.png"> | <img src="/readme_images/gg1.png"> | <img src="/readme_images/cc1.png"> |
| Classification Report | <img src="/readme_images/a1.jpg"> | <img src="/readme_images/g1.jpg"> | <img src="/readme_images/c1.jpg"> |


### Scikit-Learn
| Result | Adelie vs. All | Gentoo vs. All | Chinstrap vs. All |
| --- | --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/aa2.png"> | <img src="/readme_images/gg2.png"> | <img src="/readme_images/cc2.png"> |
| Classification Report | <img src="/readme_images/a2.jpg"> | <img src="/readme_images/g2.jpg"> | <img src="/readme_images/c2.jpg"> |

<h3> Q5: Gaussian Naive Bayes Classifier </h3>
