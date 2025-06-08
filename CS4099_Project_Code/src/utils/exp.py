import os
import matplotlib.pyplot as plt
import numpy as np


all_classes = ['D.c', 'Pm', 'Me&Mm', 'M.g', 'O.g', 'E.p', 'P.n&O.n','N.l','Lp','Mo','Pl']
x = [615, 201, 1519, 1741, 245, 573, 1733, 1986, 685, 287, 665]

# Plot stacked bar chart.
plt.figure(figsize=(12, 6))
plt.bar(all_classes, x, color='skyblue', label='Training Data')
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Number of Images per Class in Training Data")
plt.legend()
plt.tight_layout()
plt.savefig('dataset_info2.png')