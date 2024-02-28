import argparse
import numpy as np 
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt 
import os

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('cert_npz_path', type=str,)
args = parser.parse_args()

counts = np.load(args.cert_npz_path)
y = counts['a']
x = counts['b']
emp_loss = counts['c']
plt.plot(x,emp_loss,'--',linewidth=2.0, color=sns.color_palette("colorblind")[0])
plt.plot(x,y,'--',linewidth=2.0, color=sns.color_palette("colorblind")[1])
print(y-emp_loss)
a = plt.xlim([0,1.501])
a = plt.yticks(
    np.arange(0,1.01,0.05), 
    labels=[f'{i:2.1f}' if n % 2 == 0 else "" for n, i in enumerate(np.arange(0,1.01,0.05))], 
    fontsize=9
)
a = plt.xticks(
    np.arange(0,1.51,0.1), 
    labels=[f'{i:2.1f}' if n % 2 == 0 else "" for n, i in enumerate(np.arange(0,1.51,0.1))], 
    fontsize=9
)
plt.legend(['training data adversarial certificate','adversarial risk bound'])
plt.xlabel("R")
plt.ylabel("Certified 0-1 loss")
plt.savefig(os.path.join(os.path.dirname(args.cert_npz_path), 'plot.pdf'))
