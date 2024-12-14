import pyro
import numpy
import torch
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import model_selection
import pyro.distributions as pdist
import torch.distributions as tdist
import torch.nn as tnn
import pyro.nn as pnn
import arviz
from functools import partial

# some of the parameters to make it reproducible 
seed_value = 42  # Replace with your desired seed value
torch.manual_seed(seed_value)
pyro.set_rng_seed(seed_value)
numpy.random.seed(seed_value)


def accuracy(pred, data):
  """
  Calculate accuracy of predicted labels (integers).

  pred: predictions, tensor[sample_index, chain_index, data_index, logits]
  data: actual data (digit), tensor[data_index]

  Prediction is taken as most common predicted value.
  Returns accuracy (#correct/#total).
  """
  n=data.shape[0]
  correct=0
  total=0
  for i in range(0, n):
      # Get most common prediction value from logits
      pred_i=int(torch.argmax(torch.sum(pred[:,0,i,:],0)))
      # Compare prediction with data
      if int(data[i])==int(pred_i):
          correct+=1.0
      total+=1.0
  # Return fractional accuracy
  return correct/total

# load iris dataset 
# Iris data set
Dx=4 # Input vector dim
Dy=3 # Number of labels

iris=sklearn.datasets.load_iris()
x_all=torch.tensor(iris.data, dtype=torch.float) # Input vector (4D)
y_all=torch.tensor(iris.target, dtype=torch.int) # Label(3 classes)

# Make training and test set
x, x_test, y, y_test = sklearn.model_selection.train_test_split(
    x_all, y_all, test_size=0.33, random_state=42)

print("Data set / test set sizes: %i, %i." % (x.shape[0], x_test.shape[0]))

class Model:
    def __init__(self, x_dim=4, y_dim=3, h_dim=5):
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.h_dim=h_dim

    def __call__(self, x, y=None):
        """
        We need None for predictive
        """
        x_dim=self.x_dim
        y_dim=self.y_dim
        h_dim=self.h_dim
        # Number of observations
        n=x.shape[0]
        # standard deviation of Normals
        sd=1 
        # Layer 1
        w1=pyro.sample("w1", pdist.Normal(0, sd).expand([x_dim, h_dim]).to_event(2))
        b1=pyro.sample("b1", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 2 
        w2=pyro.sample("w2", pdist.Normal(0, sd).expand([h_dim, h_dim]).to_event(2))
        b2=pyro.sample("b2", pdist.Normal(0, sd).expand([h_dim]).to_event(1))
        # Layer 3
        w3=pyro.sample("w3", pdist.Normal(0, sd).expand([h_dim, y_dim]).to_event(2))
        b3=pyro.sample("b3", pdist.Normal(0, sd).expand([y_dim]).to_event(1))
        # NN
        h1=torch.tanh((x @ w1) + b1)
        h2=torch.tanh((h1 @ w2) + b2) 
        logits=(h2 @ w3 + b3)
        # Save deterministc variable (logits) in trace
        pyro.deterministic("logits", logits)
        # Categorical likelihood
        with pyro.plate("labels", n):
            obs=pyro.sample("obs", pdist.Categorical(logits=logits), obs=y)
            
#Make the Model and use NUTS for inference
# Instantiate the Model object
model=Model()
# Wrap the Model
wrapped_model = partial(model, x=x, y=y)

# use NUTS here instead of SVI
num_samples = 100
nuts_kernel = pyro.infer.NUTS(wrapped_model, jit_compile=False)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples, num_chains=2, warmup_steps=100)
mcmc.run()
samples = mcmc.get_samples()

# Clear any previously used parameters
pyro.clear_param_store()

print(f"Number of samples: {len(samples)}")
print(f"Samples keys: {samples.keys()}")

# Get posterior predictive and apply on test set 
posterior_predictive = pyro.infer.Predictive(model, samples, num_samples = num_samples, return_sites = ["logits"])(x_test)

# Print accuracy
logits=posterior_predictive['logits']
print("Shape of posterior preditive for y (logits):", logits.shape)
print("Success: %.2f" % accuracy(logits, y_test))


## arviz for quality check 

# use arviz to summarize and investigate: plot
data = arviz.from_pyro(mcmc)

# ESS, r-hat
summary = arviz.summary(data)
print(summary)
# Save summary to a CSV file
summary.to_csv("/opt/streamline/jmj/Courses/PLM/results_pyro_NUTS/week4_2_summary.csv", index=True)

# trace plot
arviz.plot_trace(data)
plt.savefig("/opt/streamline/jmj/Courses/PLM/results_pyro_NUTS/week4_2_traceplots.png")
