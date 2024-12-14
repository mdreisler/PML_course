import torch
import pyro
import pyro.distributions as pdist
import torch.distributions as tdist
import arviz
import numpy as np
from pyro.infer import MCMC, NUTS
from torch.distributions import constraints
import matplotlib.pyplot as plt
from scipy.stats import norm


class MyDensity(pdist.TorchDistribution):
    # The integration interval
    support = constraints.interval(-3,3) 
    
    # Constraint for the starting value used in the sampling
    # (should be within integration interval)
    arg_constraints = {"start": support}

    def __init__(self, start=torch.tensor(0.0)):
      # start = starting value for HMC sampling, default 0
      self.start = start
      super(pdist.TorchDistribution, self).__init__()

    def sample(self, sample_shape=torch.Size()):
        # This is only used to start the HMC sampling
        # It simply returns the starting value for the sampling
        return self.start

    def log_prob(self, x):
        # Return log of the (unnormalized) density
        dens= torch.exp(-x**2/2)*(torch.sin(x)**2+3*torch.cos(x)**2*torch.sin(7*x)**2+1)
        
        if torch.any(x < -3) or torch.any(x > 3):
            return torch.tensor(float('-inf')).to(x.device)
        
        return torch.log(dens) 


# Specify the model, which in our case is just our MyDensity distribution

def model():
    # Use your custom distribution, MyDensity
    pyro.sample('x', MyDensity())


def est_2moment(samples):
    return np.mean(np.array(samples)**2)

def evaluate_estimate(rounds, no_chains=1):
    est_mcmc=[]
    for n in [10,100,1000]:
        mc=[]
        
        for i in range(rounds):
            pyro.set_rng_seed(i + n+ 12345)

            nuts_kernel = NUTS(model) 
            mcmc = MCMC(nuts_kernel, num_samples=n, warmup_steps=100, num_chains = no_chains) 
            mcmc.run()
            posterior_samples = mcmc.get_samples()
            samples= posterior_samples['x'].numpy()
            mc.append(est_2moment(samples))

        est_mcmc.append([np.mean(mc),np.std(mc)])
    return est_mcmc

def visual_evaluation(est_mcmc):
    means=[est_mcmc[i][0] for i in range(3)]
    stds=[est_mcmc[i][1] for i in range(3)]
    print(means,stds)

    x_positions = [0.2,0.4,0.6]  # X-axis positions for each list
    fig3, ax3 = plt.subplots()
    ax3.errorbar(x_positions, means, yerr=stds, fmt='o', capsize=5, label='Mean ± StdDev')

    # Add labels, title, and legend
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(['','MCMC', ''])  # Label each list
    ax3.set_xlabel('Estimator')
    ax3.set_ylabel('Value')
    ax3.set_title('Mean and Standard deviation for n=10,100,1000')
    ax3.legend()
    ax3.grid(True)

    # Save the plot
    fig3.savefig('ex4_1_mcmc_bars.png')



def compare(est_mcmc, rounds):
    ### Compare with other sampling methods from ex3 ###

    def p(x):
        return np.exp(-x**2/2)*(np.sin(x)**2+3*np.cos(x)**2*np.sin(7*x)**2+1)

    def q1(x):
        return np.ones_like(x)

    x = np.linspace(-3,3,1000)
    k1= np.max(p(x))

    def rejection_uniform(k1, n=1000):
        #rejection sampling to get points from p(x)
        samples=[]
        for i in range(n):
            z = np.random.uniform(-3,3)
            u = np.random.uniform(0,1)
            if u*k1*q1(z) < p(z):
                samples.append(z)
        
        return samples

    def q2(x):
        return norm.pdf(x,0,1)

    #find the maximum of q2(x) 
    arg_max_p = np.argmax(p(x))
    max_norm = q2(x[arg_max_p])
    # we want max_norm*q2(x)=k1
    k2= k1 / max_norm

    def rejection_normal(k2, n=1000):
        #rejection sampling to get points from p(x)
        samples=[]
        for i in range(n):
            z2 = np.random.normal(0,1)
            if z2 < 3 and z2 > -3:
                u = np.random.uniform(0,1)
                if u*k2*q2(z2) < p(z2):
                    samples.append(z2)
        
        return samples

    #self-normalized importance sampling with gaussian
    #gives already estimate for E(x^2)
    def importance_normal(n=1000):
        z3 = np.random.normal(0,1,n)
        weights = (p(z3)/q2(z3))/(np.sum(p(z3)/q2(z3)))
        return np.sum(weights*(np.array(z3)**2))


    est_rejection_unif=[]
    est_rejection_norm=[]
    est_importance=[]

    for n in [10,100,1000]:
        ru=[]
        rn=[]
        im=[]

        for i in range(rounds):
            samples = rejection_uniform(k1,n)
            ru.append(est_2moment(samples))
            
            samples = rejection_normal(k2,n)
            rn.append(est_2moment(samples))
            
            im.append(importance_normal(n))
            
        est_rejection_unif.append([np.nanmean(ru),np.nanstd(ru)])
        est_rejection_norm.append([np.nanmean(rn), np.nanstd(rn)])   
        est_importance.append([np.nanmean(im), np.nanstd(im)])

    #########

    # Store lists in a single array for easier processing
    data = [est_mcmc, est_rejection_unif, est_rejection_norm, est_importance]

    # Calculate means and standard deviations
    means = [data[i][j][0] for i in range(len(data)) for j in range(len(data[i]))]
    std_devs = [data[i][j][1] for i in range(len(data)) for j in range(len(data[i]))]

    # Plotting
    x_positions = [0.2,0.4,0.6, 1.2,1.4,1.6, 2.2,2.4,2.6, 3.2,3.4,3.6]  # X-axis positions for each list
    fig2, ax2 = plt.subplots()
    ax2.errorbar(x_positions, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± StdDev')

    # Add labels, title, and legend
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(['','MCMC sampler', '', 
                        '','Rejection Uniform', '', 
                        '','Rejection Normal','',
                        '','Importance',''])  # Label each list
    ax2.set_xlabel('Estimator')
    ax2.set_ylabel('Value')
    ax2.set_title('Mean and Standard deviation for n=10,100,1000')
    ax2.legend()
    ax2.grid(True)

    # Save the plot
    fig2.savefig('ex4_1_all.png')

def arviz_diagnostic(sampler):
    num_chains = sampler.num_chains
    num_samples = sampler.num_samples

    data = arviz.from_pyro(sampler)
    # ESS, r-hat
    summary_df = arviz.summary(data)

    fig4, ax4 = plt.subplots(figsize=(10, 6))  # Set figure size
    ax4.axis("tight")
    ax4.axis("off")

    # Create the table
    table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns, rowLabels=summary_df.index, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Customize font size
    table.auto_set_column_width(col=list(range(len(summary_df.columns))))
    ax4.set_title(f"MCMC sampler: {num_chains} Chains, {num_samples} Samples", fontsize=14, pad=20)

    # Save and display
    fig4.savefig("summary_table.png", dpi=300, bbox_inches="tight")




if __name__ == "__main__":

    # Run HMC / NUTS
    no_chains =4
    nuts_kernel = NUTS(model) 
    mcmc= MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=no_chains) 
    mcmc.run()
    posterior_samples = mcmc.get_samples()
    arviz_diagnostic(mcmc)



    # Plot the samples and the true density
    fig1, ax1 = plt.subplots()
    ax1.hist(posterior_samples['x'].numpy(), bins=250, density=True, label='mcmc samples')
    ax1.set_xlabel('x')
    x1=np.linspace(-3,3,1000)
    ax1.plot(x1, np.exp(-x1**2/2)*(np.sin(x1)**2+3*np.cos(x1)**2*np.sin(7*x1)**2+1)/4, 'r', label='true density')
    ax1.legend()
    fig1.savefig('ex4_1_samples.png')

    # Estimate E(x^2) using the samples
    samples=posterior_samples['x'].numpy()
    print('The estimate for E(x^2) is:',est_2moment(samples))

    rounds=500
    # Evaluate the estimate 
    est_mcmc=evaluate_estimate(rounds,no_chains)
    visual_evaluation(est_mcmc)
    compare(est_mcmc,rounds)