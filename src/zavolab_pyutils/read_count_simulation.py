import numpy as np
import pandas as pd

def _generate_metadata_and_factors(conditions, n_replicates, SF_sigma=0.3, seed=42):
    """Helper function to dynamically generate metadata and log-normal size factors."""
    np.random.seed(seed)
    samples = []
    condition_labels = []
    sample_idx = 1
    
    for cond in conditions:
        for _ in range(n_replicates):
            samples.append(f"Sample_{sample_idx}")
            condition_labels.append(cond)
            sample_idx += 1
            
    metadata_df = pd.DataFrame({"sample": samples, "condition": condition_labels})
    
    # Sample true scaling factors from a Log-Normal distribution
    log_scaling_factors = np.random.normal(loc=0.0, scale=SF_sigma, size=len(samples))
    scaling_factors = np.exp(log_scaling_factors)
    
    return samples, metadata_df, scaling_factors


def simulate_negative_binomial_counts(
    N_genes=3000, conditions=["Control", "Treatment"], 
    n_replicates=3, alpha=0.05, seed=42, SF_sigma=0.3,exp_average_M=3,exp_average_S=3,
):
    """
    Simulates RNA-seq counts using a Poisson-Gamma mixture (Negative Binomial).
    This mimics the generative assumptions of DESeq2.
    """
    samples, metadata_df, scaling_factors = _generate_metadata_and_factors(conditions, n_replicates, SF_sigma=SF_sigma, seed=seed)
    
    np.random.seed(seed)
    exp_average_vals = np.random.lognormal(exp_average_M, exp_average_S, N_genes)
    genes = [f"Gene_{i}" for i in range(1, N_genes+1)]
    
    n_param = 1 / alpha
    counts_dict = {}
    
    for k, gene in enumerate(genes):
        mu_gene = exp_average_vals[k]
        gene_counts = []
        
        for s, sample in enumerate(samples):
            # Expected mean accounting for library size
            mu_sample = mu_gene * scaling_factors[s]
            
            # Dynamically calculate 'p' for the Negative Binomial distribution
            p_param = n_param / (n_param + mu_sample)
            
            # Draw count directly from the Negative Binomial (Poisson-Gamma mixture)
            count = np.random.negative_binomial(n_param, p_param)
            gene_counts.append(count)
            
        counts_dict[gene] = gene_counts
        
    counts_df = pd.DataFrame(counts_dict, index=samples, columns=genes).transpose()
    return counts_df, metadata_df, scaling_factors


def simulate_poisson_lognormal_counts(
    N_genes=3000, conditions=["Control", "Treatment"], 
    n_replicates=3, v_log=0.05, seed=42, SF_sigma=0.3,exp_average_M=3,exp_average_S=3,
):
    """
    Simulates RNA-seq counts using a Poisson-LogNormal mixture.
    This mimics the generative assumptions of the Sanity Bayesian model.
    """
    samples, metadata_df, scaling_factors = _generate_metadata_and_factors(conditions, n_replicates, SF_sigma=SF_sigma, seed=seed)
    
    np.random.seed(seed)
    exp_average_vals = np.random.lognormal(exp_average_M, exp_average_S, N_genes)
    genes = [f"Gene_{i}" for i in range(1, N_genes+1)]
    
    sigma = np.sqrt(v_log) # Standard deviation of the biological log-fold changes
    counts_dict = {}
    
    for k, gene in enumerate(genes):
        mu_gene = exp_average_vals[k]
        gene_counts = []
        
        for s, sample in enumerate(samples):
            # 1. Base expected rate accounting for library size
            base_rate = mu_gene * scaling_factors[s]
            
            # 2. Biological Noise: Draw a random log-fold change from a Normal distribution
            delta = np.random.normal(0, sigma)
            
            # 3. Apply the biological noise to the base rate to get the true Poisson lambda
            lambda_sample = base_rate * np.exp(delta)
            
            # 4. Technical Noise: Draw the final count from the Poisson distribution
            count = np.random.poisson(lambda_sample)
            gene_counts.append(count)
            
        counts_dict[gene] = gene_counts
        
    counts_df = pd.DataFrame(counts_dict, index=samples, columns=genes).transpose()
    return counts_df, metadata_df, scaling_factors