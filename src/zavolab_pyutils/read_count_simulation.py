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
    Mimics the generative assumptions of DESeq2.
    
    Parameters
    ----------
    N_genes : int, optional
        Number of genes to simulate. Default is 3000.
    conditions : list of str, optional
        List of biological conditions. Default is ["Control", "Treatment"].
    n_replicates : int, optional
        Number of biological replicates per condition. Default is 3.
    alpha : float, optional
        True biological dispersion parameter (Variance = Mean + alpha * Mean^2). Default is 0.05.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
        
    Returns
    -------
    counts_df : pd.DataFrame
        Simulated raw count matrix (genes x samples).
    metadata_df : pd.DataFrame
        Metadata mapping samples to biological conditions.
    scaling_factors : np.ndarray
        The true, hidden library size scaling factors used during generation.
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
    Mimics the generative assumptions of the Sanity Bayesian model.
    
    Parameters
    ----------
    N_genes : int, optional
        Number of genes to simulate. Default is 3000.
    conditions : list of str, optional
        List of biological conditions. Default is ["Control", "Treatment"].
    n_replicates : int, optional
        Number of biological replicates per condition. Default is 3.
    v_log : float, optional
        True biological variance (v_g) of the log-fold changes. Default is 0.05.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
        
    Returns
    -------
    counts_df : pd.DataFrame
        Simulated raw count matrix (genes x samples).
    metadata_df : pd.DataFrame
        Metadata mapping samples to biological conditions.
    scaling_factors : np.ndarray
        The true, hidden library size scaling factors used during generation.
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

def simulate_isoform_poisson_lognormal_counts(
    N_genes=3000, conditions=["Control", "Treatment"], 
    n_replicates=3, v_log_gene=0.05, v_log_iso=0.02, seed=42, 
    SF_sigma=0.3, exp_average_M=3, exp_average_S=2, frac_diff_usage=0.1,
    lambda_iso=1.5, shift_logfc_mean=0.0, shift_logfc_sd=1.0
):
    """
    Simulates isoform counts using a Poisson-LogNormal mixture.
    Samples the number of isoforms per gene from a Poisson distribution.
    Applies differential usage via a Normal-sampled log-fold change.
    Returns the isoform_to_gene mapping required for downstream Log-Odds Sanity testing.
    """
    samples, metadata_df, scaling_factors = _generate_metadata_and_factors(conditions, n_replicates, SF_sigma, seed)
    
    np.random.seed(seed)
    exp_average_vals = np.random.lognormal(exp_average_M, exp_average_S, N_genes)
    genes = [f"Gene_{i}" for i in range(1, N_genes+1)]
    
    # Determine which genes will have differential relative usage
    n_diff = int(N_genes * frac_diff_usage)
    diff_genes = set(np.random.choice(genes, n_diff, replace=False))
    
    iso_counts_dict = {}
    isoform_to_gene = {}
    
    sigma_gene = np.sqrt(v_log_gene)
    sigma_iso = np.sqrt(v_log_iso)
    
    for k, gene in enumerate(genes):
        mu_gene = exp_average_vals[k]
        
        # 1. Sample number of isoforms (minimum 1)
        n_iso = max(1, np.random.poisson(lambda_iso))
        iso_names = [f"{gene}_Iso{i+1}" for i in range(n_iso)]
        
        # Map isoforms to their parent gene
        for iso in iso_names:
            isoform_to_gene[iso] = gene
            
        if n_iso == 1 and gene in diff_genes:
            diff_genes.remove(gene) # Cannot have diff usage with 1 isoform
            
        # 2. Baseline Dirichlet proportions
        pi_base = np.random.dirichlet(np.ones(n_iso)) if n_iso > 1 else np.array([1.0])
        
        # 3. Shift proportions for Condition 2
        if gene in diff_genes:
            # Draw log-fold changes for each isoform
            log_fc = np.random.normal(shift_logfc_mean, shift_logfc_sd, size=n_iso)
            
            # Apply fold change multiplicatively to the proportions and re-normalize
            pi_cond2_unnorm = pi_base * np.exp(log_fc)
            pi_cond2 = pi_cond2_unnorm / np.sum(pi_cond2_unnorm)
        else:
            pi_cond2 = pi_base
            
        sample_iso_counts = {iso: [] for iso in iso_names}
        
        for s, sample in enumerate(samples):
            cond = metadata_df.loc[s, 'condition']
            pi_current = pi_cond2 if cond == conditions[1] else pi_base
            
            # Gene-level biological Noise (affects all isoforms equally)
            delta_g = np.random.normal(0, sigma_gene)
            
            for i, iso in enumerate(iso_names):
                # Isoform-level biological Noise (splicing/cleavage variance)
                delta_iso = np.random.normal(0, sigma_iso)
                
                # True Rate
                base_rate = mu_gene * pi_current[i] * scaling_factors[s] * np.exp(delta_g + delta_iso)
                
                # Final count drawn from Poisson
                count = np.random.poisson(base_rate)
                sample_iso_counts[iso].append(count)
                
        for iso in iso_names:
            iso_counts_dict[iso] = sample_iso_counts[iso]
            
    iso_counts_df = pd.DataFrame(iso_counts_dict, index=samples).transpose()
    
    return iso_counts_df, metadata_df, isoform_to_gene, diff_genes


def simulate_isoform_negative_binomial_counts(
    N_genes=3000, conditions=["Control", "Treatment"], 
    n_replicates=3, alpha_gene=0.05, alpha_iso=0.02, seed=42, 
    SF_sigma=0.3, exp_average_M=3, exp_average_S=2, frac_diff_usage=0.1,
    lambda_iso=1.5, shift_logfc_mean=0.0, shift_logfc_sd=1.0
):
    """
    Simulates isoform counts using a Negative Binomial mixture.
    Returns the isoform_to_gene mapping required for downstream Log-Odds Sanity testing.
    """
    samples, metadata_df, scaling_factors = _generate_metadata_and_factors(conditions, n_replicates, SF_sigma, seed)
    
    np.random.seed(seed)
    exp_average_vals = np.random.lognormal(exp_average_M, exp_average_S, N_genes)
    genes = [f"Gene_{i}" for i in range(1, N_genes+1)]
    
    n_diff = int(N_genes * frac_diff_usage)
    diff_genes = set(np.random.choice(genes, n_diff, replace=False))
    
    iso_counts_dict = {}
    isoform_to_gene = {} # <-- NEW: Replaces isoform_pairs
    
    n_param_gene = 1 / alpha_gene if alpha_gene > 0 else 1e6
    n_param_iso = 1 / alpha_iso if alpha_iso > 0 else 1e6
    
    for k, gene in enumerate(genes):
        mu_gene = exp_average_vals[k]
        
        # 1. Sample number of isoforms
        n_iso = max(1, np.random.poisson(lambda_iso))
        iso_names = [f"{gene}_Iso{i+1}" for i in range(n_iso)]
        
        # Map isoforms to their parent gene
        for iso in iso_names:
            isoform_to_gene[iso] = gene
            
        if n_iso == 1 and gene in diff_genes:
            diff_genes.remove(gene) # Cannot have diff usage with 1 isoform
            
        # 2. Baseline Dirichlet proportions
        pi_base = np.random.dirichlet(np.ones(n_iso)) if n_iso > 1 else np.array([1.0])
        
        # 3. Shift proportions for Condition 2
        if gene in diff_genes:
            log_fc = np.random.normal(shift_logfc_mean, shift_logfc_sd, size=n_iso)
            pi_cond2_unnorm = pi_base * np.exp(log_fc)
            pi_cond2 = pi_cond2_unnorm / np.sum(pi_cond2_unnorm)
        else:
            pi_cond2 = pi_base
            
        sample_iso_counts = {iso: [] for iso in iso_names}
        
        for s, sample in enumerate(samples):
            cond = metadata_df.loc[s, 'condition']
            pi_current = pi_cond2 if cond == conditions[1] else pi_base
            
            # Realized gene base rate
            mu_gene_sample = mu_gene * scaling_factors[s]
            if n_param_gene < 1e5:
                realized_gene_rate = np.random.gamma(shape=n_param_gene, scale=mu_gene_sample/n_param_gene)
            else:
                realized_gene_rate = mu_gene_sample
            
            for i, iso in enumerate(iso_names):
                mu_iso = realized_gene_rate * pi_current[i]
                
                if mu_iso > 0:
                    p_param_iso = n_param_iso / (n_param_iso + mu_iso)
                    count = np.random.negative_binomial(n_param_iso, p_param_iso)
                else:
                    count = 0
                    
                sample_iso_counts[iso].append(count)
                
        for iso in iso_names:
            iso_counts_dict[iso] = sample_iso_counts[iso]
        
    iso_counts_df = pd.DataFrame(iso_counts_dict, index=samples).transpose()
    
    return iso_counts_df, metadata_df, isoform_to_gene, diff_genes