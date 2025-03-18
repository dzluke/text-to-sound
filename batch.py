from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import ParameterGenerator
from main import run
from eval import EvaluationFramework

def run_comprehensive_evaluation(check_cache=True):
    # Initialize the evaluation framework
    framework = EvaluationFramework()
    
    # Define corpora to test
    sound_corpora = [
        # "./corpora/sound/toy",
        "./corpora/sound/anonymous_corpus",
        "./corpora/sound/mothman",
    ]
    
    text_corpora = [
        "./corpora/text/test.txt",
        # "./corpora/text/repeat.txt",
    ]

    sound_encoders = ["MuQ"]
    text_encoders = ["fastText", "word2vec"]
    mappings = ["identity", "cluster"]
    ks = [2, 3]  # Different values of k for clustering
    sound_preprocessings = [1000]
    normalizations = ["standard"]
    dims = [2]
    distance_metrics = ["euclidean"]
    mapping_evaluations = ["pairwise"]    

    # For each corpus combination, run evaluations
    for sound_corpus in sound_corpora:
        for text_corpus in text_corpora:
            print(f"Evaluating with sound corpus: {Path(sound_corpus).stem} and text corpus: {Path(text_corpus).stem}")
            
            # Create parameter combinations
            e = ParameterGenerator(
                sound_path=sound_corpus,
                text_path=text_corpus,
                sound_encoders=sound_encoders,
                text_encoders=text_encoders,
                mappings=mappings,
                sound_preprocessings=sound_preprocessings,
                normalizations=normalizations,
                dims=dims,
                distance_metrics=distance_metrics,
                mapping_evaluations=mapping_evaluations,
                ks=ks  # Include the k values
            )
            
            # Run each parameter combination
            parameter_list = e.create_params()

            # Get existing results to avoid re-running experiments
            existing_results = None
            if framework.results_file.exists():
                existing_results = framework.get_results_dataframe()
            
            for parameters in parameter_list:
                # Check if this experiment has been run before
                if check_cache and existing_results is not None and not existing_results.empty:
                    # Start with basic matching conditions
                    conditions = (
                        (existing_results['sound_corpus'] == Path(parameters.sound_path).stem) &
                        (existing_results['text_corpus'] == Path(parameters.text_path).stem) &
                        (existing_results['sound_encoder'] == parameters.sound_encoder) &
                        (existing_results['text_encoder'] == parameters.text_encoder) &
                        (existing_results['mapping_method'] == parameters.mapping) &
                        (existing_results['sound_preprocessing'] == parameters.sound_preprocessing) &
                        (existing_results['normalization'] == parameters.normalization) &
                        (existing_results['dim'] == parameters.dim) &
                        (existing_results['distance_metric'] == parameters.distance_metric)
                    )
                    
                    # For cluster mapping, also check k value
                    if parameters.mapping == 'cluster' and hasattr(parameters, 'k'):
                        # First ensure there's a k column in the dataframe
                        if 'k' in existing_results.columns:
                            conditions = conditions & (existing_results['k'] == parameters.k)
                        else:
                            # If k column doesn't exist, this experiment hasn't been run
                            pass
                    
                    # Check if ANY row matches all conditions (exact match)
                    if conditions.any():
                        # matching_experiments = existing_results[conditions]
                        # print(f"Skipping previously run experiment with parameters: {parameters.to_string()}")
                        print(f"Skipping previously run experiment")
                        continue
                
                # print(f"Running with parameters: {parameters.to_string()}")
                run(parameters, cache=True, evaluator=framework)
    
    # Generate a report of all evaluations
    report = framework.generate_report()
    print("Evaluation complete! Report generated.")
    
    # Analyze and visualize results
    analyze_results(framework)
    
    return report

def analyze_results(framework):
    """
    Simplified analysis that only shows the average pairwise distance for each mapping type.
    """
    print("\n" + "="*50)
    print("ANALYSIS OF EVALUATION RESULTS")
    print("="*50 + "\n")
    
    df = framework.get_results_dataframe()
    
    if len(df) == 0:
        print("No results to analyze.")
        return
    
    # Focus only on mapping_method and pairwise distance metrics
    # Look for both pairwise_score and pairwise_distance (for compatibility)
    if "pairwise_score" in df.columns:
        metric = "pairwise_score"
    elif "pairwise_distance" in df.columns:
        metric = "pairwise_distance"
    else:
        print("No pairwise distance/score metric found in results.")
        return
    
    print(f"AVERAGE {metric.upper()} BY MAPPING METHOD")
    print("-"*50)
    
    # Get all mapping methods
    mapping_methods = df["mapping_method"].unique()
    
    # For each mapping method, calculate average score
    for mapping in mapping_methods:
        # Get rows for this mapping method
        mapping_rows = df[df["mapping_method"] == mapping]
        
        # Get valid values (non-NaN)
        values = [v for v in mapping_rows[metric].values if not np.isnan(v)]
        
        if values:
            # Calculate average
            avg_score = sum(values) / len(values)
            print(f"Mapping: {mapping}")
            print(f"  Average {metric}: {avg_score:.4f}")
            print(f"  Number of experiments: {len(values)}")
            print()
        else:
            print(f"Mapping: {mapping}")
            print(f"  No valid results found")
            print()

def generate_parameter_comparison_plots(df, parameters, metric, plots_dir):
    """
    Generate simplified plots that clearly show which parameter values lead to 
    the best results across different corpora.
    """
    # Calculate how many parameters have multiple values
    valid_params = [p for p in parameters if len(df[p].unique()) > 1]
    num_params = len(valid_params)
    
    # Include corpus as a parameter if there are multiple corpora
    num_corpora = len(df["sound_corpus"].unique())
    if num_corpora > 1:
        num_params += 1
    
    # Calculate grid size needed (min 1x1, otherwise try to make it square-ish)
    if num_params <= 1:
        rows, cols = 1, 1
    else:
        cols = min(2, num_params)  # Max 2 columns
        rows = (num_params + cols - 1) // cols  # Ceiling division
    
    # Set up subplot grid with the exact size needed
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))
    
    # Handle single subplot case
    if num_params <= 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    param_idx = 0
    for param in parameters:
        param_values = df[param].unique()
        
        if len(param_values) <= 1:
            continue  # Skip parameters with only one value
            
        # Get mean performance for each parameter value across all corpora
        param_perf = df.groupby(param)[metric].mean().reset_index()
        param_perf = param_perf.sort_values(by=metric)
        
        # Create bar plot on the current subplot
        ax = axes[param_idx]
        # Update barplot to avoid warning - use the parameter as both x and hue
        sns.barplot(x=param, y=metric, data=param_perf, ax=ax, 
                   order=param_perf[param], hue=param, legend=False)
        
        # Add value labels on top of bars
        for p in ax.patches:
            value_text = "nan" if np.isnan(p.get_height()) else f"{p.get_height():.4f}"
            ax.annotate(value_text, 
                     (p.get_x() + p.get_width() / 2., p.get_height() or 0),
                     ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')
        
        ax.set_title(f"Performance by {param}")
        ax.set_ylabel(f"{metric} (lower is better)")
        
        # Fix the warning about set_ticklabels
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels(param_perf[param], rotation=45, ha='right')
        
        param_idx += 1
    
    # Add corpus comparison as the last plot if we have multiple corpora
    if num_corpora > 1 and param_idx < len(axes):
        ax = axes[param_idx]
        corpus_perf = df.groupby("sound_corpus")[metric].mean().reset_index()
        corpus_perf = corpus_perf.sort_values(by=metric)
        
        # Update barplot to use hue
        sns.barplot(x="sound_corpus", y=metric, data=corpus_perf, ax=ax, 
                   order=corpus_perf["sound_corpus"], hue="sound_corpus", legend=False)
        
        for p in ax.patches:
            value_text = "nan" if np.isnan(p.get_height()) else f"{p.get_height():.4f}"
            ax.annotate(value_text, 
                     (p.get_x() + p.get_width() / 2., p.get_height() or 0),
                     ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')
        
        ax.set_title("Performance by Sound Corpus")
        ax.set_ylabel(f"{metric} (lower is better)")
        
        # Fix the warning about set_ticklabels
        ax.set_xticks(range(len(corpus_perf["sound_corpus"])))
        ax.set_xticklabels(corpus_perf["sound_corpus"], rotation=45, ha='right')
        
        param_idx += 1
    
    # Hide any unused subplots
    for i in range(param_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"parameter_performance_summary.png")
    plt.close()
    
    # If we have multiple corpora, create a corpus-parameter interaction plot for the 3 most impactful parameters
    if len(df["sound_corpus"].unique()) > 1:
        # Determine the most impactful parameters by their variance in performance across corpora
        impact_scores = {}
        for param in parameters:
            if len(df[param].unique()) <= 1:
                continue
                
            # Calculate how much each parameter value affects the score across different corpora
            param_corpus_scores = []
            for corpus in df["sound_corpus"].unique():
                corpus_df = df[df["sound_corpus"] == corpus]
                if len(corpus_df) < 2:  # Skip if not enough data
                    continue
                param_scores = corpus_df.groupby(param)[metric].mean()
                param_corpus_scores.append(param_scores)
            
            # If we have scores from multiple corpora, compute the variance
            if len(param_corpus_scores) > 1:
                # Align indices and compute variance across corpora
                param_values = set()
                for scores in param_corpus_scores:
                    param_values.update(scores.index)
                
                # Compute impact as max variation across corpora for any parameter value
                max_variation = 0
                for val in param_values:
                    val_scores = [scores.get(val, np.nan) for scores in param_corpus_scores]
                    val_scores = [s for s in val_scores if not np.isnan(s)]
                    if len(val_scores) > 1:
                        variation = max(val_scores) - min(val_scores)
                        max_variation = max(max_variation, variation)
                
                impact_scores[param] = max_variation
        
        # Get top 3 most impactful parameters
        top_params = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_param_names = [p[0] for p in top_params]
        
        # Create a plot showing parameter performance by corpus for top parameters
        if top_param_names:
            fig, axes = plt.subplots(len(top_param_names), 1, figsize=(10, 4 * len(top_param_names)))
            if len(top_param_names) == 1:
                axes = [axes]
            
            for i, param in enumerate(top_param_names):
                ax = axes[i]
                
                # Get data in proper format for grouped bar chart
                pivot_data = df.pivot_table(
                    index=param, columns="sound_corpus", values=metric, aggfunc="mean"
                ).reset_index()
                
                # Melt the data for seaborn
                plot_data = pivot_data.melt(id_vars=[param], var_name="sound_corpus", value_name=metric)
                
                # Create grouped bar chart
                sns.barplot(x=param, y=metric, hue="sound_corpus", data=plot_data, ax=ax, palette="Set2")
                
                ax.set_title(f"Performance by {param} across Sound Corpora")
                ax.set_ylabel(f"{metric} (lower is better)")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend(title="Sound Corpus")
            
            plt.tight_layout()
            plt.savefig(plots_dir / "corpus_parameter_interaction.png")
            plt.close()
    
    # Create a simple finding summary text file
    with open(plots_dir / "parameter_findings.txt", "w") as f:
        f.write("PARAMETER PERFORMANCE SUMMARY\n")
        f.write("============================\n\n")
        
        # Overall best parameter values
        f.write("BEST PARAMETER VALUES ACROSS ALL CORPORA:\n")
        for param in parameters:
            param_values = df[param].unique()
            if len(param_values) <= 1:
                continue
                
            param_perf = df.groupby(param)[metric].mean().reset_index()
            param_perf = param_perf.sort_values(by=metric)
            best_value = param_perf.iloc[0][param]
            best_score = param_perf.iloc[0][metric]
            
            f.write(f"- {param}: Best value is '{best_value}' (avg score: {best_score:.4f})\n")
        
        f.write("\n")
        
        # Corpus-specific findings
        if len(df["sound_corpus"].unique()) > 1:
            f.write("CORPUS-SPECIFIC BEST PARAMETERS:\n")
            for corpus in df["sound_corpus"].unique():
                corpus_df = df[df["sound_corpus"] == corpus]
                f.write(f"\nFor sound corpus '{corpus}':\n")
                
                for param in parameters:
                    param_values = corpus_df[param].unique()
                    if len(param_values) <= 1:
                        continue
                        
                    param_perf = corpus_df.groupby(param)[metric].mean().reset_index()
                    param_perf = param_perf.sort_values(by=metric)
                    best_value = param_perf.iloc[0][param]
                    best_score = param_perf.iloc[0][metric]
                    
                    f.write(f"- {param}: Best value is '{best_value}' (avg score: {best_score:.4f})\n")

# Remove/replace previous plotting functions that are no longer needed
def plot_metrics(framework, plots_dir):
    """Simplified function to just call our new analysis approach"""
    df = framework.get_results_dataframe()
    key_parameters = ["sound_encoder", "text_encoder", "mapping_method", 
                     "sound_preprocessing", "dim", "distance_metric"]
    generate_parameter_comparison_plots(df, key_parameters, "pairwise_distance", plots_dir)

if __name__ == "__main__":
    run_comprehensive_evaluation()
