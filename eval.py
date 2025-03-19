import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EvaluationFramework:
    """
    Framework for evaluating text-to-sound mapping with different parameters
    across multiple input corpora.
    """
    def __init__(self, results_dir="./evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "evaluation_results.csv"
        self.initialize_results_file()
    
    def initialize_results_file(self):
        """Initialize the results file if it doesn't exist."""
        if not self.results_file.exists():
            columns = [
                "sound_corpus", "text_corpus", "sound_encoder", "text_encoder",
                "mapping_method", "sound_preprocessing", "normalization",
                "dim", "distance_metric", "pairwise_distance", "wasserstein_distance",
                # Sound cluster metrics
                "sound_silhouette_score", "sound_calinski_harabasz_score", "sound_davies_bouldin_score",
                # Text cluster metrics
                "text_silhouette_score", "text_calinski_harabasz_score", "text_davies_bouldin_score",
                # Combined cluster metrics
                "combined_silhouette_score", "combined_calinski_harabasz_score", "combined_davies_bouldin_score",
                # Domain separation metrics
                "domain_silhouette_score", "domain_calinski_harabasz_score", "domain_davies_bouldin_score"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.results_file, index=False)
    
    def save_result(self, params, scores):
        """
        Save evaluation results for a parameter set.
        
        Args:
            params: Parameter object with configuration
            scores: Dict containing evaluation scores
        """
        # Read existing results
        df = pd.read_csv(self.results_file)
        
        # Create a new row with the parameters and scores
        sound_corpus = Path(params.sound_path).stem
        text_corpus = Path(params.text_path).stem
        
        # Clean up metric names - extract only the metric name without the description
        cleaned_scores = {}
        for key, value in scores.items():
            # Extract the base metric name if it has a description in parentheses
            if ("(" in key):
                base_key = key.split("(")[0].strip()
                cleaned_scores[base_key] = value
            else:
                cleaned_scores[key] = value
        
        # Create base row with parameters
        row = {
            "sound_corpus": sound_corpus,
            "text_corpus": text_corpus,
            "sound_encoder": params.sound_encoder,
            "text_encoder": params.text_encoder, 
            "mapping_method": params.mapping,
            "sound_preprocessing": params.sound_preprocessing,
            "normalization": params.normalization,
            "dim": params.dim,
            "distance_metric": params.distance_metric,
            "k": params.k if params.mapping == "cluster" else 0  # Set k to 0 for identity mapping
        }
        
        # Add all scores to the row
        for key, value in cleaned_scores.items():
            row[key] = value
        
        # Add the new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Save the updated results
        df.to_csv(self.results_file, index=False)
        
        # Save detailed results as JSON for this specific run
        # details_file = self.results_dir / f"{params.filename()}_details.json"
        # with open(details_file, "w") as f:
        #     json.dump({
        #         "parameters": params.to_string(),
        #         "scores": scores
        #     }, f, indent=2)
    
    def find_best_parameters(self, metric="pairwise_distance", lower_is_better=True, 
                            sound_corpus=None, text_corpus=None):
        """
        Find the best parameter set based on a specific metric.
        
        Args:
            metric: The metric to optimize
            lower_is_better: Whether a lower score is better
            sound_corpus: Filter by sound corpus (optional)
            text_corpus: Filter by text corpus (optional)
            
        Returns:
            pd.DataFrame: The best parameter combinations
        """
        df = pd.read_csv(self.results_file)
        
        # Apply filters if provided
        if sound_corpus:
            df = df[df["sound_corpus"] == sound_corpus]
        if text_corpus:
            df = df[df["text_corpus"] == text_corpus]
        
        # Sort based on the metric
        ascending = lower_is_better
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        
        # Return the top results
        return df_sorted.head(10)
    
    def compare_encoders(self, metric="pairwise_distance", lower_is_better=True):
        """
        Compare performance of different encoder combinations across all corpora.
        
        Args:
            metric: The metric to compare
            lower_is_better: Whether a lower score is better
            
        Returns:
            pd.DataFrame: Average performance by encoder combination
        """
        df = pd.read_csv(self.results_file)
        
        # Group by encoder combinations and compute average score
        grouped = df.groupby(["sound_encoder", "text_encoder"])[metric].agg(["mean", "std", "count"])
        
        # Sort based on mean score
        ascending = lower_is_better
        return grouped.sort_values(by="mean", ascending=ascending)
    
    def plot_parameter_impact(self, parameter, metric="pairwise_distance"):
        """
        Create boxplot showing impact of a parameter on the evaluation metric.
        
        Args:
            parameter: Parameter to analyze (e.g., "sound_encoder", "dim")
            metric: Metric to evaluate
        """
        df = pd.read_csv(self.results_file)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=parameter, y=metric, data=df)
        plt.title(f"Impact of {parameter} on {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.results_dir / f"impact_{parameter}_on_{metric}.png"
        plt.savefig(plot_file)
        plt.close()
    
    def generate_report(self):
        """
        Generate a comprehensive report of all evaluation results.
        
        Returns:
            dict: A dictionary containing the report data
        """
        df = pd.read_csv(self.results_file)
        
        # Check which metrics are available in the dataframe
        available_metrics = set(df.columns)
        
        report = {
            "total_runs": len(df),
            "unique_sound_corpora": df["sound_corpus"].nunique(),
            "unique_text_corpora": df["text_corpus"].nunique(),
        }
        
        # Add best metric scores if available
        for metric_name in ["pairwise_distance", "wasserstein_distance"]:
            if metric_name in available_metrics:
                report[f"best_{metric_name}"] = self.find_best_parameters(metric_name).to_dict(orient="records")[0]
        
        # Add best silhouette scores if available
        if "combined_silhouette_score" in available_metrics:
            report["best_combined_silhouette"] = self.find_best_parameters(
                "combined_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        if "sound_silhouette_score" in available_metrics:
            report["best_sound_silhouette"] = self.find_best_parameters(
                "sound_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        if "text_silhouette_score" in available_metrics:
            report["best_text_silhouette"] = self.find_best_parameters(
                "text_silhouette_score", 
                lower_is_better=False
            ).to_dict(orient="records")[0]
        
        # Summary by sound_corpus
        report["by_sound_corpus"] = {}
        for corpus in df["sound_corpus"].unique():
            corpus_df = df[df["sound_corpus"] == corpus]
            
            corpus_report = {
                "runs": len(corpus_df)
            }
            
            # Add average metrics if available
            for metric_name in ["pairwise_distance", "wasserstein_distance"]:
                if metric_name in available_metrics:
                    corpus_report[f"avg_{metric_name}"] = corpus_df[metric_name].mean()
                    corpus_report[f"best_{metric_name}"] = self.find_best_parameters(
                        metric_name, 
                        sound_corpus=corpus
                    ).to_dict(orient="records")[0]
            
            report["by_sound_corpus"][corpus] = corpus_report
        
        # Summary by text_corpus (similar structure)
        report["by_text_corpus"] = {}
        for corpus in df["text_corpus"].unique():
            corpus_df = df[df["text_corpus"] == corpus]
            
            corpus_report = {
                "runs": len(corpus_df)
            }
            
            # Add average pairwise distance if available
            if "pairwise_distance" in available_metrics:
                corpus_report["avg_pairwise_distance"] = corpus_df["pairwise_distance"].mean()
                corpus_report["best_pairwise"] = self.find_best_parameters(
                    "pairwise_distance", 
                    text_corpus=corpus
                ).to_dict(orient="records")[0]
            
            report["by_text_corpus"][corpus] = corpus_report
        
        # Save report to JSON
        report_path = self.results_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
        
        return report

    def get_results_dataframe(self):
        """
        Returns the complete results dataframe.
        """
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame()

    def get_unique_values(self, column):
        """
        Returns the unique values in a specific column.
        
        Args:
            column: The column name to get unique values from
            
        Returns:
            list: The unique values in the column
        """
        df = self.get_results_dataframe()
        if column in df.columns:
            return df[column].unique().tolist()
        return []

def analyze_experiments(results_file="./evaluation_results/evaluation_results.csv", filter_params=None):
    """
    Analyze experiments from the evaluation results file, with optional filtering by parameters.

    Args:
        results_file (str): Path to the evaluation results CSV file.
        filter_params (dict, optional): Dictionary of parameters to filter by.
            Example: {'sound_corpus': 'mothman', 'dim': [2, 5], 'mapping_method': 'cluster'}
            If None, all experiments are analyzed.
    """
    # Load the results file
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)

    if df.empty:
        print("No results to analyze.")
        return
    
    # Apply filters if provided
    if filter_params:
        print("\n" + "="*50)
        print(f"FILTERED ANALYSIS - Using the following filters:")
        for param, values in filter_params.items():
            if isinstance(values, list):
                print(f"  * {param}: {values}")
            else:
                print(f"  * {param}: {values}")
        print("="*50)
        
        filtered_df = df.copy()
        filter_description = []
        
        for param, values in filter_params.items():
            if param in filtered_df.columns:
                # Handle both single values and lists of values
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[param].isin(values)]
                    filter_description.append(f"{param} in {values}")
                else:
                    filtered_df = filtered_df[filtered_df[param] == values]
                    filter_description.append(f"{param} = {values}")
        
        # Check if we have any results after filtering
        if filtered_df.empty:
            print(f"No results match the filter criteria: {', '.join(filter_description)}")
            return
            
        print(f"\nAnalyzing {len(filtered_df)} experiments matching the filter criteria")
        print(f"(Total experiments in dataset: {len(df)})")
        df = filtered_df
    else:
        print(f"\nAnalyzing all {len(df)} experiments (no filters applied)")

    print("\n" + "=" * 50)
    print("ANALYSIS OF EVALUATION RESULTS")
    print("=" * 50 + "\n")

    # Analyze both pairwise and Wasserstein distance metrics
    metrics = []
    if "pairwise_distance" in df.columns:
        metrics.append("pairwise_distance")
    if "wasserstein_distance" in df.columns:
        metrics.append("wasserstein_distance")
    
    if not metrics:
        print("No distance metrics found in results.")
        return

    for metric in metrics:
        print(f"\nAVERAGE {metric.upper()} BY MAPPING METHOD")
        print("-" * 50)

        # Analyze by mapping method
        mapping_methods = df["mapping_method"].unique()
        for mapping in mapping_methods:
            mapping_rows = df[df["mapping_method"] == mapping]
            values = mapping_rows[metric].dropna()

            if not values.empty:
                avg_score = values.mean()
                print(f"Mapping: {mapping}")
                print(f"  Average {metric}: {avg_score:.4f}")
                print(f"  Number of experiments: {len(values)}")

                # If cluster mapping, analyze metrics by k
                if mapping == "cluster" and "k" in mapping_rows.columns:
                    print(f"\n  {metric} by k:")
                    for k in sorted(mapping_rows["k"].dropna().unique()):
                        k_rows = mapping_rows[mapping_rows["k"] == k]
                        values = k_rows[metric].dropna()
                        if not values.empty:
                            avg = values.mean()
                            print(f"    k={k}: {avg:.4f}")
                    print()
            else:
                print(f"Mapping: {mapping}")
                print(f"  No valid {metric} results found")
                print()
    
    # Analyze clustering metrics if present
    clustering_metrics = [
        "combined_silhouette_score", "domain_silhouette_score",
        "sound_silhouette_score", "text_silhouette_score"
    ]
    
    # Check if we have clustering metrics in the results
    has_clustering_metrics = any(metric in df.columns for metric in clustering_metrics)
    
    if has_clustering_metrics and "cluster" in df["mapping_method"].unique():
        print("\nCLUSTERING METRICS ANALYSIS")
        print("-" * 50)
        
        cluster_df = df[df["mapping_method"] == "cluster"]
        
        # Analyze combined silhouette scores by k
        if "combined_silhouette_score" in cluster_df.columns:
            print("\nCOMBINED SILHOUETTE SCORE BY K")
            print("(Higher values indicate better-defined clusters across both domains)")
            print("-" * 50)
            
            for k in sorted(cluster_df["k"].dropna().unique()):
                k_rows = cluster_df[cluster_df["k"] == k]
                values = k_rows["combined_silhouette_score"].dropna()
                if not values.empty:
                    avg = values.mean()
                    print(f"k={k}: {avg:.4f}")
                    
                    # Show by corpus if we have multiple
                    if k_rows["sound_corpus"].nunique() > 1:
                        print("  By sound corpus:")
                        for corpus in k_rows["sound_corpus"].unique():
                            corpus_rows = k_rows[k_rows["sound_corpus"] == corpus]
                            corpus_avg = corpus_rows["combined_silhouette_score"].mean()
                            print(f"    {corpus}: {corpus_avg:.4f}")
            
        # Analyze domain silhouette scores
        if "domain_silhouette_score" in cluster_df.columns:
            print("\nDOMAIN SILHOUETTE SCORE BY K")
            print("(Lower values are better - indicate better blending between domains)")
            print("-" * 50)
            
            for k in sorted(cluster_df["k"].dropna().unique()):
                k_rows = cluster_df[cluster_df["k"] == k]
                values = k_rows["domain_silhouette_score"].dropna()
                if not values.empty:
                    avg = values.mean()
                    print(f"k={k}: {avg:.4f}")
                    
                    # Interpretation guide
                    if avg < -0.1:
                        print("  Interpretation: Excellent domain blending (strong negative score)")
                    elif avg < 0:
                        print("  Interpretation: Good domain blending (negative score)")
                    elif avg < 0.1:
                        print("  Interpretation: Moderate domain blending (near zero)")
                    else:
                        print("  Interpretation: Poor domain separation (positive score)")
                    
                    # Show by corpus if we have multiple
                    if k_rows["sound_corpus"].nunique() > 1:
                        print("  By sound corpus:")
                        for corpus in k_rows["sound_corpus"].unique():
                            corpus_rows = k_rows[k_rows["sound_corpus"] == corpus]
                            corpus_avg = corpus_rows["domain_silhouette_score"].mean()
                            print(f"    {corpus}: {corpus_avg:.4f}")
        
        # Compare sound and text clustering quality
        if "sound_silhouette_score" in cluster_df.columns and "text_silhouette_score" in cluster_df.columns:
            print("\nSOUND VS TEXT CLUSTERING QUALITY")
            print("(Higher values indicate better-defined clusters in each domain)")
            print("-" * 50)
            
            sound_avg = cluster_df["sound_silhouette_score"].dropna().mean()
            text_avg = cluster_df["text_silhouette_score"].dropna().mean()
            
            print(f"Average sound silhouette score: {sound_avg:.4f}")
            print(f"Average text silhouette score: {text_avg:.4f}")
            
            if sound_avg > text_avg:
                diff = sound_avg - text_avg
                print(f"Sound clusters are better defined by {diff:.4f}")
            elif text_avg > sound_avg:
                diff = text_avg - sound_avg
                print(f"Text clusters are better defined by {diff:.4f}")
            else:
                print("Sound and text clusters are equally well-defined")
            
            print("\nBy k value:")
            for k in sorted(cluster_df["k"].dropna().unique()):
                k_rows = cluster_df[cluster_df["k"] == k]
                sound_k_avg = k_rows["sound_silhouette_score"].dropna().mean()
                text_k_avg = k_rows["text_silhouette_score"].dropna().mean()
                print(f"  k={k}:")
                print(f"    Sound: {sound_k_avg:.4f}, Text: {text_k_avg:.4f}")
    
    # Additional summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 50)
    
    # Print unique values for key parameters
    print("Parameter distribution in analyzed experiments:")
    for param in ['sound_corpus', 'text_corpus', 'sound_encoder', 'text_encoder', 'dim']:
        if param in df.columns:
            unique_values = df[param].value_counts().to_dict()
            print(f"  {param}: {unique_values}")
    
    # Print best overall results for each metric
    for metric in metrics:
        print(f"\nBest {metric} results:")
        best_idx = df[metric].idxmin()  # Lower is better for both pairwise and Wasserstein
        if pd.notna(best_idx):
            best_row = df.loc[best_idx]
            print(f"  Best {metric}: {best_row[metric]:.4f}")
            for param in ['sound_corpus', 'text_corpus', 'sound_encoder', 'text_encoder', 
                        'mapping_method', 'dim', 'k']:
                if param in best_row and pd.notna(best_row[param]):
                    print(f"    {param}: {best_row[param]}")

    print("\nAnalysis complete.")
    
    return df  # Return the filtered dataframe for further analysis if needed

def generate_plots(results_file="./evaluation_results/evaluation_results.csv", output_dir="./evaluation_results/plots"):
    """
    Generate plots from the evaluation results.

    Args:
        results_file (str): Path to the evaluation results CSV file.
        output_dir (str): Directory to save the generated plots.
    """
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    df = pd.read_csv(results_file)

    if df.empty:
        print("No results to analyze.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    plot_parameters = ["mapping_method", "sound_encoder", "text_encoder", "dim", "k"]
    
    # Include all relevant metrics
    metrics = [
        'pairwise_distance', 'wasserstein_distance',
        'sound_silhouette_score', 'text_silhouette_score',
        'combined_silhouette_score', 'domain_silhouette_score'
    ]
    
    # Filter to only metrics that exist in the dataframe
    metrics = [m for m in metrics if m in df.columns]
    
    # Create standard parameter impact plots
    for param in plot_parameters:
        if param in df.columns and len(df[param].unique()) > 1:
            for metric in metrics:
                if metric in df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=param, y=metric, data=df)
                    plt.title(f"Impact of {param} on {metric}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plot_file = output_dir / f"{param}_impact_on_{metric}.png"
                    plt.savefig(plot_file)
                    plt.close()
                    print(f"Saved plot: {plot_file}")
    
    # Create specialized plots for clustering metrics
    if 'combined_silhouette_score' in df.columns and 'domain_silhouette_score' in df.columns:
        # Only include clustering experiments
        cluster_df = df[df['mapping_method'] == 'cluster'].copy()
        
        if not cluster_df.empty:
            # Plot relationship between combined silhouette and domain silhouette
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='combined_silhouette_score', 
                y='domain_silhouette_score', 
                hue='k' if 'k' in cluster_df.columns else None,
                size='dim' if 'dim' in cluster_df.columns else None,
                data=cluster_df
            )
            plt.title("Domain Blending vs. Cluster Quality")
            plt.xlabel("Combined Silhouette Score (higher = better clusters)")
            plt.ylabel("Domain Silhouette Score (lower = better blending)")
            plt.tight_layout()
            plot_file = output_dir / "domain_vs_combined_silhouette.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved plot: {plot_file}")
            
            # Plot relationship between distance metrics and clustering quality
            if 'pairwise_distance' in cluster_df.columns and 'combined_silhouette_score' in cluster_df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    x='pairwise_distance', 
                    y='combined_silhouette_score', 
                    hue='k' if 'k' in cluster_df.columns else None,
                    size='dim' if 'dim' in cluster_df.columns else None,
                    data=cluster_df
                )
                plt.title("Relationship Between Pairwise Distance and Cluster Quality")
                plt.xlabel("Pairwise Distance (lower = better mapping)")
                plt.ylabel("Combined Silhouette Score (higher = better clusters)")
                plt.tight_layout()
                plot_file = output_dir / "pairwise_vs_silhouette.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"Saved plot: {plot_file}")
            
            # Compare sound and text clustering quality
            if 'sound_silhouette_score' in cluster_df.columns and 'text_silhouette_score' in cluster_df.columns:
                plt.figure(figsize=(10, 6))
                
                # Create a temporary melted dataframe for this plot
                plot_data = cluster_df[['sound_corpus', 'k', 'sound_silhouette_score', 'text_silhouette_score']].copy()
                plot_data = pd.melt(
                    plot_data, 
                    id_vars=['sound_corpus', 'k'], 
                    value_vars=['sound_silhouette_score', 'text_silhouette_score'],
                    var_name='domain',
                    value_name='silhouette_score'
                )
                
                # Create the grouped bar chart
                sns.barplot(
                    x='k', 
                    y='silhouette_score', 
                    hue='domain',
                    data=plot_data
                )
                plt.title("Sound vs Text Clustering Quality by k")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Silhouette Score (higher = better clusters)")
                plt.tight_layout()
                plot_file = output_dir / "sound_vs_text_clustering.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"Saved plot: {plot_file}")

if __name__ == "__main__":
    filter = {'sound_corpus': 'mothman', 'k': [2], 'mapping_method': 'cluster'}
    filter = None
    analyze_experiments(filter_params=filter)
