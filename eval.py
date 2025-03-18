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
                "dim", "distance_metric", "pairwise_score", 
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
            "distance_metric": params.distance_metric
        }
        
        # Add all scores to the row
        for key, value in cleaned_scores.items():
            row[key] = value
        
        # Add the new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Save the updated results
        df.to_csv(self.results_file, index=False)
        
        # Save detailed results as JSON for this specific run
        details_file = self.results_dir / f"{params.filename()}_details.json"
        with open(details_file, "w") as f:
            json.dump({
                "parameters": params.to_string(),
                "scores": scores
            }, f, indent=2)
    
    def find_best_parameters(self, metric="pairwise_score", lower_is_better=True, 
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
    
    def compare_encoders(self, metric="pairwise_score", lower_is_better=True):
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
    
    def plot_parameter_impact(self, parameter, metric="pairwise_score"):
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
        
        # Add best pairwise score if available
        if "pairwise_score" in available_metrics:
            report["best_pairwise"] = self.find_best_parameters("pairwise_score").to_dict(orient="records")[0]
        elif "pairwise_distance" in available_metrics:
            report["best_pairwise"] = self.find_best_parameters("pairwise_distance").to_dict(orient="records")[0]
        
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
            
            # Add average pairwise score if available
            if "pairwise_score" in available_metrics:
                corpus_report["avg_pairwise_score"] = corpus_df["pairwise_score"].mean()
                corpus_report["best_pairwise"] = self.find_best_parameters(
                    "pairwise_score", 
                    sound_corpus=corpus
                ).to_dict(orient="records")[0]
            elif "pairwise_distance" in available_metrics:
                corpus_report["avg_pairwise_distance"] = corpus_df["pairwise_distance"].mean()
                corpus_report["best_pairwise"] = self.find_best_parameters(
                    "pairwise_distance", 
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
            
            # Add average pairwise score if available
            if "pairwise_score" in available_metrics:
                corpus_report["avg_pairwise_score"] = corpus_df["pairwise_score"].mean()
                corpus_report["best_pairwise"] = self.find_best_parameters(
                    "pairwise_score", 
                    text_corpus=corpus
                ).to_dict(orient="records")[0]
            elif "pairwise_distance" in available_metrics:
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
