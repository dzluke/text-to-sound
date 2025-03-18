from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import ParameterGenerator
from main import run
from eval import EvaluationFramework

def run_comprehensive_evaluation():
    # Initialize the evaluation framework
    framework = EvaluationFramework()
    
    # Define corpora to test
    sound_corpora = [
        "./corpora/sound/toy",
        "./corpora/sound/anonymous_corpus",
    ]
    
    text_corpora = [
        "./corpora/text/test.txt",
        "./corpora/text/repeat.txt",
    ]

    # sound_encoders = ["MuQ"]
    # text_encoders = ["fastText", "RoBERTa", "word2vec"]
    # mappings = ["identity", "cluster"]
    # sound_preprocessings = [1000, 'onsets', 'full']
    # normalizations = ["standard"]
    # dims = [2, 10, 30]
    # distance_metrics = ["euclidean", "cosine"]
    # mapping_evaluations = ["pairwise"]

    sound_encoders = ["MuQ"]
    text_encoders = ["fastText"]
    mappings = ["cluster"]
    sound_preprocessings = ['full']
    normalizations = ["standard"]
    dims = [2, 5]
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
                mapping_evaluations=mapping_evaluations
            )
            
            # Run each parameter combination
            parameter_list = e.create_params()
            for parameters in parameter_list:
                print(f"Running with parameters: {parameters.to_string()}")
                run(parameters, cache=True, evaluator=framework)
    
    # Generate a report of all evaluations
    report = framework.generate_report()
    print("Evaluation complete! Report generated.")
    
    # Analyze and visualize results
    analyze_results(framework)
    
    return report

def analyze_results(framework):
    """
    Analyze results and generate visualizations.
    """
    print("\n" + "="*50)
    print("ANALYSIS OF EVALUATION RESULTS")
    print("="*50 + "\n")
    
    # Get dataframe
    df = framework.get_results_dataframe()
    
    # 1. Best parameter combinations overall (pairwise score)
    print("\n--- BEST PARAMETER COMBINATIONS (LOWEST PAIRWISE SCORE) ---")
    best_overall = framework.find_best_parameters(metric="pairwise_score")
    print(best_overall.head(5).to_string(index=False))
    
    # 2. Best parameter combinations for clustering quality
    if "combined_silhouette_score" in df.columns:
        print("\n--- BEST PARAMETER COMBINATIONS (HIGHEST COMBINED SILHOUETTE SCORE) ---")
        best_clustering = framework.find_best_parameters(
            metric="combined_silhouette_score", 
            lower_is_better=False
        )
        print(best_clustering.head(5).to_string(index=False))
    
    # 3. Best parameter combinations for each corpus
    sound_corpora = framework.get_unique_values("sound_corpus")
    for corpus in sound_corpora:
        print(f"\n--- BEST PARAMETERS FOR SOUND CORPUS: {corpus} ---")
        best_for_corpus = framework.find_best_parameters(
            metric="pairwise_score", 
            sound_corpus=corpus
        )
        print(best_for_corpus.head(3).to_string(index=False))
    
    # 4. Encoder comparison
    print("\n--- ENCODER PERFORMANCE COMPARISON ---")
    encoder_performance = framework.compare_encoders()
    print(encoder_performance.to_string())
    
    # Generate visualizations
    print("\n--- GENERATING VISUALIZATION PLOTS ---")
    
    # Create output directory for plots
    plots_dir = Path("./evaluation_plots")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot various metrics
    plot_metrics(framework, plots_dir)
    
    print(f"\nAll plots saved to {plots_dir}")
    print("\n" + "="*50)

def plot_metrics(framework, plots_dir):
    """Generate comprehensive plots for all metrics."""
    df = framework.get_results_dataframe()
    
    # 1. Plot encoder comparison for pairwise score
    plot_encoder_comparison(df, "pairwise_score", "Pairwise Score (lower is better)", plots_dir)
    
    # 2. If clustering metrics are available, create plots for them too
    if "combined_silhouette_score" in df.columns:
        plot_encoder_comparison(df, "combined_silhouette_score", 
                               "Combined Silhouette Score (higher is better)", 
                               plots_dir, lower_is_better=False)
    
    # 3. Plot parameter impact on key metrics
    key_parameters = ["sound_encoder", "text_encoder", "mapping_method", 
                     "sound_preprocessing", "dim", "distance_metric"]
    
    for param in key_parameters:
        plot_parameter_impact(df, param, "pairwise_score", 
                            "Pairwise Score (lower is better)", plots_dir)
        
        if "combined_silhouette_score" in df.columns:
            plot_parameter_impact(df, param, "combined_silhouette_score", 
                                "Combined Silhouette Score (higher is better)", 
                                plots_dir, lower_is_better=False)
    
    # 4. Plot corpus performance
    plot_corpus_performance(df, plots_dir)

def plot_encoder_comparison(df, metric, title, plots_dir, lower_is_better=True):
    """Plot comparison of encoder combinations."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x="sound_encoder", 
        y=metric, 
        hue="text_encoder"
    )
    plt.title(f"Performance by Encoder Combination ({title})")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f"encoder_comparison_{metric}.png")
    plt.close()

def plot_parameter_impact(df, parameter, metric, title, plots_dir, lower_is_better=True):
    """Plot impact of a specific parameter on performance."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=parameter, y=metric, data=df)
    plt.title(f"Impact of {parameter} on Performance ({title})")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f"impact_{parameter}_{metric}.png")
    plt.close()

def plot_corpus_performance(df, plots_dir):
    """Plot performance across different corpora."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="sound_corpus", 
        y="pairwise_score", 
        hue="text_corpus", 
        data=df
    )
    plt.title("Performance Across Different Corpora")
    plt.ylabel("Pairwise Score (lower is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "corpus_performance.png")
    plt.close()

if __name__ == "__main__":
    run_comprehensive_evaluation()
