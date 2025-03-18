from pathlib import Path
from util import ParameterGenerator, Parameter
from main import run
from eval import EvaluationFramework

def run_comprehensive_evaluation():
    # Initialize the evaluation framework
    framework = EvaluationFramework()
    
    # Define corpora to test
    sound_corpora = [
        "./corpora/sound/toy",
        "./corpora/sound/corpus1",
        "./corpora/sound/corpus2",
    ]
    
    text_corpora = [
        "./corpora/text/test.txt",
        "./corpora/text/corpus1.txt",
        "./corpora/text/corpus2.txt",
    ]
    
    # For each corpus combination, run evaluations
    for sound_corpus in sound_corpora:
        for text_corpus in text_corpora:
            print(f"Evaluating with sound corpus: {Path(sound_corpus).stem} and text corpus: {Path(text_corpus).stem}")
            
            # Create parameter combinations
            e = ParameterGenerator(
                sound_path=sound_corpus,
                text_path=text_corpus,
                sound_encoders=["MuQ"],
                text_encoders=["fastText", "RoBERTa", "word2vec"],
                mappings=["identity", "cluster"],
                sound_preprocessings=[1000, 'onsets', 'full'],
                normalizations=["standard"],
                dims=[2, 10, 30],
                distance_metrics=["euclidean", "cosine"],
                mapping_evaluations=["pairwise"]
            )
            
            # Run each parameter combination
            parameter_list = e.create_params()
            for parameters in parameter_list:
                print(f"Running with parameters: {parameters.to_string()}")
                run(parameters, cache=True, evaluator=framework)
    
    # Generate a report of all evaluations
    report = framework.generate_report()
    print("Evaluation complete! Report generated.")
    
    return report

if __name__ == "__main__":
    run_comprehensive_evaluation()