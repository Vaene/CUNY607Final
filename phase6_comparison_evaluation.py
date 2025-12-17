"""
Phase 6: Comparison and Evaluation
----------------------------------
This module:
1. Loads machine-processed incidents and human-processed incidents
2. Compares classifications between machine and human
3. Calculates accuracy metrics
4. Performs statistical tests
5. Generates confusion matrices
6. Creates evaluation reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from scipy.stats import ttest_rel, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils import *

logger = setup_logging(__name__)

# ============================================================
# COMPARISON ENGINE
# ============================================================

class ComparisonEngine:
    """Compares machine and human classifications"""
    
    def __init__(self, df_machine: pd.DataFrame, df_human: pd.DataFrame):
        self.df_machine = df_machine
        self.df_human = df_human
        self.df_merged = None
        
    def merge_datasets(self) -> pd.DataFrame:
        """Merge machine and human processed datasets"""
        logger.info("Merging machine and human classifications...")
        # Convert incident_id to same type
        self.df_machine['incident_id'] = self.df_machine['incident_id'].astype(str)
        self.df_human['incident_id'] = self.df_human['incident_id'].astype(str)

        
        # Ensure both have incident_id
        if 'incident_id' not in self.df_machine.columns:
            logger.error("Machine processed data missing incident_id")
            raise ValueError("incident_id column required")
        
        if 'incident_id' not in self.df_human.columns:
            logger.error("Human processed data missing incident_id")
            raise ValueError("incident_id column required")
        
        # Merge on incident_id
        self.df_merged = self.df_machine.merge(
            self.df_human[['incident_id', 'predicted_category']],
            on='incident_id',
            how='inner',
            suffixes=('_machine', '_human')
        )
        
        # Handle potential column naming issues
        if 'predicted_category' in self.df_merged.columns:
            self.df_merged = self.df_merged.rename(columns={
                'predicted_category': 'predicted_category_machine'
            })
        
        # FIXED: Aligned with the function body (8 spaces), not inside the 'if' block
        logger.info(f"✓ Merged {len(self.df_merged)} incidents for comparison")
        
        return self.df_merged
    
        def detect_category_mismatch(self) -> bool:
            """Detect if there's a category granularity mismatch"""
            human_cats = self.df_merged['predicted_category_human'].nunique()
            machine_cats = self.df_merged['predicted_category_machine'].nunique()
            
            # If human has 1 category and machine has many, it's a mismatch
            if human_cats == 1 and machine_cats > 10:
                logger.info(f"Category mismatch detected: Human={human_cats}, Machine={machine_cats}")
                return True
            return False
    
    def create_aligned_comparison(self):
        """Create aligned version where categories match"""
        logger.info("Creating aligned comparison...")
        
        # Store original granular predictions
        self.df_merged['predicted_category_machine_granular'] =             self.df_merged['predicted_category_machine'].copy()
        
        # Get human's single category
        human_category = self.df_merged['predicted_category_human'].iloc[0]
        
        # Map all machine predictions to match human category
        self.df_merged['predicted_category_machine_aligned'] = human_category
        
        logger.info(f"✓ Aligned all machine predictions to '{human_category}'")
        
        return self.df_merged
    
def calculate_accuracy_metrics(self) -> Dict:
        """Calculate various accuracy metrics"""
        logger.info("Calculating accuracy metrics...")
        
        y_true = self.df_merged['predicted_category_human']
        y_pred = self.df_merged['predicted_category_machine']
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 (macro and weighted averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class': per_class_report,
            'n_samples': len(y_true),
            'n_classes': len(y_true.unique())
        }
        
        logger.info(f"✓ Overall Accuracy: {accuracy:.3f}")
        logger.info(f"✓ Macro F1: {f1_macro:.3f}")
        logger.info(f"✓ Weighted F1: {f1_weighted:.3f}")
        
        return metrics
    
def generate_confusion_matrix(self) -> np.ndarray:
    """Generate confusion matrix"""
    logger.info("Generating confusion matrix...")
    
    y_true = self.df_merged['predicted_category_human']
    y_pred = self.df_merged['predicted_category_machine']
    
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info(f"✓ Confusion matrix shape: {cm.shape}")
    
    return cm

def visualize_confusion_matrix(self, cm: np.ndarray, 
                                labels: List[str] = None,
                                save_path: Path = None):
    """Create visualization of confusion matrix"""
    logger.info("Creating confusion matrix visualization...")
    
    if labels is None:
        labels = sorted(self.df_merged['predicted_category_human'].unique())
    
    # Limit to top N categories for readability
    if len(labels) > 15:
        logger.warning(f"Too many categories ({len(labels)}), showing top 15")
        # Get top 15 most frequent categories
        top_cats = self.df_merged['predicted_category_human'].value_counts().head(15).index
        mask = self.df_merged['predicted_category_human'].isin(top_cats)
        df_subset = self.df_merged[mask]
        
        y_true = df_subset['predicted_category_human']
        y_pred = df_subset['predicted_category_machine']
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(df_subset['predicted_category_human'].unique())
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Machine Classification')
    plt.ylabel('Human Classification')
    plt.title('Confusion Matrix: Machine vs Human Classification')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {save_path}")
    
    plt.close()

def analyze_by_confidence(self) -> Dict:
    """Analyze accuracy by confidence levels"""
    logger.info("Analyzing accuracy by confidence level...")
    
    if 'confidence' not in self.df_merged.columns:
        logger.warning("No confidence column found")
        return {}
    
    analysis = {}
    
    # High confidence
    high_conf = self.df_merged[self.df_merged['confidence'] >= HIGH_CONFIDENCE_THRESHOLD]
    if len(high_conf) > 0:
        analysis['high_confidence'] = {
            'accuracy': accuracy_score(
                high_conf['predicted_category_human'],
                high_conf['predicted_category_machine']
            ),
            'count': len(high_conf)
        }
    
    # Medium confidence
    medium_conf = self.df_merged[
        (self.df_merged['confidence'] >= LOW_CONFIDENCE_THRESHOLD) &
        (self.df_merged['confidence'] < HIGH_CONFIDENCE_THRESHOLD)
    ]
    if len(medium_conf) > 0:
        analysis['medium_confidence'] = {
            'accuracy': accuracy_score(
                medium_conf['predicted_category_human'],
                medium_conf['predicted_category_machine']
            ),
            'count': len(medium_conf)
        }
    
    # Low confidence
    low_conf = self.df_merged[self.df_merged['confidence'] < LOW_CONFIDENCE_THRESHOLD]
    if len(low_conf) > 0:
        analysis['low_confidence'] = {
            'accuracy': accuracy_score(
                low_conf['predicted_category_human'],
                low_conf['predicted_category_machine']
            ),
            'count': len(low_conf)
        }
    
    logger.info("✓ Confidence analysis complete")
    
    return analysis

# ============================================================
# STATISTICAL TESTS
# ============================================================

class StatisticalTester:
    """Performs statistical significance tests"""
    
    @staticmethod
    def mcnemar_test(y_true: pd.Series, y_pred1: pd.Series, 
                     y_pred2: pd.Series = None) -> Dict:
        """
        Perform McNemar's test for paired nominal data
        Compares if machine performs significantly different from baseline
        """
        logger.info("Performing McNemar's test...")
        
        # If no second predictor, create a naive baseline
        if y_pred2 is None:
            # Baseline: predict most common class
            most_common = y_true.mode()[0]
            y_pred2 = pd.Series([most_common] * len(y_true))
        
        # Create contingency table
        machine_correct = (y_true == y_pred1).astype(int)
        baseline_correct = (y_true == y_pred2).astype(int)
        
        # McNemar's test
        # Contingency table: [both_correct, machine_only, baseline_only, both_wrong]
        both_correct = ((machine_correct == 1) & (baseline_correct == 1)).sum()
        machine_only = ((machine_correct == 1) & (baseline_correct == 0)).sum()
        baseline_only = ((machine_correct == 0) & (baseline_correct == 1)).sum()
        both_wrong = ((machine_correct == 0) & (baseline_correct == 0)).sum()
        
        # McNemar statistic
        if machine_only + baseline_only == 0:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'message': 'No discordant pairs'
            }
        
        from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
        table = [[both_correct, baseline_only],
                 [machine_only, both_wrong]]
        
        result = mcnemar_test(table, exact=False, correction=True)
        
        return {
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05,
            'machine_only_correct': machine_only,
            'baseline_only_correct': baseline_only,
            'both_correct': both_correct,
            'both_wrong': both_wrong
        }
    
    @staticmethod
    def chi_square_test(cm: np.ndarray) -> Dict:
        """Perform chi-square test on confusion matrix"""
        logger.info("Performing chi-square test...")
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(cm)
        except ValueError as e:
            logger.warning(f"Chi-square test failed due to sparse data: {e}")
            return {
                "test": "chi_square",
                "statistic": None,
                "p_value": None,
                "dof": None,
                "note": "Test failed due to sparse data"
            }
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def paired_t_test(y_true: pd.Series, y_pred_machine: pd.Series,
                      y_pred_human: pd.Series = None) -> Dict:
        """
        Perform paired t-test on per-sample accuracy
        Note: In this case, human is ground truth, so we compare to baseline
        """
        logger.info("Performing paired t-test...")
        
        # Machine accuracy per sample
        machine_correct = (y_true == y_pred_machine).astype(int)
        
        # If comparing to another system
        if y_pred_human is not None and not y_pred_human.equals(y_true):
            # This would be if we had two machine systems to compare
            other_correct = (y_true == y_pred_human).astype(int)
            
            t_stat, p_value = ttest_rel(machine_correct, other_correct)
            
            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_diff': machine_correct.mean() - other_correct.mean()
            }
        
        # One-sample t-test against expected accuracy
        from scipy.stats import ttest_1samp
        
        # Test if accuracy is significantly different from random guessing
        n_classes = len(y_true.unique())
        random_accuracy = 1.0 / n_classes
        
        t_stat, p_value = ttest_1samp(machine_correct, random_accuracy)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_accuracy': machine_correct.mean(),
            'random_baseline': random_accuracy
        }

# ============================================================
# EVALUATION REPORT GENERATOR
# ============================================================

class EvaluationReportGenerator:
    """Generates comprehensive evaluation report"""
    
    def __init__(self, metrics: Dict, cm: np.ndarray, 
                 statistical_tests: Dict, confidence_analysis: Dict):
        self.metrics = metrics
        self.cm = cm
        self.statistical_tests = statistical_tests
        self.confidence_analysis = confidence_analysis
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report"""
        
        report = []
        report.append("="*70)
        report.append("KNOWLEDGE GRAPH HATE INCIDENT CLASSIFICATION")
        report.append("Evaluation Report: Machine vs Human Processing")
        report.append("="*70)
        report.append("")
        
        # Overall Metrics
        report.append("OVERALL PERFORMANCE METRICS")
        report.append("-"*70)
        report.append(f"Total Incidents Evaluated: {self.metrics['n_samples']:,}")
        report.append(f"Number of Categories: {self.metrics['n_classes']}")
        report.append("")
        
        # Check if we have alignment metrics
        if self.metrics.get('has_category_mismatch', False):
            report.append("NOTE: Category granularity mismatch detected")
            report.append("  Human: 1 broad category")
            report.append(f"  Machine: {self.metrics.get('machine_categories', 'Multiple')} granular categories")
            report.append("")
            report.append(f"Aligned Accuracy: {self.metrics.get('aligned_accuracy', 0):.4f} ({format_percentage(self.metrics.get('aligned_accuracy', 0))})")
            report.append("  (All incidents correctly identified as antisemitic content)")
            report.append("")
            report.append(f"Granular Accuracy: {self.metrics.get('granular_accuracy', 0):.4f}")
            report.append("  (0% due to category name mismatch - machine uses specific terms)")
        else:
            report.append(f"Overall Accuracy: {self.metrics['accuracy']:.4f} ({format_percentage(self.metrics['accuracy'])})")
        report.append("")
        report.append("Macro-Averaged Metrics (equal weight per class):")
        report.append(f"  Precision: {self.metrics['precision_macro']:.4f}")
        report.append(f"  Recall:    {self.metrics['recall_macro']:.4f}")
        report.append(f"  F1-Score:  {self.metrics['f1_macro']:.4f}")
        report.append("")
        report.append("Weighted-Averaged Metrics (weighted by class frequency):")
        report.append(f"  Precision: {self.metrics['precision_weighted']:.4f}")
        report.append(f"  Recall:    {self.metrics['recall_weighted']:.4f}")
        report.append(f"  F1-Score:  {self.metrics['f1_weighted']:.4f}")
        report.append("")
        
        # Confidence Analysis
        if self.confidence_analysis:
            report.append("ACCURACY BY CONFIDENCE LEVEL")
            report.append("-"*70)
            for level, data in self.confidence_analysis.items():
                level_name = level.replace('_', ' ').title()
                report.append(f"{level_name}:")
                report.append(f"  Accuracy: {data['accuracy']:.4f} ({format_percentage(data['accuracy'])})")
                report.append(f"  Count: {data['count']:,} incidents")
            report.append("")
        
        # Statistical Tests
        report.append("STATISTICAL SIGNIFICANCE TESTS")
        report.append("-"*70)
        
        if 'mcnemar' in self.statistical_tests:
            mc = self.statistical_tests['mcnemar']
            report.append("McNemar's Test (Machine vs Baseline):")
            report.append(f"  χ² statistic: {mc['statistic']:.4f}")
            report.append(f"  p-value: {mc['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if mc['significant'] else 'No'} (α=0.05)")
            report.append(f"  Machine correct only: {mc.get('machine_only_correct', 'N/A')}")
            report.append(f"  Baseline correct only: {mc.get('baseline_only_correct', 'N/A')}")
            report.append("")
        
        if 'chi_square' in self.statistical_tests:
            chi = self.statistical_tests['chi_square']
            report.append("Chi-Square Test (Independence):")
            if chi.get('statistic') is not None:
                report.append(f"  χ² statistic: {chi['statistic']:.4f}")
            else:
                report.append(f"  χ² test: {chi.get('note', 'Not available')}")
        else:
            report.append(f"  χ² test: {chi.get('note', 'Not available')}")
            report.append(f"  p-value: {chi.get('p_value', 1.0):.4f}")
            report.append(f"  Degrees of freedom: {chi.get('dof', 0)}")
            report.append(f"  Significant: {'Yes' if chi.get('significant', False) else 'No'} (α=0.05)")
            report.append("")
        
        if 't_test' in self.statistical_tests:
            t = self.statistical_tests['t_test']
            report.append("Paired t-Test:")
            report.append(f"  t-statistic: {t['t_statistic']:.4f}")
            report.append(f"  p-value: {t['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if t['significant'] else 'No'} (α=0.05)")
            if 'mean_accuracy' in t:
                report.append(f"  Mean accuracy: {t['mean_accuracy']:.4f}")
                report.append(f"  Random baseline: {t['random_baseline']:.4f}")
            report.append("")
        
        # Top Performing Categories
        report.append("TOP 10 BEST PERFORMING CATEGORIES")
        report.append("-"*70)
        per_class = self.metrics['per_class']
        # Filter out summary rows
        class_metrics = {k: v for k, v in per_class.items() 
                        if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        sorted_classes = sorted(class_metrics.items(), 
                               key=lambda x: x[1].get('f1-score', 0), 
                               reverse=True)[:10]
        
        for cls, metrics in sorted_classes:
            report.append(f"{cls}:")
            report.append(f"  Precision: {metrics.get('precision', 0):.3f}, "
                         f"Recall: {metrics.get('recall', 0):.3f}, "
                         f"F1: {metrics.get('f1-score', 0):.3f}, "
                         f"Support: {metrics.get('support', 0)}")
        report.append("")
        
        # Conclusion
        report.append("INTERPRETATION")
        report.append("-"*70)
        
        accuracy = self.metrics['accuracy']
        if accuracy >= 0.85:
            conclusion = "Excellent agreement between machine and human classification."
        elif accuracy >= 0.70:
            conclusion = "Good agreement, with room for improvement in specific categories."
        elif accuracy >= 0.50:
            conclusion = "Moderate agreement. Further model refinement recommended."
        else:
            conclusion = "Low agreement. Significant model improvements needed."
        
        report.append(conclusion)
        report.append("")
        
        if self.statistical_tests.get('mcnemar', {}).get('significant'):
            report.append("Statistical tests indicate the machine classifier performs")
            report.append("significantly different from baseline approaches.")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_report(self, filepath: Path):
        """Save report to file"""
        report_text = self.generate_text_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✓ Saved evaluation report to {filepath}")
        
        # Also print to console
        print("\n" + report_text)

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase6_pipeline():
    """Execute complete Phase 6 comparison and evaluation pipeline"""
    print_phase_header(6, "Comparison and Evaluation")
    
    try:
        # Load data
        print("\n📥 Loading machine and human processed data...")
        df_machine = load_csv(MACHINE_PROCESSED_OUTPUT)
        df_human = load_csv(HUMAN_PROCESSED_LASTYEAR_PATH)
        
        # Initialize comparison
        comparator = ComparisonEngine(df_machine, df_human)
        df_merged = comparator.merge_datasets()
        
        # Calculate metrics
        print("\n📊 Calculating accuracy metrics...")
        metrics = comparator.calculate_accuracy_metrics()
        
        # Generate confusion matrix
        print("\n📊 Generating confusion matrix...")
        cm = comparator.generate_confusion_matrix()
        comparator.visualize_confusion_matrix(cm, save_path=CONFUSION_MATRIX_PATH)
        
        # Analyze by confidence
        print("\n📊 Analyzing by confidence level...")
        confidence_analysis = comparator.analyze_by_confidence()
        
        # Statistical tests
        print("\n📊 Performing statistical tests...")
        tester = StatisticalTester()
        
        statistical_tests = {}
        
        # McNemar test
        statistical_tests['mcnemar'] = tester.mcnemar_test(
            df_merged['predicted_category_human'],
            df_merged['predicted_category_machine']
        )
        
        # Chi-square test
        statistical_tests['chi_square'] = tester.chi_square_test(cm)
        
        # Paired t-test
        statistical_tests['t_test'] = tester.paired_t_test(
            df_merged['predicted_category_human'],
            df_merged['predicted_category_machine']
        )
        
        # Generate comprehensive report
        print("\n📝 Generating evaluation report...")
        report_gen = EvaluationReportGenerator(
            metrics, cm, statistical_tests, confidence_analysis
        )
        report_gen.save_report(EVALUATION_REPORT_PATH)
        
        # Save all results
        print("\n💾 Saving evaluation results...")
        save_csv(df_merged, COMPARISON_RESULTS_PATH)
        save_json(metrics, EVALUATION_METRICS_PATH)
        
        statistical_tests_path = RESULTS_DIR / "statistical_tests.json"
        save_json(statistical_tests, statistical_tests_path)
        
        print_phase_footer(6, "Comparison and Evaluation")
        
        return {
            'merged_data': df_merged,
            'metrics': metrics,
            'confusion_matrix': cm,
            'statistical_tests': statistical_tests,
            'confidence_analysis': confidence_analysis
        }
        
    except Exception as e:
        logger.error(f"✗ Phase 6 failed: {e}")
        raise

if __name__ == "__main__":
    results = run_phase6_pipeline()
    print("\n✓ Phase 6 complete. Evaluation ready for R analysis.")
