import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from dataloader.util import encode_image_to_base64
import base64
from PIL import Image
import io

class PromptAnalyzer:
    def __init__(self, json_file_path="face_validation_results.json"):
        self.json_file_path = json_file_path
        self.failed_cases = []
        self.analysis_results = {}
        
    def load_failed_cases(self):
        """Load all unsuccessful predictions from the JSON report."""
        if not os.path.exists(self.json_file_path):
            print(f"JSON file {self.json_file_path} not found!")
            return
            
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract failed cases
        self.failed_cases = [case for case in data if not case.get('is_correct', False)]
        
        print(f"Loaded {len(self.failed_cases)} failed cases out of {len(data)} total cases")
        print(f"Success rate: {((len(data) - len(self.failed_cases)) / len(data)) * 100:.2f}%")
        
    def analyze_failure_patterns(self):
        """Analyze patterns in failed predictions."""
        if not self.failed_cases:
            print("No failed cases to analyze!")
            return
            
        analysis = {
            'error_types': defaultdict(int),
            'pair_types': defaultdict(int),
            'model_response_patterns': defaultdict(int),
            'prediction_errors': defaultdict(int),
            'response_lengths': [],
            'common_failure_words': defaultdict(int)
        }
        
        for case in self.failed_cases:
            # Analyze error types
            gold_standard = case.get('gold_standard', 0)
            model_prediction = case.get('model_prediction')
            pair_type = case.get('pair_type', 'unknown')
            response = case.get('model_response', '')
            
            # Track pair types that fail
            analysis['pair_types'][pair_type] += 1
            
            # Track prediction error types
            if model_prediction is None:
                analysis['prediction_errors']['no_prediction'] += 1
            elif gold_standard == 1 and model_prediction == 0:
                analysis['prediction_errors']['false_negative'] += 1  # Said NO when should be YES
            elif gold_standard == 0 and model_prediction == 1:
                analysis['prediction_errors']['false_positive'] += 1  # Said YES when should be NO
                
            # Analyze response patterns
            response_upper = response.upper()
            if 'YES' in response_upper and 'NO' in response_upper:
                analysis['model_response_patterns']['contradictory'] += 1
            elif 'YES' not in response_upper and 'NO' not in response_upper:
                analysis['model_response_patterns']['no_clear_answer'] += 1
            elif 'UNCERTAIN' in response_upper or 'UNSURE' in response_upper:
                analysis['model_response_patterns']['uncertain'] += 1
                
            # Track response lengths
            analysis['response_lengths'].append(len(response))
            
            # Extract common words in failed responses
            words = response.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    analysis['common_failure_words'][word] += 1
                    
        self.analysis_results = analysis
        return analysis
        
    def generate_failure_report(self):
        """Generate a detailed failure analysis report."""
        if not self.analysis_results:
            self.analyze_failure_patterns()
            
        analysis = self.analysis_results
        
        print("\n" + "="*80)
        print("FAILURE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 FAILURE STATISTICS:")
        print(f"Total failed cases: {len(self.failed_cases)}")
        
        print(f"\n🔍 ERROR TYPES:")
        for error_type, count in analysis['prediction_errors'].items():
            percentage = (count / len(self.failed_cases)) * 100
            print(f"  {error_type}: {count} ({percentage:.1f}%)")
            
        print(f"\n📈 PAIR TYPES THAT FAIL:")
        for pair_type, count in analysis['pair_types'].items():
            percentage = (count / len(self.failed_cases)) * 100
            print(f"  {pair_type}: {count} ({percentage:.1f}%)")
            
        print(f"\n💬 RESPONSE PATTERNS:")
        for pattern, count in analysis['model_response_patterns'].items():
            percentage = (count / len(self.failed_cases)) * 100
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
            
        print(f"\n📝 RESPONSE LENGTH ANALYSIS:")
        lengths = analysis['response_lengths']
        if lengths:
            print(f"  Average response length: {sum(lengths) / len(lengths):.1f} characters")
            print(f"  Min length: {min(lengths)}, Max length: {max(lengths)}")
            
        print(f"\n🔤 MOST COMMON WORDS IN FAILED RESPONSES:")
        most_common = Counter(analysis['common_failure_words']).most_common(10)
        for word, count in most_common:
            print(f"  '{word}': {count}")
            
    def extract_specific_failures(self, limit=10):
        """Extract specific failure cases for detailed analysis."""
        print(f"\n" + "="*80)
        print("SPECIFIC FAILURE EXAMPLES")
        print("="*80)
        
        # Group by failure type
        false_negatives = []
        false_positives = []
        no_predictions = []
        
        for case in self.failed_cases:
            gold_standard = case.get('gold_standard', 0)
            model_prediction = case.get('model_prediction')
            
            if model_prediction is None:
                no_predictions.append(case)
            elif gold_standard == 1 and model_prediction == 0:
                false_negatives.append(case)
            elif gold_standard == 0 and model_prediction == 1:
                false_positives.append(case)
                
        print(f"\n❌ FALSE NEGATIVES (Model said NO, should be YES) - {len(false_negatives)} cases:")
        for i, case in enumerate(false_negatives[:limit]):
            print(f"\n  Case {i+1}:")
            print(f"    Images: {Path(case['image1_path']).name} vs {Path(case['image2_path']).name}")
            print(f"    Model Response: {case['model_response'][:100]}...")
            
        print(f"\n✅ FALSE POSITIVES (Model said YES, should be NO) - {len(false_positives)} cases:")
        for i, case in enumerate(false_positives[:limit]):
            print(f"\n  Case {i+1}:")
            print(f"    Images: {Path(case['image1_path']).name} vs {Path(case['image2_path']).name}")
            print(f"    Model Response: {case['model_response'][:100]}...")
            
        print(f"\n❓ NO CLEAR PREDICTION - {len(no_predictions)} cases:")
        for i, case in enumerate(no_predictions[:limit]):
            print(f"\n  Case {i+1}:")
            print(f"    Images: {Path(case['image1_path']).name} vs {Path(case['image2_path']).name}")
            print(f"    Model Response: {case['model_response'][:100]}...")
            
    def generate_prompt_improvements(self):
        """Generate specific prompt improvement suggestions based on analysis."""
        print(f"\n" + "="*80)
        print("PROMPT IMPROVEMENT RECOMMENDATIONS")
        print("="*80)
        
        analysis = self.analysis_results
        
        # Analyze the main failure types
        false_negatives = analysis['prediction_errors'].get('false_negative', 0)
        false_positives = analysis['prediction_errors'].get('false_positive', 0)
        no_predictions = analysis['prediction_errors'].get('no_prediction', 0)
        
        print(f"\n🔧 SPECIFIC IMPROVEMENTS NEEDED:")
        
        if false_negatives > false_positives:
            print(f"  ⚠️  HIGH FALSE NEGATIVE RATE ({false_negatives} cases)")
            print(f"     Problem: Model is too conservative, missing same-person matches")
            print(f"     Solution: Adjust prompts to be more sensitive to subtle similarities")
            print(f"     Suggestions:")
            print(f"       - Add emphasis on 'even subtle similarities matter'")
            print(f"       - Include 'consider photos may be years apart'")
            print(f"       - Add 'focus on bone structure which doesn't change'")
            
        elif false_positives > false_negatives:
            print(f"  ⚠️  HIGH FALSE POSITIVE RATE ({false_positives} cases)")
            print(f"     Problem: Model is too liberal, seeing similarities where none exist")
            print(f"     Solution: Adjust prompts to be more discriminating")
            print(f"     Suggestions:")
            print(f"       - Add emphasis on 'distinctive differences matter'")
            print(f"       - Include 'similarity is not enough, features must match precisely'")
            print(f"       - Add 'consider ethnic similarity vs individual identity'")
            
        if no_predictions > 0:
            print(f"  ⚠️  NO CLEAR PREDICTIONS ({no_predictions} cases)")
            print(f"     Problem: Model responses are ambiguous")
            print(f"     Solution: Improve response format requirements")
            print(f"     Suggestions:")
            print(f"       - Add 'You must answer with exactly YES or NO'")
            print(f"       - Include 'If uncertain, choose based on strongest evidence'")
            print(f"       - Add examples of proper response format")
            
        # Analyze response patterns
        contradictory = analysis['model_response_patterns'].get('contradictory', 0)
        if contradictory > 0:
            print(f"  ⚠️  CONTRADICTORY RESPONSES ({contradictory} cases)")
            print(f"     Problem: Model gives both YES and NO in same response")
            print(f"     Solution: Improve response structure")
            print(f"     Suggestions:")
            print(f"       - Add 'Give only ONE final answer: YES or NO'")
            print(f"       - Include 'Do not include both YES and NO in response'")
            
        # Generate improved prompt based on analysis
        self.generate_improved_prompt()
        
    def generate_improved_prompt(self):
        """Generate an improved prompt based on the failure analysis."""
        analysis = self.analysis_results
        
        print(f"\n🚀 GENERATED IMPROVED PROMPT:")
        print(f"="*60)
        
        # Base prompt structure
        improved_prompt = """You are an expert facial identification specialist. Analyze these two face images with extreme precision.

CRITICAL ANALYSIS STEPS:
1. Compare bone structure (cheekbones, jawline, forehead shape)
2. Analyze eye geometry (shape, spacing, brow structure)
3. Compare nose characteristics (shape, width, nostril form)
4. Examine mouth and lip structure
5. Assess overall facial proportions and symmetry

IMPORTANT CONSIDERATIONS:"""
        
        # Add specific guidance based on failure patterns
        false_negatives = analysis['prediction_errors'].get('false_negative', 0)
        false_positives = analysis['prediction_errors'].get('false_positive', 0)
        
        if false_negatives > false_positives:
            improved_prompt += """
- Photos may be taken years apart - focus on unchanging bone structure
- Lighting and angles can dramatically change appearance
- Even subtle feature matches may indicate the same person
- Consider age-related changes in weight, hair, skin"""
        else:
            improved_prompt += """
- Ethnic or regional similarity does NOT mean same person
- Features must match precisely, not just be similar
- Pay attention to distinctive differences, even if subtle
- General resemblance is not sufficient evidence"""
            
        improved_prompt += """

RESPONSE FORMAT REQUIREMENTS:
- Answer with exactly "YES" or "NO" only
- Follow with 1-2 sentences explaining your key evidence
- Do not include both YES and NO in your response
- If uncertain, choose based on strongest evidence

Your analysis:"""
        
        print(improved_prompt)
        
        # Save to file
        with open('improved_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(improved_prompt)
        print(f"\n💾 Improved prompt saved to 'improved_prompt.txt'")
        
    def save_analysis_report(self):
        """Save the complete analysis to a file."""
        report_data = {
            'total_failed_cases': len(self.failed_cases),
            'analysis_results': self.analysis_results,
            'failed_cases': self.failed_cases
        }
        
        output_file = 'failure_analysis_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 Complete analysis saved to '{output_file}'")
        
    def run_complete_analysis(self):
        """Run the complete failure analysis pipeline."""
        print("🔍 Loading failed cases...")
        self.load_failed_cases()
        
        print("\n📊 Analyzing failure patterns...")
        self.analyze_failure_patterns()
        
        print("\n📋 Generating failure report...")
        self.generate_failure_report()
        
        print("\n🔍 Extracting specific failures...")
        self.extract_specific_failures()
        
        print("\n💡 Generating prompt improvements...")
        self.generate_prompt_improvements()
        
        print("\n💾 Saving analysis report...")
        self.save_analysis_report()
        
        print(f"\n✅ Analysis complete! Check the generated files for detailed insights.")

def main():
    """Main function to run the prompt analysis."""
    analyzer = PromptAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
