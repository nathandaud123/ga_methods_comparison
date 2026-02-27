"""
Detailed analysis of Manuscript.docx Bab 3 against repository implementation
"""

import docx
import os
import re

def safe_print(text, max_len=200):
    """Safely print text with encoding handling"""
    try:
        print(text[:max_len])
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text[:max_len])

def extract_bab3():
    """Extract and analyze Bab 3"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manuscript_path = os.path.join(script_dir, 'Manuscript.docx')
    
    doc = docx.Document(manuscript_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    
    # Find Bab 3 (section 3)
    bab3_start = None
    for i, para in enumerate(paragraphs):
        if para.strip().startswith('3.') and len(para.strip()) < 100:
            bab3_start = i
            break
    
    if bab3_start is None:
        print("Bab 3 not found!")
        return None
    
    # Find end (section 4)
    bab3_end = len(paragraphs)
    for i in range(bab3_start + 1, len(paragraphs)):
        if paragraphs[i].strip().startswith('4.') and len(paragraphs[i].strip()) < 100:
            bab3_end = i
            break
    
    return paragraphs[bab3_start:bab3_end]

def check_objective_function(bab3_text):
    """Check objective function formulation"""
    print("\n" + "="*80)
    print("OBJECTIVE FUNCTION CHECK")
    print("="*80)
    
    # Look for objective function formula
    obj_patterns = [
        r'minimize.*total.*distance',
        r'objective.*function',
        r'min.*\s*Σ',
        r'min.*sum',
    ]
    
    found_patterns = []
    for pattern in obj_patterns:
        matches = re.finditer(pattern, bab3_text, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(bab3_text), match.end() + 200)
            context = bab3_text[start:end]
            found_patterns.append((pattern, context))
    
    print(f"\nFound {len(found_patterns)} objective function mentions:")
    for i, (pattern, context) in enumerate(found_patterns[:5], 1):
        print(f"\n{i}. Pattern: {pattern}")
        safe_print(f"   Context: ...{context}...")
    
    # Repository implementation
    print("\n" + "-"*80)
    print("REPOSITORY IMPLEMENTATION:")
    print("-"*80)
    print("""
    Formula: Total Distance = Σ (distance between consecutive nodes in all routes)
    
    Implementation (src/representation/permutation.py, line 75-82):
    ```python
    def calculate_fitness(self, chromosome, dist_matrix):
        routes = self.decode_to_routes(chromosome)
        total_distance = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total_distance += dist_matrix[route[i]][route[i+1]]
        return total_distance
    ```
    
    Distance calculation (src/data/solomon_parser.py, line 47-48):
    ```python
    dist = sqrt((coords[i][0] - coords[j][0])**2 + 
                (coords[i][1] - coords[j][1])**2)
    ```
    
    Goal: Minimize total_distance (lower is better)
    """)
    
    return found_patterns

def check_ga_parameters(bab3_text):
    """Check GA parameters mentioned"""
    print("\n" + "="*80)
    print("GA PARAMETERS CHECK")
    print("="*80)
    
    # Parameters to check
    params = {
        'population_size': ['population size', 'populasi', 'pop_size'],
        'max_generations': ['max generation', 'maximum generation', 'generasi maksimum'],
        'crossover_rate': ['crossover rate', 'probabilitas crossover', 'pc'],
        'mutation_rate': ['mutation rate', 'probabilitas mutasi', 'pm'],
        'elitism': ['elitism', 'elitisme'],
    }
    
    found_params = {}
    for param, keywords in params.items():
        found_params[param] = []
        for keyword in keywords:
            matches = re.finditer(keyword, bab3_text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(bab3_text), match.end() + 100)
                context = bab3_text[start:end]
                found_params[param].append(context)
    
    print("\nParameters found in manuscript:")
    for param, contexts in found_params.items():
        if contexts:
            print(f"\n  {param}: FOUND ({len(contexts)} mentions)")
            safe_print(f"    Example: ...{contexts[0]}...")
        else:
            print(f"  {param}: NOT FOUND")
    
    # Repository values
    print("\n" + "-"*80)
    print("REPOSITORY VALUES (config.yaml):")
    print("-"*80)
    print("""
    population_size: 80
    max_generations: 500
    crossover_rate: 0.9
    mutation_rate: 0.3
    elitism_rate: 0.0  (DISABLED - no elitism)
    tournament_size: 3
    """)
    
    return found_params

def check_operators(bab3_text):
    """Check GA operators mentioned"""
    print("\n" + "="*80)
    print("GA OPERATORS CHECK")
    print("="*80)
    
    # Operators from repository
    repo_operators = {
        'Representation': ['permutation', 'binary', 'real_valued', 'real-valued'],
        'Selection': ['roulette_wheel', 'roulette wheel', 'tournament', 'rank', 
                     'stochastic_universal', 'stochastic universal', 'SUS',
                     'elitism', 'boltzmann'],
        'Crossover (Permutation)': ['PMX', 'OX', 'CX', 'OBX', 'POS', 'ERX', 'SCX'],
        'Crossover (Binary)': ['single_point', 'single-point', 'two_point', 'two-point',
                              'multi_point', 'multi-point', 'uniform', 'shuffle'],
        'Crossover (Real)': ['SBX', 'BLX', 'BLX-alpha', 'flat'],
        'Mutation (Permutation)': ['swap', 'insert', 'inversion', 'scramble', 'displacement'],
        'Mutation (Binary)': ['bit_flip', 'bit-flip', 'uniform', 'interchanging'],
        'Mutation (Real)': ['gaussian', 'polynomial', 'uniform', 'non_uniform', 'non-uniform'],
    }
    
    found_operators = {}
    for category, operators in repo_operators.items():
        found_operators[category] = []
        for op in operators:
            # Search for operator name
            pattern = r'\b' + re.escape(op) + r'\b'
            matches = re.finditer(pattern, bab3_text, re.IGNORECASE)
            if list(matches):
                found_operators[category].append(op)
    
    print("\nOperators found in manuscript:")
    for category, found in found_operators.items():
        if found:
            print(f"\n  {category}:")
            for op in found:
                print(f"    - {op}")
        else:
            print(f"\n  {category}: NOT FOUND")
    
    return found_operators

def check_evaluation_method(bab3_text):
    """Check evaluation methodology"""
    print("\n" + "="*80)
    print("EVALUATION METHODOLOGY CHECK")
    print("="*80)
    
    # Check for evaluation details
    eval_keywords = {
        'n_runs': ['5 runs', 'five runs', '5 kali', 'lima kali', 'independent runs'],
        'average': ['average', 'rata-rata', 'mean'],
        'generation history': ['generation history', 'history', 'convergence'],
        'solomon': ['solomon', 'benchmark'],
    }
    
    found_eval = {}
    for key, keywords in eval_keywords.items():
        found_eval[key] = []
        for keyword in keywords:
            matches = re.finditer(keyword, bab3_text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(bab3_text), match.end() + 100)
                context = bab3_text[start:end]
                found_eval[key].append(context)
    
    print("\nEvaluation methodology found:")
    for key, contexts in found_eval.items():
        if contexts:
            print(f"\n  {key}: FOUND")
            safe_print(f"    Example: ...{contexts[0]}...")
        else:
            print(f"  {key}: NOT FOUND")
    
    # Repository implementation
    print("\n" + "-"*80)
    print("REPOSITORY EVALUATION METHOD:")
    print("-"*80)
    print("""
    - Each method combination runs 5 times (n_runs = 5)
    - Generation history saved for each run
    - Average fitness and diversity calculated from 5 runs
    - Results saved to CSV and JSON
    - All combinations tested on all Solomon instances
    """)
    
    return found_eval

def main():
    print("="*80)
    print("MANUSCRIPT BAB 3 ANALYSIS")
    print("="*80)
    
    bab3_paras = extract_bab3()
    if bab3_paras is None:
        return
    
    bab3_text = '\n'.join(bab3_paras)
    
    print(f"\nBab 3 extracted: {len(bab3_paras)} paragraphs")
    print(f"Total characters: {len(bab3_text)}")
    
    # Run checks
    check_objective_function(bab3_text)
    check_ga_parameters(bab3_text)
    check_operators(bab3_text)
    check_evaluation_method(bab3_text)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nRecommendations:")
    print("1. Verify objective function formula matches implementation")
    print("2. Check if all GA parameters are documented")
    print("3. Ensure all operators are mentioned")
    print("4. Verify evaluation methodology (5 runs, averaging) is described")
    print("5. Check if Solomon benchmark datasets are mentioned")

if __name__ == '__main__':
    main()

