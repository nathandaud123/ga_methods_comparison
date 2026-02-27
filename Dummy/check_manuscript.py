"""
Script to check Manuscript.docx Bab 3 against repository implementation
"""

import docx
import os
import sys

def read_docx(filepath):
    """Read Word document and extract text"""
    try:
        doc = docx.Document(filepath)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return paragraphs
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def find_bab3_section(paragraphs):
    """Find Bab 3 section in document"""
    bab3_start = None
    bab3_end = None
    
    # Try multiple patterns: "3.", "3 ", "Bab 3", "BAB 3", "Chapter 3", etc.
    patterns = [
        lambda t: t.strip().startswith('3.') and len(t.strip()) < 100,  # "3. Method" format
        lambda t: 'BAB 3' in t.upper() or 'BAB III' in t.upper(),
        lambda t: 'CHAPTER 3' in t.upper() or 'CHAPTER III' in t.upper(),
        lambda t: t.strip().startswith('3 ') and len(t.strip()) < 100,
    ]
    
    for i, para in enumerate(paragraphs):
        text_upper = para.upper().strip()
        # Check if it's a section header (short, starts with number)
        for pattern in patterns:
            if pattern(para):
                bab3_start = i
                break
        if bab3_start is not None:
            break
    
    if bab3_start is None:
        return None, None
    
    # Look for next major section (4., Bab 4, etc.)
    for i in range(bab3_start + 1, len(paragraphs)):
        para = paragraphs[i]
        text_upper = para.upper().strip()
        if (para.strip().startswith('4.') and len(para.strip()) < 100) or \
           ('BAB 4' in text_upper or 'BAB IV' in text_upper) or \
           ('CHAPTER 4' in text_upper or 'CHAPTER IV' in text_upper):
            bab3_end = i
            break
    
    if bab3_end is None:
        bab3_end = len(paragraphs)
    
    return bab3_start, bab3_end

def extract_bab3_content(paragraphs, start, end):
    """Extract Bab 3 content"""
    if start is None:
        return []
    return paragraphs[start:end]

def analyze_objective_function(text):
    """Check if objective function is mentioned correctly"""
    text_lower = text.lower()
    
    # Keywords to look for
    keywords = {
        'objective': ['objective', 'fungsi objektif', 'tujuan', 'goal'],
        'distance': ['distance', 'jarak', 'total distance', 'total jarak'],
        'minimize': ['minimize', 'minimisasi', 'minimasi', 'minimum'],
        'formula': ['formula', 'rumus', 'equation', 'persamaan']
    }
    
    findings = {}
    for category, terms in keywords.items():
        findings[category] = []
        for term in terms:
            if term in text_lower:
                # Find context around the term
                idx = text_lower.find(term)
                start = max(0, idx - 100)
                end = min(len(text), idx + 200)
                context = text[start:end]
                findings[category].append({
                    'term': term,
                    'context': context
                })
    
    return findings

def check_ga_structure(text):
    """Check if GA structure is mentioned correctly"""
    text_lower = text.lower()
    
    # Check for GA components
    components = {
        'representation': ['representation', 'representasi', 'encoding', 'kode'],
        'selection': ['selection', 'seleksi', 'pemilihan'],
        'crossover': ['crossover', 'persilangan', 'rekombinasi'],
        'mutation': ['mutation', 'mutasi'],
        'population': ['population', 'populasi'],
        'generation': ['generation', 'generasi']
    }
    
    findings = {}
    for component, terms in components.items():
        findings[component] = False
        for term in terms:
            if term in text_lower:
                findings[component] = True
                break
    
    return findings

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manuscript_path = os.path.join(script_dir, 'Manuscript.docx')
    
    if not os.path.exists(manuscript_path):
        print(f"Error: {manuscript_path} not found!")
        return
    
    print("="*80)
    print("Checking Manuscript.docx - Bab 3")
    print("="*80)
    
    # Read document
    print("\nReading document...")
    paragraphs = read_docx(manuscript_path)
    print(f"Total paragraphs: {len(paragraphs)}")
    
    # Find Bab 3
    print("\nSearching for Bab 3 section...")
    bab3_start, bab3_end = find_bab3_section(paragraphs)
    
    if bab3_start is None:
        print("ERROR: Bab 3 section not found!")
        print("\nFirst 20 paragraphs:")
        for i, para in enumerate(paragraphs[:20]):
            try:
                print(f"{i}: {para[:100]}")
            except UnicodeEncodeError:
                print(f"{i}: [Contains special characters]")
        return
    
    print(f"Bab 3 found: paragraphs {bab3_start} to {bab3_end}")
    
    # Extract Bab 3 content
    bab3_content = extract_bab3_content(paragraphs, bab3_start, bab3_end)
    bab3_text = '\n'.join(bab3_content)
    
    print(f"\nBab 3 contains {len(bab3_content)} paragraphs")
    print(f"Total characters: {len(bab3_text)}")
    
    # Show first few paragraphs of Bab 3
    print("\n" + "="*80)
    print("First 10 paragraphs of Bab 3:")
    print("="*80)
    for i, para in enumerate(bab3_content[:10]):
        try:
            print(f"\n[{bab3_start + i}] {para[:200]}...")
        except UnicodeEncodeError:
            print(f"\n[{bab3_start + i}] [Contains special characters - see output file]")
    
    # Analyze objective function
    print("\n" + "="*80)
    print("Objective Function Analysis:")
    print("="*80)
    obj_findings = analyze_objective_function(bab3_text)
    for category, findings in obj_findings.items():
        print(f"\n{category.upper()}:")
        if findings:
            for finding in findings[:3]:  # Show first 3
                print(f"  - Found '{finding['term']}'")
                print(f"    Context: ...{finding['context'][:150]}...")
        else:
            print("  - NOT FOUND")
    
    # Check GA structure
    print("\n" + "="*80)
    print("GA Structure Components Check:")
    print("="*80)
    ga_findings = check_ga_structure(bab3_text)
    for component, found in ga_findings.items():
        status = "[OK] FOUND" if found else "[X] NOT FOUND"
        print(f"  {component}: {status}")
    
    # Repository implementation summary
    print("\n" + "="*80)
    print("Repository Implementation Summary:")
    print("="*80)
    print("""
    Objective Function:
    - Formula: Total Distance = Σ (distance between consecutive nodes in all routes)
    - Distance calculation: Euclidean distance = sqrt((x1-x2)² + (y1-y2)²)
    - Goal: Minimize total distance
    - Implementation: calculate_fitness() in representation classes
    
    GA Structure:
    1. Representation: permutation, binary, real_valued
    2. Selection: roulette_wheel, tournament, rank, stochastic_universal, elitism, boltzmann
    3. Crossover: PMX, OX, CX, OBX, POS, ERX, SCX (permutation)
                  Single-point, Two-point, Multi-point, Uniform, Shuffle (binary)
                  SBX, BLX-α, Flat (real_valued)
    4. Mutation: Swap, Insert, Inversion, Scramble, Displacement (permutation)
                 Bit Flip, Uniform, Interchanging (binary)
                 Gaussian, Polynomial, Uniform (real_valued)
    5. No elitism (elitism_rate = 0.0)
    6. Population size: 80 (configurable)
    7. Max generations: 500 (configurable)
    8. Crossover rate: 0.9 (configurable)
    9. Mutation rate: 0.3 (configurable)
    
    Evaluation:
    - Each combination runs 5 times (n_runs = 5)
    - Average of 5 runs is saved
    - Generation history saved for each run
    """)
    
    # Save Bab 3 to text file for easier review
    output_file = os.path.join(script_dir, 'bab3_extracted.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BAB 3 - EXTRACTED FROM MANUSCRIPT.DOCX\n")
        f.write("="*80 + "\n\n")
        for i, para in enumerate(bab3_content):
            f.write(f"[Paragraph {bab3_start + i}]\n")
            f.write(f"{para}\n\n")
    
    print(f"\n✓ Bab 3 content saved to: {output_file}")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

