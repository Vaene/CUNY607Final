"""
Convert ADL Glossary CSV to JSON Dictionary Format
-------------------------------------------------
Converts the comprehensive glossary CSV into the JSON structure
required by the pipeline's Phase 1 ingestion.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import re

def clean_text(text):
    """Clean and normalize text fields"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_variations(title: str) -> List[str]:
    """Extract variations from term titles (e.g., 'Term1/Term2/Term3')"""
    if pd.isna(title):
        return []
    
    # Split on common separators
    variations = []
    
    # Handle slashes
    if '/' in title:
        parts = [p.strip() for p in title.split('/')]
        variations.extend(parts[1:])  # First part is main term
    
    return variations

def extract_related_terms(summary: str, category: str) -> List[str]:
    """Extract related terms from summary text"""
    if pd.isna(summary):
        return []
    
    related = []
    
    # Look for phrases that indicate related terms
    patterns = [
        r'also known as[:\s]+([^.]+)',
        r'refers to[:\s]+([^.]+)',
        r'similar to[:\s]+([^.]+)',
        r'related to[:\s]+([^.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, summary, re.IGNORECASE)
        for match in matches:
            # Clean and extract potential terms
            terms = [t.strip().strip('"\'') for t in match.split(',')]
            related.extend([t for t in terms if len(t) > 3 and len(t) < 100])
    
    # Limit to reasonable number
    return list(set(related))[:5]

def determine_severity(category: str, priority: str, ideology: str) -> str:
    """Determine severity level based on category and priority"""
    priority = str(priority).lower() if pd.notna(priority) else ''
    
    if 'priority' in priority or 'high' in priority:
        return 'high'
    elif 'medium' in priority:
        return 'medium'
    elif 'low' in priority:
        return 'low'
    else:
        # Default based on category/ideology
        if any(term in str(ideology).lower() for term in ['antisemitism', 'white supremacist']):
            return 'high'
        elif any(term in str(category).lower() for term in ['tactics', 'slogans']):
            return 'medium'
        else:
            return 'medium'

def convert_csv_to_json(csv_path: str, output_path: str = None):
    """
    Convert glossary CSV to JSON dictionary format
    
    Expected JSON structure:
    {
        "term_name": {
            "definition": "...",
            "category": "White Supremacy",
            "variations": ["variant1", "variant2"],
            "related_terms": ["term1", "term2"],
            "context": "...",
            "severity": "high/medium/low",
            "ideology": "...",
            "url": "...",
            "redirect_url": "..."
        }
    }
    """
    
    print(f"Loading glossary from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Filter out empty rows
    df = df[df['Title'].notna()]
    
    print(f"Processing {len(df)} glossary terms...")
    
    # Create dictionary
    glossary_dict = {}
    
    for idx, row in df.iterrows():
        # Get main term (title)
        title = clean_text(row['Title'])
        
        if not title:
            continue
        
        # Extract main term (before any slashes)
        main_term = title.split('/')[0].strip()
        
        # Extract variations
        variations = extract_variations(title)
        
        # Get other fields
        definition = clean_text(row.get('Summary', ''))
        category = clean_text(row.get('Category', 'General Hate'))
        ideology = clean_text(row.get('Ideology', ''))
        url = clean_text(row.get('URL', ''))
        redirect_url = clean_text(row.get('URL to be Redirected to (finalized)', ''))
        priority = clean_text(row.get('PRIORITY', ''))
        notes = clean_text(row.get('Notes', ''))
        
        # Determine severity
        severity = determine_severity(category, priority, ideology)
        
        # Extract related terms from definition
        related_terms = extract_related_terms(definition, category)
        
        # Build term entry
        term_entry = {
            'definition': definition,
            'category': category if category else 'General Hate',
            'variations': variations,
            'related_terms': related_terms,
            'context': notes if notes else '',
            'severity': severity,
            'ideology': ideology if ideology else '',
            'url': url,
            'redirect_url': redirect_url if redirect_url else url
        }
        
        # Add to dictionary
        glossary_dict[main_term] = term_entry
        
        # Also add variations as separate entries pointing to main term
        for variation in variations:
            if variation and variation != main_term:
                glossary_dict[variation] = {
                    **term_entry,
                    'is_variation_of': main_term
                }
    
    print(f"\n✓ Created dictionary with {len(glossary_dict)} entries")
    
    # Count by category
    categories = {}
    for term, data in glossary_dict.items():
        cat = data.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nTerms by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Count by severity
    severities = {}
    for term, data in glossary_dict.items():
        sev = data.get('severity', 'unknown')
        severities[sev] = severities.get(sev, 0) + 1
    
    print(f"\nTerms by severity:")
    for sev, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sev}: {count}")
    
    # Save to JSON
    if output_path is None:
        output_path = 'data/raw/adl_hate_terms.json'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glossary_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved glossary to {output_path}")
    
    # Also save a sample for inspection
    sample_path = output_path.parent / 'adl_hate_terms_sample.json'
    sample = dict(list(glossary_dict.items())[:10])
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved sample (10 terms) to {sample_path}")
    
    return glossary_dict

def validate_json_structure(json_path: str):
    """Validate the generated JSON has correct structure"""
    print(f"\nValidating JSON structure...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_fields = ['definition', 'category']
    optional_fields = ['variations', 'related_terms', 'context', 'severity', 
                      'ideology', 'url', 'redirect_url']
    
    issues = []
    
    for term, details in data.items():
        # Check required fields
        for field in required_fields:
            if field not in details:
                issues.append(f"Term '{term}' missing required field '{field}'")
        
        # Check data types
        if 'variations' in details and not isinstance(details['variations'], list):
            issues.append(f"Term '{term}' has non-list variations")
        
        if 'related_terms' in details and not isinstance(details['related_terms'], list):
            issues.append(f"Term '{term}' has non-list related_terms")
    
    if issues:
        print(f"\n⚠ Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print("✓ JSON structure is valid")
    
    return len(issues) == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ADL Glossary CSV to JSON dictionary format"
    )
    parser.add_argument(
        'input_csv',
        help='Path to input CSV file (COEGlossaryTermsComprehensive.csv)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='data/raw/adl_hate_terms.json',
        help='Path to output JSON file (default: data/raw/adl_hate_terms.json)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate JSON structure after conversion'
    )
    
    args = parser.parse_args()
    
    # Convert
    glossary_dict = convert_csv_to_json(args.input_csv, args.output)
    
    # Validate if requested
    if args.validate:
        validate_json_structure(args.output)
    
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the output file: {args.output}")
    print(f"  2. Check the sample file for correctness")
    print(f"  3. Run the pipeline: python main_pipeline.py")
