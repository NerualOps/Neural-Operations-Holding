"""
Epsilon AI Metadata Extractor
-----------------------
Extracts structured metadata from documents to enable fast knowledge retrieval
and understanding. Creates a "brain" of knowledge that Epsilon AI can query instantly.
"""

import re
import json
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EpsilonMetadataExtractor:
    """Extract structured metadata from documents for fast retrieval"""
    
    def __init__(self):
        self.concepts: Dict[str, Dict] = {}  # concept -> {definition, examples, related}
        self.facts: List[Dict] = []  # Structured facts
        self.processes: List[Dict] = []  # Step-by-step processes
        self.rules: List[Dict] = []  # Rules and guidelines
        self.examples: List[Dict] = []  # Examples and case studies
        self.key_points: List[Dict] = []  # Key insights
        
    def extract_metadata(self, content: str, document_type: str, category: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from document content"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return {
                'concepts': {},
                'facts': [],
                'processes': [],
                'rules': [],
                'examples': [],
                'key_points': [],
                'definitions': {},
                'relationships': {},
                'language_patterns': [],
                'business_insights': [],
                'sales_insights': []
            }
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        if not isinstance(document_type, str):
            document_type = 'general'
        if not isinstance(category, str):
            category = 'general'
        
        metadata = {
            'concepts': {},
            'facts': [],
            'processes': [],
            'rules': [],
            'examples': [],
            'key_points': [],
            'definitions': {},
            'relationships': {},
            'language_patterns': [],
            'business_insights': [],
            'sales_insights': []
        }
        
        # Extract definitions (X is Y, X means Y, X refers to)
        definitions = self._extract_definitions(content)
        metadata['definitions'] = definitions
        metadata['concepts'] = {k: {'definition': v} for k, v in definitions.items()}
        
        # Extract facts (statements of truth)
        facts = self._extract_facts(content)
        metadata['facts'] = facts
        
        # Extract processes (step-by-step, numbered lists)
        processes = self._extract_processes(content)
        metadata['processes'] = processes
        
        # Extract rules (should, must, always, never)
        rules = self._extract_rules(content)
        metadata['rules'] = rules
        
        # Extract examples (for example, such as, case study)
        examples = self._extract_examples(content)
        metadata['examples'] = examples
        
        # Extract key points (important, key, critical, essential)
        key_points = self._extract_key_points(content)
        metadata['key_points'] = key_points
        
        # Extract language patterns (for sales documents)
        if category in ['sales_training', 'communication_guide']:
            language_patterns = self._extract_language_patterns(content)
            metadata['language_patterns'] = language_patterns
        
        # Extract business insights
        if category == 'knowledge':
            business_insights = self._extract_business_insights(content)
            metadata['business_insights'] = business_insights
        
        # Extract sales insights
        if category in ['sales_training', 'communication_guide']:
            sales_insights = self._extract_sales_insights(content)
            metadata['sales_insights'] = sales_insights
        
        # Build relationships between concepts
        relationships = self._build_relationships(content, definitions)
        metadata['relationships'] = relationships
        
        return metadata
    
    def _extract_definitions(self, content: str) -> Dict[str, str]:
        """Extract term definitions"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return {}
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        definitions = {}
        
        # Pattern: "X is Y", "X means Y", "X refers to Y", "X: Y"
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([^\.]+)\.',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+means\s+([^\.]+)\.',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+refers\s+to\s+([^\.]+)\.',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s+([^\.]+)\.'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                term = match[0].strip()
                definition = match[1].strip()
                if len(definition) > 10 and len(definition) < 200:
                    definitions[term.lower()] = definition
        
        return definitions
    
    def _extract_facts(self, content: str) -> List[Dict]:
        """Extract factual statements"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        facts = []
        
        # Pattern: Sentences with numbers, statistics, or definitive statements
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Look for facts (contains numbers, percentages, or definitive language)
            if (re.search(r'\d+%|\d+\s+(percent|million|billion|thousand)', sentence, re.IGNORECASE) or
                re.search(r'\b(is|are|was|were|has|have|can|will)\s+', sentence, re.IGNORECASE)):
                
                facts.append({
                    'statement': sentence,
                    'type': 'statistical' if re.search(r'\d+', sentence) else 'declarative'
                })
        
        return facts[:50]  # Limit to top 50
    
    def _extract_processes(self, content: str) -> List[Dict]:
        """Extract step-by-step processes"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        processes = []
        
        # Pattern: Numbered lists, "Step 1", "First", "Then", "Finally"
        numbered_pattern = r'(?:Step\s+)?(\d+)[\.\)]\s+([^\.]+)\.'
        matches = re.findall(numbered_pattern, content, re.IGNORECASE)
        
        if matches:
            steps = []
            for num, step in matches:
                steps.append({
                    'step_number': int(num),
                    'description': step.strip()
                })
            
            if len(steps) >= 3:  # Valid process has at least 3 steps
                processes.append({
                    'steps': steps,
                    'total_steps': len(steps)
                })
        
        # Pattern: "First... Then... Finally"
        sequence_pattern = r'(?:First|Initially|To begin)[^\.]+\.\s+(?:Then|Next|After that)[^\.]+\.\s+(?:Finally|Lastly|In conclusion)[^\.]+\.'
        sequence_matches = re.findall(sequence_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in sequence_matches:
            sentences = re.split(r'[.!?]+', match)
            steps = []
            for i, sentence in enumerate(sentences[:5], 1):  # Max 5 steps
                if sentence.strip():
                    steps.append({
                        'step_number': i,
                        'description': sentence.strip()
                    })
            
            if len(steps) >= 3:
                processes.append({
                    'steps': steps,
                    'total_steps': len(steps)
                })
        
        return processes[:10]  # Limit to top 10 processes
    
    def _extract_rules(self, content: str) -> List[Dict]:
        """Extract rules and guidelines"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        rules = []
        
        # Pattern: "should", "must", "always", "never", "rule", "guideline"
        rule_patterns = [
            r'([^\.]+(?:should|must|always|never|rule|guideline)[^\.]+)\.',
            r'Rule\s+\d+[\.:]?\s+([^\.]+)\.',
            r'Guideline[^\.]+:\s+([^\.]+)\.'
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                rule_text = match.strip() if isinstance(match, str) else match[0].strip()
                if 20 < len(rule_text) < 200:
                    rules.append({
                        'rule': rule_text,
                        'type': 'guideline' if 'guideline' in rule_text.lower() else 'rule'
                    })
        
        return rules[:30]  # Limit to top 30 rules
    
    def _extract_examples(self, content: str) -> List[Dict]:
        """Extract examples and case studies"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        examples = []
        
        # Pattern: "for example", "such as", "case study", "instance"
        example_patterns = [
            r'(?:For example|For instance|Such as)[^\.]+:?\s+([^\.]+)\.',
            r'Case\s+study[^\.]+:\s+([^\.]+)\.',
            r'Example[^\.]+:\s+([^\.]+)\.'
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                example_text = match.strip()
                if 15 < len(example_text) < 300:
                    examples.append({
                        'example': example_text,
                        'type': 'case_study' if 'case study' in content.lower() else 'example'
                    })
        
        return examples[:20]  # Limit to top 20 examples
    
    def _extract_key_points(self, content: str) -> List[Dict]:
        """Extract key points and insights"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        key_points = []
        
        # Pattern: "important", "key", "critical", "essential", "main point"
        key_patterns = [
            r'([^\.]+(?:important|key|critical|essential|main point|crucial)[^\.]+)\.',
            r'Key\s+point[^\.]+:\s+([^\.]+)\.',
            r'Main\s+point[^\.]+:\s+([^\.]+)\.'
        ]
        
        for pattern in key_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                point_text = match.strip()
                if 20 < len(point_text) < 250:
                    key_points.append({
                        'point': point_text,
                        'importance': 'high' if 'critical' in point_text.lower() or 'essential' in point_text.lower() else 'medium'
                    })
        
        return key_points[:25]  # Limit to top 25 key points
    
    def _extract_language_patterns(self, content: str) -> List[Dict]:
        """Extract language patterns for sales/communication"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        patterns = []
        
        # Extract phrases that show communication style
        phrases = re.findall(r'"([^"]+)"', content)  # Quoted phrases
        for phrase in phrases:
            if 10 < len(phrase) < 100:
                patterns.append({
                    'phrase': phrase,
                    'type': 'communication_pattern',
                    'context': 'sales'
                })
        
        # Extract questions (for objection handling)
        questions = re.findall(r'([^\.]+\?)', content)
        for question in questions[:10]:  # Top 10 questions
            if 15 < len(question) < 150:
                patterns.append({
                    'phrase': question.strip(),
                    'type': 'question',
                    'context': 'sales'
                })
        
        return patterns
    
    def _extract_business_insights(self, content: str) -> List[Dict]:
        """Extract business insights and strategies"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        insights = []
        
        # Pattern: Strategic statements, recommendations
        insight_patterns = [
            r'([^\.]+(?:strategy|approach|method|technique|best practice)[^\.]+)\.',
            r'Recommendation[^\.]+:\s+([^\.]+)\.',
            r'Best\s+practice[^\.]+:\s+([^\.]+)\.'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                insight_text = match.strip()
                if 25 < len(insight_text) < 300:
                    insights.append({
                        'insight': insight_text,
                        'type': 'strategy' if 'strategy' in insight_text.lower() else 'insight'
                    })
        
        return insights[:15]  # Limit to top 15 insights
    
    def _extract_sales_insights(self, content: str) -> List[Dict]:
        """Extract sales-specific insights"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return []
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        
        insights = []
        
        # Pattern: Sales techniques, customer psychology, closing strategies
        sales_patterns = [
            r'([^\.]+(?:customer|client|prospect|close|objection|relationship)[^\.]+)\.',
            r'Sales\s+technique[^\.]+:\s+([^\.]+)\.',
            r'Customer\s+psychology[^\.]+:\s+([^\.]+)\.'
        ]
        
        for pattern in sales_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                insight_text = match.strip()
                if 25 < len(insight_text) < 300:
                    insights.append({
                        'insight': insight_text,
                        'type': 'sales_technique',
                        'context': 'sales'
                    })
        
        return insights[:15]  # Limit to top 15 insights
    
    def _build_relationships(self, content: str, definitions: Dict[str, str]) -> Dict[str, List[str]]:
        """Build relationships between concepts"""
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            return {}
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            content = content[:10000000]
        if not definitions or not isinstance(definitions, dict):
            return {}
        
        relationships = defaultdict(list)
        
        # Find co-occurrences (concepts mentioned together)
        defined_terms = list(definitions.keys())
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            mentioned_terms = [term for term in defined_terms if term in sentence_lower]
            
            # Create relationships between co-mentioned terms
            for i, term1 in enumerate(mentioned_terms):
                for term2 in mentioned_terms[i+1:]:
                    if term2 not in relationships[term1]:
                        relationships[term1].append(term2)
                    if term1 not in relationships[term2]:
                        relationships[term2].append(term1)
        
        return dict(relationships)

