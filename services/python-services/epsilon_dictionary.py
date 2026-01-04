"""
Epsilon AI Dictionary System
----------------------
Word-level understanding system that allows Epsilon AI to understand individual words
and their meanings, enabling better language comprehension and generation.
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EpsilonDictionary:
    """Dictionary system for word-level understanding"""
    
    def __init__(self):
        self.words: Dict[str, Dict] = {}  # word -> {definition, synonyms, context, usage}
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.business_terms: Set[str] = set()
        self.sales_terms: Set[str] = set()
        self.general_terms: Set[str] = set()
        self.word_relationships: Dict[str, List[str]] = defaultdict(list)  # word -> related words
        
    def add_word(self, word: str, definition: str, synonyms: List[str] = None, 
                 context: str = "general", usage_examples: List[str] = None,
                 related_words: List[str] = None, source: str = None):
        """Add a word to the dictionary with full context
        
        Args:
            word: The word to add
            definition: The definition of the word
            synonyms: List of synonyms
            context: Context category ('business', 'sales', 'general')
            usage_examples: List of usage examples
            related_words: List of related words
            source: Source document ID or name (for tracking)
        """
        # Safety check: validate inputs
        if not word or not isinstance(word, str):
            logger.warning("Invalid word input, skipping")
            return
        if len(word) > 200:  # Prevent DoS
            logger.warning(f"Word too long ({len(word)} chars), truncating to 200")
            word = word[:200]
        if not definition or not isinstance(definition, str):
            logger.warning("Invalid definition input, skipping")
            return
        if len(definition) > 10000:  # Prevent DoS
            logger.warning(f"Definition too long ({len(definition)} chars), truncating to 10000")
            definition = definition[:10000]
        if synonyms is not None and not isinstance(synonyms, list):
            synonyms = None
        if synonyms and len(synonyms) > 100:  # Prevent DoS
            logger.warning(f"Too many synonyms ({len(synonyms)}), limiting to 100")
            synonyms = synonyms[:100]
        if synonyms:
            synonyms = [s for s in synonyms if isinstance(s, str) and len(s) <= 200][:100]
        if usage_examples is not None and not isinstance(usage_examples, list):
            usage_examples = None
        if usage_examples and len(usage_examples) > 50:  # Prevent DoS
            logger.warning(f"Too many usage examples ({len(usage_examples)}), limiting to 50")
            usage_examples = usage_examples[:50]
        if usage_examples:
            usage_examples = [e for e in usage_examples if isinstance(e, str) and len(e) <= 500][:50]
        if related_words is not None and not isinstance(related_words, list):
            related_words = None
        if related_words and len(related_words) > 100:  # Prevent DoS
            logger.warning(f"Too many related words ({len(related_words)}), limiting to 100")
            related_words = related_words[:100]
        if related_words:
            related_words = [r for r in related_words if isinstance(r, str) and len(r) <= 200][:100]
        if context and not isinstance(context, str):
            context = "general"
        if context not in ['business', 'sales', 'general']:
            context = "general"
        if source and not isinstance(source, str):
            source = None
        if source and len(source) > 500:  # Prevent DoS
            source = source[:500]
        
        word_lower = word.lower().strip()
        
        # Check if word already exists
        if word_lower in self.words:
            existing = self.words[word_lower]
            # If definition is different, warn but allow update (latest wins)
            if existing['definition'] != definition:
                logger.warning(f"Warning: Word '{word_lower}' already exists with different definition. "
                            f"Old: '{existing['definition'][:50]}...' "
                            f"New: '{definition[:50]}...' "
                            f"Updating to new definition.")
            # Merge synonyms and related words if they exist
            merged_synonyms = list(set((synonyms or []) + existing.get('synonyms', [])))
            merged_related = list(set((related_words or []) + existing.get('related_words', [])))
            merged_examples = list(set((usage_examples or []) + existing.get('usage_examples', [])))
        else:
            merged_synonyms = synonyms or []
            merged_related = related_words or []
            merged_examples = usage_examples or []
        
        self.words[word_lower] = {
            'definition': definition,
            'synonyms': merged_synonyms,
            'context': context,  # 'business', 'sales', 'general'
            'usage_examples': merged_examples,
            'related_words': merged_related,
            'frequency': self.word_frequencies.get(word_lower, 0),
            'source': source  # Track where this definition came from
        }
        
        # Categorize word
        if context == 'business':
            self.business_terms.add(word_lower)
        elif context == 'sales':
            self.sales_terms.add(word_lower)
        else:
            self.general_terms.add(word_lower)
        
        # Build relationships
        if related_words:
            for related in related_words:
                self.word_relationships[word_lower].append(related.lower())
                if related.lower() in self.words:
                    self.word_relationships[related.lower()].append(word_lower)
    
    def learn_from_text(self, text: str, context: str = "general"):
        """Extract words from text and learn their meanings from context"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input, skipping")
            return
        if len(text) > 1000000:  # Prevent DoS - 1MB max
            logger.warning(f"Text too long ({len(text)} chars), truncating to 1MB")
            text = text[:1000000]
        if context and not isinstance(context, str):
            context = "general"
        if context not in ['business', 'sales', 'general']:
            context = "general"
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        for word in words:
            self.word_frequencies[word] += 1
            
            # If word not in dictionary, add basic entry
            if word not in self.words:
                # Try to infer meaning from context
                definition = self._infer_meaning_from_context(word, text, context)
                self.add_word(word, definition, context=context)
    
    def _infer_meaning_from_context(self, word: str, text: str, context: str) -> str:
        """Infer word meaning from surrounding context"""
        # Safety check: validate inputs
        if not word or not isinstance(word, str) or len(word) > 200:
            return f"A {context} term"
        if not text or not isinstance(text, str):
            return f"A {context} term"
        if len(text) > 1000000:  # Prevent DoS
            text = text[:1000000]
        
        # Find sentences containing the word
        sentences = re.split(r'[.!?]+', text)
        word_sentences = [s for s in sentences if word.lower() in s.lower()]
        
        if word_sentences:
            # Use first sentence as context
            context_sentence = word_sentences[0]
            return f"Used in context: {context_sentence.strip()[:100]}"
        
        return f"A {context} term"
    
    def get_word_meaning(self, word: str) -> Optional[Dict]:
        """Get full meaning and context for a word"""
        # Safety check: validate inputs
        if not word or not isinstance(word, str):
            return None
        if len(word) > 200:  # Prevent DoS
            return None
        
        return self.words.get(word.lower())
    
    def understand_sentence(self, sentence: str) -> Dict[str, Dict]:
        """Understand each word in a sentence"""
        # Safety check: validate inputs
        if not sentence or not isinstance(sentence, str):
            return {}
        if len(sentence) > 100000:  # Prevent DoS - 100KB max
            logger.warning(f"Sentence too long ({len(sentence)} chars), truncating to 100KB")
            sentence = sentence[:100000]
        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        understanding = {}
        
        for word in words:
            meaning = self.get_word_meaning(word)
            if meaning:
                understanding[word] = meaning
            else:
                # Unknown word - mark for learning
                understanding[word] = {
                    'definition': 'Unknown - needs learning',
                    'context': 'general',
                    'needs_learning': True
                }
        
        return understanding
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        # Safety check: validate inputs
        if not word or not isinstance(word, str):
            return []
        if len(word) > 200:  # Prevent DoS
            return []
        
        word_data = self.words.get(word.lower())
        if word_data:
            return word_data.get('synonyms', [])
        return []
    
    def get_related_words(self, word: str) -> List[str]:
        """Get related words"""
        # Safety check: validate inputs
        if not word or not isinstance(word, str):
            return []
        if len(word) > 200:  # Prevent DoS
            return []
        
        return self.word_relationships.get(word.lower(), [])
    
    def to_dict(self) -> Dict:
        """Serialize dictionary"""
        return {
            'words': self.words,
            'word_frequencies': dict(self.word_frequencies),
            'business_terms': list(self.business_terms),
            'sales_terms': list(self.sales_terms),
            'general_terms': list(self.general_terms),
            'word_relationships': dict(self.word_relationships)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EpsilonDictionary":
        """Deserialize dictionary"""
        dictionary = cls()
        dictionary.words = data.get('words', {})
        dictionary.word_frequencies = defaultdict(int, data.get('word_frequencies', {}))
        dictionary.business_terms = set(data.get('business_terms', []))
        dictionary.sales_terms = set(data.get('sales_terms', []))
        dictionary.general_terms = set(data.get('general_terms', []))
        dictionary.word_relationships = defaultdict(list, data.get('word_relationships', {}))
        return dictionary
    
    def load_from_dictionary_file(self, content: str, context: str = "general", source: str = None) -> int:
        """Load words and definitions from a dictionary file (JSON format)
        
        Expected formats:
        1. JSON object: {"word": "definition", "word2": "definition2", ...}
        2. JSON array: [{"word": "word1", "definition": "def1"}, ...]
        3. JSON object with nested structure: {"words": {"word1": "def1", ...}}
        4. Plain text: "word: definition\nword2: definition2"
        """
        # Safety check: validate inputs
        if not content or not isinstance(content, str):
            logger.warning("Invalid content input, returning 0")
            return 0
        if len(content) > 10000000:  # Prevent DoS - 10MB max
            logger.warning(f"Content too long ({len(content)} chars), truncating to 10MB")
            content = content[:10000000]
        if context and not isinstance(context, str):
            context = "general"
        if context not in ['business', 'sales', 'general']:
            context = "general"
        if source and not isinstance(source, str):
            source = None
        if source and len(source) > 500:  # Prevent DoS
            source = source[:500]
        
        words_loaded = 0
        
        try:
            # Try JSON format first
            try:
                data = json.loads(content)
                
                # Format 1: Simple object {"word": "definition"}
                if isinstance(data, dict):
                    # Check if it's format 3 (nested)
                    if "words" in data and isinstance(data["words"], dict):
                        words_dict = data["words"]
                    else:
                        words_dict = data
                    
                    for word, definition in words_dict.items():
                        if isinstance(definition, str) and len(definition.strip()) > 0:
                            word_lower = word.lower().strip()
                            if word_lower and len(word_lower) >= 2:
                                self.add_word(word_lower, definition.strip(), context=context, source=source)
                                words_loaded += 1
                        elif isinstance(definition, dict):
                            # Format with additional fields
                            word_lower = word.lower().strip()
                            if word_lower and len(word_lower) >= 2:
                                def_text = definition.get('definition', definition.get('meaning', ''))
                                synonyms = definition.get('synonyms', [])
                                related = definition.get('related_words', definition.get('related', []))
                                examples = definition.get('usage_examples', definition.get('examples', []))
                                if def_text:
                                    self.add_word(word_lower, def_text, synonyms=synonyms, 
                                                 related_words=related, usage_examples=examples, context=context, source=source)
                                    words_loaded += 1
                
                # Format 2: Array of objects
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            word = item.get('word', item.get('term', ''))
                            definition = item.get('definition', item.get('meaning', item.get('def', '')))
                            if word and definition:
                                word_lower = word.lower().strip()
                                if word_lower and len(word_lower) >= 2:
                                    synonyms = item.get('synonyms', [])
                                    related = item.get('related_words', item.get('related', []))
                                    examples = item.get('usage_examples', item.get('examples', []))
                                    self.add_word(word_lower, definition, synonyms=synonyms,
                                                 related_words=related, usage_examples=examples, context=context, source=source)
                                    words_loaded += 1
                
            except json.JSONDecodeError:
                # Not JSON, try plain text format: "word: definition"
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try "word: definition" format
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            word = parts[0].strip()
                            definition = parts[1].strip()
                            if word and definition:
                                word_lower = word.lower().strip()
                                if word_lower and len(word_lower) >= 2:
                                    self.add_word(word_lower, definition, context=context, source=source)
                                    words_loaded += 1
                    
                    # Try "word - definition" format
                    elif ' - ' in line:
                        parts = line.split(' - ', 1)
                        if len(parts) == 2:
                            word = parts[0].strip()
                            definition = parts[1].strip()
                            if word and definition:
                                word_lower = word.lower().strip()
                                if word_lower and len(word_lower) >= 2:
                                    self.add_word(word_lower, definition, context=context, source=source)
                                    words_loaded += 1
        
        except Exception as e:
            logger.error(f"Error loading dictionary file: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return words_loaded
    
    def build_from_documents(self, documents: List[Dict[str, str]], context_map: Dict[str, str] = None):
        """Build dictionary from uploaded documents"""
        # Safety check: validate inputs
        if not documents or not isinstance(documents, list):
            logger.warning("Invalid documents input, skipping")
            return
        if len(documents) > 10000:  # Prevent DoS
            logger.warning(f"Too many documents ({len(documents)}), limiting to 10000")
            documents = documents[:10000]
        if context_map is not None and not isinstance(context_map, dict):
            context_map = {}
        if context_map and len(context_map) > 10000:  # Prevent DoS
            logger.warning(f"Context map too large ({len(context_map)}), limiting to 10000")
            context_map = dict(list(context_map.items())[:10000])
        
        context_map = context_map or {}
        
        for doc in documents:
            content = doc.get('content', '')
            doc_id = doc.get('id', '')
            doc_type = doc.get('document_type', '')
            category = context_map.get(doc_id, doc.get('learning_category', 'general'))
            
            # Check if this is a dictionary document
            if doc_type == 'dictionary':
                # Load from dictionary file format
                words_loaded = self.load_from_dictionary_file(content, context='general', source=doc_id)
                logger.info(f"[DICTIONARY] Loaded {words_loaded} words from dictionary document {doc_id}")
                continue
            
            # Map category to context
            if category in ['sales_training', 'communication_guide']:
                context = 'sales'
            elif category == 'knowledge':
                context = 'business'
            else:
                context = 'general'
            
            # Learn words from document
            self.learn_from_text(content, context)
            
            # Extract key terms (capitalized words, technical terms)
            key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            for term in key_terms:
                term_lower = term.lower()
                if term_lower not in self.words and len(term_lower) > 3:
                    # Extract definition from surrounding context
                    pattern = rf'\b{re.escape(term)}\b[^.]*\.'
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        definition = matches[0][:200]
                        self.add_word(term_lower, definition, context=context)

