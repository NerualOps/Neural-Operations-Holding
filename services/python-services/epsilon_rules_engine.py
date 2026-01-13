"""
Epsilon AI Rules Engine
-----------------
Defines what Epsilon AI can say, what it's meant to do, and rules for behavior.
Ensures Epsilon AI stays on-topic and helpful while following guidelines.
"""

from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules"""
    PURPOSE = "purpose"  # What Epsilon AI is meant to do
    BEHAVIOR = "behavior"  # How Epsilon AI should act
    TOPIC = "topic"  # What topics to focus on
    LANGUAGE = "language"  # How to communicate
    RESTRICTION = "restriction"  # What not to say/do


class EpsilonRulesEngine:
    """Rules engine for Epsilon AI's behavior and communication"""
    
    def __init__(self):
        self.purpose_rules: List[str] = []
        self.behavior_rules: List[str] = []
        self.topic_rules: List[str] = []
        self.language_rules: List[str] = []
        self.restrictions: List[str] = []
        self.allowed_topics: Set[str] = set()
        self.restricted_topics: Set[str] = set()
        self.required_phrases: List[str] = []
        self.forbidden_phrases: List[str] = []
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default rules for Epsilon AI"""
        
        # Purpose rules - what Epsilon AI is meant to do
        self.purpose_rules = [
            "Help businesses with automation, website development, and AI strategy",
            "Provide sales and communication guidance",
            "Assist with business challenges and solutions",
            "Offer professional, helpful advice",
            "Focus on business growth and efficiency"
        ]
        
        # Behavior rules - how Epsilon AI should act
        self.behavior_rules = [
            "Be professional and courteous",
            "Be concise but thorough",
            "Ask clarifying questions when needed",
            "Provide actionable advice",
            "Stay focused on business topics",
            "Be empathetic and understanding",
            "Use clear, simple language"
        ]
        
        # Topic rules - what to focus on
        self.allowed_topics = {
            'business_automation', 'website_development', 'ai_strategy',
            'sales', 'communication', 'customer_relations',
            'business_growth', 'efficiency', 'productivity',
            'technology', 'digital_transformation', 'marketing'
        }
        
        self.topic_rules = [
            "Focus on business automation solutions",
            "Discuss website development and design",
            "Provide AI strategy guidance",
            "Offer sales and communication advice",
            "Help with business challenges"
        ]
        
        # Language rules - how to communicate
        self.language_rules = [
            "Use professional but friendly tone",
            "Avoid jargon unless explaining it",
            "Be clear and direct",
            "Use examples when helpful",
            "Match the user's communication style"
        ]
        
        # Restrictions - what not to say/do
        self.restrictions = [
            "Do not provide medical, legal, or financial advice",
            "Do not make promises about outcomes",
            "Do not share personal information",
            "Do not engage in non-business topics",
            "Do not use offensive language",
            "Do not make guarantees about results"
        ]
        
        self.restricted_topics = {
            'medical', 'legal', 'financial_advice', 'personal_health',
            'politics', 'religion', 'illegal_activities'
        }
        
        self.forbidden_phrases = [
            "I guarantee", "I promise", "100% certain",
            "This will definitely", "You must", "You have to"
        ]
    
    def add_rule(self, rule_type: RuleType, rule: str):
        """Add a custom rule"""
        # Safety check: validate inputs
        if not isinstance(rule_type, RuleType):
            logger.warning(f"Invalid rule_type: {rule_type}")
            return
        if not rule or not isinstance(rule, str):
            logger.warning("Invalid rule: must be a non-empty string")
            return
        if len(rule) > 1000:  # Prevent DoS
            rule = rule[:1000]
            logger.warning("Rule truncated to 1000 characters")
        
        if rule_type == RuleType.PURPOSE:
            self.purpose_rules.append(rule)
        elif rule_type == RuleType.BEHAVIOR:
            self.behavior_rules.append(rule)
        elif rule_type == RuleType.TOPIC:
            self.topic_rules.append(rule)
        elif rule_type == RuleType.LANGUAGE:
            self.language_rules.append(rule)
        elif rule_type == RuleType.RESTRICTION:
            self.restrictions.append(rule)
    
    def validate_response(self, response: str, user_message: str = "") -> Tuple[bool, Optional[str]]:
        """Validate if a response follows the rules"""
        # Safety check: validate inputs
        if not response or not isinstance(response, str):
            return False, "Response must be a non-empty string"
        if len(response) > 10000:  # Prevent DoS
            return False, "Response too long (max 10000 characters)"
        if user_message and not isinstance(user_message, str):
            user_message = ""
        if user_message and len(user_message) > 10000:  # Prevent DoS
            user_message = user_message[:10000]
        
        # Check for forbidden phrases
        response_lower = response.lower()
        for phrase in self.forbidden_phrases:
            if phrase.lower() in response_lower:
                return False, f"Response contains forbidden phrase: {phrase}"
        
        # Check topic relevance
        if user_message:
            user_topics = self._extract_topics(user_message)
            if not any(topic in self.allowed_topics for topic in user_topics):
                # Check if it's a restricted topic
                if any(topic in self.restricted_topics for topic in user_topics):
                    return False, "Topic is restricted"
        
        # Check response length (should be reasonable)
        if len(response) < 10:
            return False, "Response too short"
        if len(response) > 2000:
            return False, "Response too long"
        
        return True, None
    
    def _extract_topics(self, text: str) -> Set[str]:
        """Extract topics from text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return set()
        if len(text) > 10000:  # Prevent DoS
            text = text[:10000]
        
        topics = set()
        text_lower = text.lower()
        
        # Simple keyword matching
        topic_keywords = {
            'business_automation': ['automation', 'automate', 'process', 'workflow'],
            'website_development': ['website', 'web', 'site', 'development', 'design'],
            'ai_strategy': ['ai', 'artificial intelligence', 'machine learning', 'strategy'],
            'sales': ['sales', 'selling', 'customer', 'client', 'prospect'],
            'communication': ['communication', 'talk', 'speak', 'conversation']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.add(topic)
        
        return topics
    
    def get_guidance_for_topic(self, topic: str) -> List[str]:
        """Get guidance rules for a specific topic"""
        # Safety check: validate inputs
        if not topic or not isinstance(topic, str):
            return []
        if len(topic) > 200:  # Prevent DoS
            topic = topic[:200]
        
        guidance = []
        
        if topic in ['business_automation', 'website_development', 'ai_strategy']:
            guidance.extend(self.purpose_rules)
            guidance.extend([
                "Focus on practical solutions",
                "Provide step-by-step guidance when possible",
                "Explain technical concepts clearly"
            ])
        elif topic in ['sales', 'communication']:
            guidance.extend(self.language_rules)
            guidance.extend([
                "Use empathetic language",
                "Focus on customer needs",
                "Provide actionable communication strategies"
            ])
        
        return guidance
    
    def to_dict(self) -> Dict:
        """Serialize rules"""
        return {
            'purpose_rules': self.purpose_rules,
            'behavior_rules': self.behavior_rules,
            'topic_rules': self.topic_rules,
            'language_rules': self.language_rules,
            'restrictions': self.restrictions,
            'allowed_topics': list(self.allowed_topics),
            'restricted_topics': list(self.restricted_topics),
            'forbidden_phrases': self.forbidden_phrases
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EpsilonRulesEngine":
        """Deserialize rules"""
        # Safety check: validate inputs
        if not data or not isinstance(data, dict):
            logger.warning("Invalid data: must be a non-empty dict")
            return cls()
        
        engine = cls()
        # Validate and limit list sizes to prevent DoS
        purpose_rules = data.get('purpose_rules', [])
        if isinstance(purpose_rules, list) and len(purpose_rules) > 100:
            purpose_rules = purpose_rules[:100]
        engine.purpose_rules = purpose_rules if isinstance(purpose_rules, list) else []
        
        behavior_rules = data.get('behavior_rules', [])
        if isinstance(behavior_rules, list) and len(behavior_rules) > 100:
            behavior_rules = behavior_rules[:100]
        engine.behavior_rules = behavior_rules if isinstance(behavior_rules, list) else []
        
        topic_rules = data.get('topic_rules', [])
        if isinstance(topic_rules, list) and len(topic_rules) > 100:
            topic_rules = topic_rules[:100]
        engine.topic_rules = topic_rules if isinstance(topic_rules, list) else []
        
        language_rules = data.get('language_rules', [])
        if isinstance(language_rules, list) and len(language_rules) > 100:
            language_rules = language_rules[:100]
        engine.language_rules = language_rules if isinstance(language_rules, list) else []
        
        restrictions = data.get('restrictions', [])
        if isinstance(restrictions, list) and len(restrictions) > 100:
            restrictions = restrictions[:100]
        engine.restrictions = restrictions if isinstance(restrictions, list) else []
        
        allowed_topics = data.get('allowed_topics', [])
        if isinstance(allowed_topics, list) and len(allowed_topics) > 200:
            allowed_topics = allowed_topics[:200]
        engine.allowed_topics = set(allowed_topics) if isinstance(allowed_topics, list) else set()
        
        restricted_topics = data.get('restricted_topics', [])
        if isinstance(restricted_topics, list) and len(restricted_topics) > 200:
            restricted_topics = restricted_topics[:200]
        engine.restricted_topics = set(restricted_topics) if isinstance(restricted_topics, list) else set()
        
        forbidden_phrases = data.get('forbidden_phrases', [])
        if isinstance(forbidden_phrases, list) and len(forbidden_phrases) > 100:
            forbidden_phrases = forbidden_phrases[:100]
        engine.forbidden_phrases = forbidden_phrases if isinstance(forbidden_phrases, list) else []
        
        return engine

