#!/usr/bin/env python3
"""
Epsilon AI - Content Generation & Enhancement Service
© 2025 Neural Operation's & Holding's LLC. All rights reserved.

This service provides advanced content generation and enhancement for Epsilon AI including:
- Dynamic response generation
- Content personalization
- Text summarization
- Content optimization
- Multi-language support
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentRequest:
    """Content generation request"""
    content_type: str
    topic: str
    user_context: Dict[str, Any]
    requirements: Dict[str, Any]
    style: str = "professional"

@dataclass
class GeneratedContent:
    """Generated content response"""
    content: str
    metadata: Dict[str, Any]
    quality_score: float
    personalization_level: float
    suggestions: List[str]

class EpsilonContentGenerator:
    """Advanced content generation for Epsilon AI"""
    
    def __init__(self):
        self.session = None
        self.templates = {}
        self.knowledge_base = {}
        self.user_preferences = {}
        
    async def initialize(self):
        """Initialize the content generator"""
        try:
            # self.session = aiohttp.ClientSession()  # Not needed - using FastAPI
            await self._load_templates()
            await self._load_knowledge_base()
            logger.info("[CONTENT] Epsilon AI Content Generator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[CONTENT] Failed to initialize Content Generator: {e}")
            return False
    
    async def _load_templates(self):
        """Load content templates"""
        self.templates = {
            "greeting": [
                "Hello! I'm Epsilon AI, your AI automation specialist. How can I help transform your business today?",
                "Hi there! I'm Epsilon AI, and I specialize in business automation and website development. What can I help you with?",
                "Welcome! I'm Epsilon AI, your AI assistant for business solutions. How can I assist you today?"
            ],
            "business_inquiry": [
                "I'd love to help with your business needs! We specialize in website development, business automation, and AI strategy. What specific challenge can I help you solve?",
                "Great question! Our expertise covers business automation, website development, and AI implementation. What's your main business goal?",
                "Excellent! I can help with business automation, website development, and AI strategy. What specific area interests you most?"
            ],
            "website_development": [
                "Website development is one of our core specialties! We create modern, responsive websites that drive business growth. What type of website are you looking to build?",
                "I'd be happy to help with your website development needs! We build custom websites that are optimized for performance and user experience. Tell me about your project.",
                "Website development is our passion! We create websites that not only look great but also convert visitors into customers. What's your vision for your website?"
            ],
            "business_automation": [
                "Business automation can transform your operations! We help businesses streamline processes, reduce costs, and improve efficiency. What processes would you like to automate?",
                "Automation is the future of business! We specialize in creating automated workflows that save time and money. What manual tasks are taking up too much of your time?",
                "Let's automate your business processes! We design custom automation solutions that work seamlessly with your existing systems. What's your biggest operational challenge?"
            ],
            "ai_strategy": [
                "AI strategy is crucial for modern businesses! We help companies implement AI solutions that drive real results. What AI opportunities are you exploring?",
                "AI can revolutionize your business! We develop AI strategies that align with your goals and deliver measurable value. What's your current AI experience?",
                "Let's develop your AI strategy! We create comprehensive AI roadmaps that position your business for success. What AI challenges are you facing?"
            ],
            "pricing": [
                "I'd be happy to discuss pricing! Our solutions are designed to provide excellent value and ROI. What type of project are you considering?",
                "Let's talk about investment! We offer competitive pricing with flexible options to fit your budget. What's your project scope?",
                "Pricing depends on your specific needs! We provide transparent, value-based pricing that delivers results. What are you looking to achieve?"
            ],
            "technical_support": [
                "I'm here to help with any technical issues! Can you describe the problem you're experiencing in more detail?",
                "Let's troubleshoot this together! I'll help you resolve the technical challenge you're facing. What exactly is happening?",
                "Technical support is my specialty! I'll work with you to find a solution. Can you provide more details about the issue?"
            ],
            "closing": [
                "Thank you for chatting with me today! Feel free to reach out anytime you need assistance. Have a great day!",
                "It was great helping you today! Don't hesitate to contact us if you have any more questions. Take care!",
                "Thanks for the conversation! We're here whenever you need us. Have a wonderful day!"
            ]
        }
        logger.info("[CONTENT] Content templates loaded")
    
    async def _search_documents(self, query: str, limit: int = 3, document_type: Optional[str] = None, learning_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search uploaded documents for relevant knowledge, optionally filtered by type/category"""
        if not query or not isinstance(query, str):
            return []
        if len(query) > 1000:
            query = query[:1000]
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            limit = 3
        if document_type is not None and (not isinstance(document_type, str) or len(document_type) > 100):
            document_type = None
        if learning_category is not None and (not isinstance(learning_category, str) or len(learning_category) > 100):
            learning_category = None
        
        try:
            import aiohttp
            import os
            
            supabase_url = os.getenv('SUPABASE_URL')
            if not supabase_url:
                raise ValueError('SUPABASE_URL environment variable is required')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY', '')
            
            if not supabase_key:
                logger.warning("[CONTENT] Supabase key not set, cannot search documents")
                return []
            
            # Build query with optional filters
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': supabase_key,
                    'Authorization': f'Bearer {supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                # Build Supabase query with filters - include chunked flag
                url = f"{supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,content,learning_category,doc_type,document_type,is_chunked,total_chunks',
                    'limit': '100'
                }
                
                # Add document type filter if specified
                if document_type:
                    params['document_type'] = f'eq.{document_type}'
                    logger.info(f"🔍 Filtering documents by type: {document_type}")
                
                # Add learning category filter if specified
                if learning_category:
                    params['learning_category'] = f'eq.{learning_category}'
                    logger.info(f"🔍 Filtering documents by category: {learning_category}")
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        all_documents = await response.json()
                        
                        # Simple keyword matching with relevance scoring
                        query_words = [w for w in query.lower().split() if len(w) > 2]
                        scored_docs = []
                        
                        for doc in all_documents:
                            # Handle chunked documents - fetch full content from chunks if needed
                            content = doc.get('content', '').lower()
                            is_chunked = doc.get('is_chunked', False)
                            doc_id = doc.get('id')
                            
                            if is_chunked and doc_id:
                                try:
                                    doc_id_str = str(doc_id).strip()
                                    if not doc_id_str or doc_id_str.lower() in ('none', 'null', 'undefined'):
                                        logger.warning(f"[CONTENT] Invalid document_id: {doc_id}, skipping chunks")
                                    else:
                                        chunks_url = f"{supabase_url}/rest/v1/doc_chunks"
                                        chunks_params = {
                                            'select': 'chunk_text,chunk_index',
                                            'document_id': f'eq.{doc_id_str}',
                                            'order': 'chunk_index.asc',
                                            'limit': '1000'
                                        }
                                        try:
                                            timeout = aiohttp.ClientTimeout(total=30)
                                        except (AttributeError, TypeError):
                                            timeout = 30
                                        async with session.get(chunks_url, headers=headers, params=chunks_params, timeout=timeout) as chunks_response:
                                            if chunks_response.status == 200:
                                                chunks = await chunks_response.json()
                                                if chunks:
                                                    chunks_sorted = sorted(chunks, key=lambda c: c.get('chunk_index', 0))
                                                    full_content = '\n\n'.join(chunk.get('chunk_text', '') for chunk in chunks_sorted)
                                                    content = full_content.lower()
                                                    logger.info(f"[CONTENT] Reconstructed chunked document {doc_id_str} from {len(chunks)} chunks")
                                            elif chunks_response.status in (521, 522, 503):
                                                logger.warning(f"[CONTENT] Supabase connection issue (status {chunks_response.status}) while fetching chunks for document {doc_id_str}")
                                            else:
                                                logger.warning(f"[CONTENT] Failed to fetch chunks for document {doc_id_str}: HTTP {chunks_response.status}")
                                except asyncio.TimeoutError:
                                    logger.warning(f"[CONTENT] Timeout fetching chunks for document {doc_id}")
                                    # Continue with preview content
                                except Exception as chunk_error:
                                    logger.warning(f"[CONTENT] Failed to fetch chunks for document {doc_id}: {chunk_error}")
                                    # Continue with preview content
                            
                            title = doc.get('title', '').lower()
                            doc_type = doc.get('doc_type', '') or doc.get('document_type', '')
                            doc_category = doc.get('learning_category', '')
                            
                            # Calculate relevance score
                            score = 0
                            title_matches = sum(1 for word in query_words if word in title)
                            content_matches = sum(1 for word in query_words if word in content)
                            
                            # Prioritize title matches
                            score += title_matches * 3
                            score += content_matches
                            
                            # Boost score based on document type relevance
                            if doc_category == 'sales_training' and any(word in ['sell', 'sales', 'customer', 'client', 'pitch', 'conversion'] for word in query_words):
                                score += 2
                            elif doc_category == 'knowledge' and any(word in ['what', 'how', 'explain', 'tell', 'information'] for word in query_words):
                                score += 2
                            elif doc_type == 'case_study' and any(word in ['example', 'case', 'success', 'client'] for word in query_words):
                                score += 2
                            
                            if score > 0:
                                scored_docs.append({
                                    'doc': doc,
                                    'score': score,
                                    'type': doc_type,
                                    'category': doc_category
                                })
                        
                        scored_docs.sort(key=lambda x: x['score'], reverse=True)
                        relevant_docs = [item['doc'] for item in scored_docs[:limit]]
                        
                        logger.info(f"[CONTENT] Found {len(relevant_docs)} relevant documents (filtered by type/category: {document_type}/{learning_category})")
                        if relevant_docs:
                            logger.info(f"[CONTENT] Document types found: {[doc.get('doc_type') or doc.get('document_type') for doc in relevant_docs]}")
                            logger.info(f"[CONTENT] Document categories found: {[doc.get('learning_category') for doc in relevant_docs]}")
                        return relevant_docs
            
            return []
        except Exception as e:
            logger.error(f"[CONTENT] Error searching documents: {e}")
            return []
    
    async def _load_knowledge_base(self):
        """Load knowledge base for content generation"""
        self.knowledge_base = {
            "business_automation": {
                "benefits": [
                    "Reduces manual work by up to 80%",
                    "Improves accuracy and consistency",
                    "Saves time and reduces costs",
                    "Enhances customer experience",
                    "Scales with business growth"
                ],
                "examples": [
                    "Email marketing automation",
                    "Customer onboarding workflows",
                    "Invoice processing automation",
                    "Lead qualification systems",
                    "Inventory management automation"
                ],
                "technologies": [
                    "Zapier integration",
                    "Custom API development",
                    "Database automation",
                    "Workflow orchestration",
                    "AI-powered decision making"
                ]
            },
            "website_development": {
                "benefits": [
                    "Professional online presence",
                    "Mobile-responsive design",
                    "SEO optimization",
                    "Fast loading speeds",
                    "User-friendly interface"
                ],
                "examples": [
                    "Business websites",
                    "E-commerce platforms",
                    "Portfolio sites",
                    "Landing pages",
                    "Web applications"
                ],
                "technologies": [
                    "React and Next.js",
                    "WordPress customization",
                    "E-commerce solutions",
                    "Database integration",
                    "Cloud hosting"
                ]
            },
            "ai_strategy": {
                "benefits": [
                    "Data-driven decision making",
                    "Predictive analytics",
                    "Process optimization",
                    "Customer insights",
                    "Competitive advantage"
                ],
                "examples": [
                    "Chatbot implementation",
                    "Predictive maintenance",
                    "Customer behavior analysis",
                    "Automated reporting",
                    "Intelligent recommendations"
                ],
                "technologies": [
                    "Machine learning models",
                    "Natural language processing",
                    "Computer vision",
                    "Predictive analytics",
                    "AI integration platforms"
                ]
            }
        }
        logger.info("[CONTENT] Knowledge base loaded")
    
    async def generate_response(self, user_message: str, context: Dict[str, Any] = None) -> GeneratedContent:
        """
        Advanced multi-step reasoning for response generation:
        1. Recall - Get relevant memories and patterns
        2. Reason - Plan optimal response approach  
        3. Act - Generate personalized content
        4. Evaluate - Check if response meets quality bar
        """
        if not user_message or not isinstance(user_message, str):
            raise ValueError("user_message must be a non-empty string")
        if len(user_message) > 10000:
            raise ValueError("user_message too long (max 10000 characters)")
        if context is not None and not isinstance(context, dict):
            raise ValueError("context must be a dictionary or None")
        
        try:
            logger.info(f"[CONTENT] Generating response for: {user_message[:50]}...")
            
            if context is None:
                context = {}
            
            # STEP 1: RECALL - Gather relevant knowledge
            logger.info("[REASONING] STEP 1: Recalling relevant knowledge...")
            intent = await self._analyze_intent(user_message)
            topic = await self._identify_topic(user_message)
            user_profile = context.get('user_profile', {})
            
            # Get learned patterns from past successful interactions
            learned_patterns = await self._get_learned_patterns(user_message)
            logger.info(f"[REASONING] Loaded {len(learned_patterns)} learned patterns")
            
            # Search uploaded documents for relevant knowledge
            # Intelligently determine if we should filter by document type based on query
            document_type_filter = None
            learning_category_filter = None
            
            # Detect if query is about sales/communication (use sales_training docs)
            if any(word in user_message.lower() for word in ['sell', 'sales', 'customer', 'client', 'pitch', 'tone', 'communicate', 'approach']):
                learning_category_filter = 'sales_training'
                logger.info("[CONTENT] Detected sales/communication query - filtering for sales_training documents")
            # Detect if query is about case studies/examples
            elif any(word in user_message.lower() for word in ['example', 'case study', 'success story', 'client story']):
                document_type_filter = 'case_study'
                logger.info("[CONTENT] Detected case study query - filtering for case_study documents")
            # Default to knowledge base documents for general questions
            elif any(word in user_message.lower() for word in ['what', 'how', 'explain', 'tell me', 'information']):
                learning_category_filter = 'knowledge'
                logger.info("[CONTENT] Detected knowledge query - filtering for knowledge documents")
            
            relevant_docs = await self._search_documents(
                user_message, 
                limit=5,
                document_type=document_type_filter,
                learning_category=learning_category_filter
            )
            logger.info(f"[CONTENT] Found {len(relevant_docs)} relevant documents from knowledge base")
            
            # STEP 2: REASON - Plan response strategy
            logger.info("[REASONING] STEP 2: Planning response strategy...")
            response_strategy = self._plan_response_strategy(intent, topic, learned_patterns, relevant_docs)
            logger.info(f"[CONTENT] Response strategy: {response_strategy}")
            
            # STEP 3: ACT - Generate content using strategy
            logger.info("[REASONING] STEP 3: Generating content...")
            base_content = await self._generate_base_content(intent, topic, relevant_docs, learned_patterns)
            personalized_content = await self._personalize_content(base_content, user_profile, context, topic)
            enhanced_content = await self._enhance_content(personalized_content, topic, context, relevant_docs)
            
            # STEP 4: EVALUATE - Check response quality
            logger.info("[REASONING] STEP 4: Evaluating response quality...")
            quality_score = await self._calculate_quality_score(enhanced_content, intent, topic)
            personalization_level = await self._calculate_personalization_level(enhanced_content, user_profile)
            evaluation = self._evaluate_response_quality(enhanced_content, quality_score, personalization_level)
            
            if evaluation['needs_improvement']:
                logger.info("[REASONING] Quality below threshold, refining response...")
                enhanced_content = self._refine_response(enhanced_content, evaluation['improvement_areas'])
            
            # Generate suggestions for future improvements
            suggestions = await self._generate_suggestions(enhanced_content, topic, context)
            
            content = GeneratedContent(
                content=enhanced_content,
                metadata={
                    "intent": intent,
                    "topic": topic,
                    "generation_time": datetime.now().isoformat(),
                    "template_used": intent,
                    "personalization_applied": True
                },
                quality_score=quality_score,
                personalization_level=personalization_level,
                suggestions=suggestions
            )
            
            logger.info(f"[CONTENT] Content generated: quality={quality_score:.2f}, personalization={personalization_level:.2f}")
            return content
            
        except Exception as e:
            logger.error(f"[ERROR] Error generating content: {e}")
            # NO FALLBACKS - Python LLM service is required
            raise Exception(f"Content generation failed: {str(e)}. Epsilon AI requires trained models.")
    
    async def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        intent_patterns = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "business_inquiry": ["business", "company", "services", "help", "need"],
            "website_development": ["website", "web", "development", "design", "site"],
            "business_automation": ["automation", "workflow", "process", "efficiency"],
            "ai_strategy": ["ai", "artificial intelligence", "machine learning", "strategy"],
            "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable"],
            "technical_support": ["support", "help", "technical", "bug", "issue", "problem"],
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "goodbye": ["bye", "goodbye", "see you", "farewell", "thanks"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return intent
        
        return "general"
    
    async def _identify_topic(self, message: str) -> str:
        """Identify main topic from message"""
        message_lower = message.lower()
        
        topic_keywords = {
            "business_automation": ["automation", "workflow", "process", "efficiency", "streamline"],
            "website_development": ["website", "web", "development", "design", "site", "online"],
            "ai_strategy": ["ai", "artificial intelligence", "machine learning", "strategy", "intelligence"],
            "technical_support": ["support", "help", "technical", "bug", "issue", "problem", "fix"],
            "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable", "investment"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        
        return "general"
    
    async def _get_learned_patterns(self, query: str) -> List[Dict]:
        """Get learned patterns from successful interactions"""
        if not query or not isinstance(query, str):
            return []
        if len(query) > 1000:
            query = query[:1000]
        
        try:
            import aiohttp
            import os
            
            supabase_url = os.getenv('SUPABASE_URL')
            if not supabase_url:
                raise ValueError('SUPABASE_URL environment variable is required')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY', '')
            
            if not supabase_key:
                return []
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': supabase_key,
                    'Authorization': f'Bearer {supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{supabase_url}/rest/v1/epsilon_learning_patterns"
                params = {
                    'select': '*',
                    'is_active': 'eq.true',
                    'order': 'confidence_score.desc',
                    'limit': '5'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        patterns = await response.json()
                        return patterns
                    
            return []
        except Exception as e:
            logger.error(f"[CONTENT] Error getting learned patterns: {e}")
            return []
    
    async def _get_model_weights(self) -> Dict[str, float]:
        """Get current model weights to adjust responses (models now stored on Podrun, not Supabase)"""
        try:
            # Models are now stored on Podrun, not Supabase
            # Return empty dict since weights are managed on Podrun
            logger.debug("[CONTENT] Model weights are managed on Podrun, not retrieved from Supabase")
            return {}
        except Exception as e:
            logger.error(f"[CONTENT] Error getting model weights: {e}")
            return {}
    
    async def _generate_base_content(self, intent: str, topic: str, relevant_docs: List[Dict] = None, learned_patterns: List[Dict] = None) -> str:
        """Generate base content from templates and learned patterns"""
        try:
            if learned_patterns:
                logger.info(f"🎓 Using {len(learned_patterns)} learned patterns")
                
                # Get patterns with highest confidence and success
                successful_patterns = [
                    p for p in learned_patterns 
                    if p.get('pattern_type') in ['success_factor', 'user_satisfaction']
                    and p.get('confidence_score', 0) > 0.7
                ]
                
                if successful_patterns:
                    best_pattern = successful_patterns[0]
                    pattern_data = best_pattern.get('pattern_data', {})
                    
                    if isinstance(pattern_data, str):
                        import json
                        pattern_data = json.loads(pattern_data)
                    
                    # Use learned response approach
                    logger.info("[CONTENT] Applying learned pattern for better response")
            
            # Get template for intent
            if intent in self.templates:
                templates = self.templates[intent]
                base_content = random.choice(templates)
            else:
                if topic in self.templates:
                    templates = self.templates[topic]
                    base_content = random.choice(templates)
                else:
                    base_content = "I'm here to help! How can I assist you with your business needs today?"
            
            # Get model weights to adjust response style
            weights = await self._get_model_weights()
            if weights:
                # Adjust based on learned weights
                if weights.get('response_quality', 0.5) > 0.7:
                    # High quality learned responses - be more confident
                    base_content = base_content.replace("I can", "I definitely can")
                    base_content = base_content.replace("we help", "we excel at helping")
                elif weights.get('response_quality', 0.5) < 0.3:
                    # Low quality - be more conservative
                    base_content = base_content.replace("I'm", "I'm here to")
                    base_content = base_content.replace("!", ".")
            
            return base_content
            
        except Exception as e:
            logger.error(f"[CONTENT] Error generating base content: {e}")
            return "I'm here to help! How can I assist you today?"
    
    async def _personalize_content(self, content: str, user_profile: Dict[str, Any], context: Dict[str, Any], topic: str = "general") -> str:
        """Personalize content based on user profile"""
        try:
            personalized_content = content
            
            # Add user name if available
            user_name = user_profile.get('name')
            if user_name and 'Hello' in personalized_content:
                personalized_content = personalized_content.replace('Hello!', f'Hello {user_name}!')
            
            # Add industry-specific examples
            industry = user_profile.get('industry')
            if industry and topic in self.knowledge_base:
                # Add industry-specific benefit
                benefits = self.knowledge_base[topic].get('benefits', [])
                if benefits:
                    benefit = random.choice(benefits)
                    personalized_content += f" For {industry} businesses, this means {benefit.lower()}."
            
            # Add company size considerations
            company_size = user_profile.get('company_size')
            if company_size:
                if company_size == 'small':
                    personalized_content += " We understand the unique challenges small businesses face and offer solutions that scale with your growth."
                elif company_size == 'enterprise':
                    personalized_content += " We have extensive experience with enterprise-level implementations and can handle complex, large-scale projects."
            
            return personalized_content
            
        except Exception as e:
            logger.error(f"[CONTENT] Error personalizing content: {e}")
            return content
    
    async def _enhance_content(self, content: str, topic: str, context: Dict[str, Any], relevant_docs: List[Dict] = None) -> str:
        """Enhance content with additional value"""
        try:
            enhanced_content = content
            
            # Add knowledge from uploaded documents
            if relevant_docs:
                logger.info(f"[CONTENT] Enhancing with {len(relevant_docs)} documents")
                for doc in relevant_docs[:2]:  # Use top 2 docs
                    doc_content = doc.get('content', '')
                    doc_title = doc.get('title', '')
                    
                    # Extract a relevant snippet from the document and use naturally (never quote)
                    if doc_content and len(doc_content) > 50:
                        # Find a sentence that might be relevant
                        sentences = doc_content.split('.')
                        if sentences:
                            relevant_snippet = sentences[0][:150].strip()
                            # Remove document references and quoting language
                            relevant_snippet = relevant_snippet.replace('According to', '').replace('Based on', '')
                            relevant_snippet = relevant_snippet.replace('the document', '').replace('our knowledge base', '')
                            relevant_snippet = relevant_snippet.strip(' ,:;-')
                            if relevant_snippet:
                                # Use naturally in first person, not as a quote
                                if not relevant_snippet.lower().startswith('i ') and not relevant_snippet.lower().startswith('we '):
                                    relevant_snippet = 'I ' + relevant_snippet[0].lower() + relevant_snippet[1:] if relevant_snippet else relevant_snippet
                                enhanced_content += f" {relevant_snippet}."
            
            # Add relevant examples if topic is in knowledge base
            if topic in self.knowledge_base:
                examples = self.knowledge_base[topic].get('examples', [])
                if examples and len(enhanced_content) < 200:  # Don't make it too long
                    example = random.choice(examples)
                    enhanced_content += f" For example, we can help with {example}."
            
            # Add call-to-action if appropriate
            if any(word in enhanced_content.lower() for word in ['help', 'assist', 'support']):
                enhanced_content += " What specific challenge can I help you solve?"
            
            return enhanced_content
            
        except Exception as e:
            logger.error(f"[CONTENT] Error enhancing content: {e}")
            return content
    
    async def _calculate_quality_score(self, content: str, intent: str, topic: str) -> float:
        """Calculate content quality score"""
        try:
            score = 0.5  # Base score
            
            # Length appropriateness
            if 50 <= len(content) <= 300:
                score += 0.2
            elif len(content) > 300:
                score += 0.1
            
            # Intent alignment
            if intent in self.templates:
                score += 0.2
            
            # Topic relevance
            if topic in self.knowledge_base:
                score += 0.1
            
            # Professional tone
            professional_words = ['specialize', 'expertise', 'solutions', 'help', 'assist']
            if any(word in content.lower() for word in professional_words):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"[CONTENT] Error calculating quality score: {e}")
            return 0.5
    
    async def _calculate_personalization_level(self, content: str, user_profile: Dict[str, Any]) -> float:
        """Calculate personalization level"""
        try:
            level = 0.0
            
            # Name personalization
            if user_profile.get('name') and user_profile['name'] in content:
                level += 0.3
            
            # Industry personalization
            if user_profile.get('industry') and user_profile['industry'].lower() in content.lower():
                level += 0.3
            
            # Company size personalization
            if user_profile.get('company_size') and any(size in content.lower() for size in ['small', 'enterprise', 'medium']):
                level += 0.2
            
            # Previous interaction reference
            if user_profile.get('conversation_count', 0) > 1 and 'again' in content.lower():
                level += 0.2
            
            return min(level, 1.0)
            
        except Exception as e:
            logger.error(f"[CONTENT] Error calculating personalization level: {e}")
            return 0.0
    
    async def _generate_suggestions(self, content: str, topic: str, context: Dict[str, Any]) -> List[str]:
        """Generate content improvement suggestions"""
        try:
            suggestions = []
            
            # Length suggestions
            if len(content) < 50:
                suggestions.append("Consider adding more detail to provide better value")
            elif len(content) > 400:
                suggestions.append("Consider shortening the response for better readability")
            
            # Topic-specific suggestions
            if topic in self.knowledge_base:
                if not any(example in content.lower() for example in self.knowledge_base[topic].get('examples', [])):
                    suggestions.append("Add a specific example to illustrate the point")
            
            # Personalization suggestions
            user_profile = context.get('user_profile', {})
            if user_profile.get('name') and user_profile['name'] not in content:
                suggestions.append("Consider personalizing with the user's name")
            
            # Call-to-action suggestions
            if not any(word in content.lower() for word in ['?', 'what', 'how', 'tell me']):
                suggestions.append("Add a question to encourage engagement")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"[CONTENT] Error generating suggestions: {e}")
            return []
    
    async def summarize_content(self, content: str, max_length: int = 100) -> str:
        """Summarize content to specified length"""
        if not content or not isinstance(content, str):
            return ""
        if len(content) > 100000:
            content = content[:100000]
        if not isinstance(max_length, int) or max_length < 1 or max_length > 10000:
            max_length = 100
        
        try:
            if len(content) <= max_length:
                return content
            
            # Simple summarization by taking first sentences
            sentences = content.split('. ')
            summary = ""
            
            for sentence in sentences:
                if len(summary + sentence) <= max_length:
                    summary += sentence + ". "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"[CONTENT] Error summarizing content: {e}")
            return content[:max_length] + "..."
    
    async def optimize_content(self, content: str, optimization_goals: List[str]) -> str:
        """Optimize content for specific goals"""
        if not content or not isinstance(content, str):
            return ""
        if len(content) > 100000:
            content = content[:100000]
        if not isinstance(optimization_goals, list):
            optimization_goals = []
        # Filter out invalid goals
        valid_goals = ['engagement', 'clarity', 'professionalism']
        optimization_goals = [g for g in optimization_goals if isinstance(g, str) and g in valid_goals]
        
        try:
            optimized_content = content
            
            for goal in optimization_goals:
                if goal == "engagement":
                    # Add questions to increase engagement
                    if not any(word in optimized_content.lower() for word in ['?', 'what', 'how']):
                        optimized_content += " What would you like to know more about?"
                
                elif goal == "clarity":
                    # Simplify complex sentences
                    optimized_content = re.sub(r'(\w+), (\w+), and (\w+)', r'\1, \2, and \3', optimized_content)
                
                elif goal == "professionalism":
                    casual_words = {'hey', 'yeah', 'cool', 'awesome'}
                    for word in casual_words:
                        if word in optimized_content.lower():
                            optimized_content = optimized_content.replace(word, 'excellent')
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"[CONTENT] Error optimizing content: {e}")
            return content
    
    def _get_fallback_content(self, user_message: str) -> GeneratedContent:
        """Get fallback content when generation fails"""
        return GeneratedContent(
            content="I'm here to help! How can I assist you with your business automation or website development needs?",
            metadata={
                "intent": "general",
                "topic": "general",
                "generation_time": datetime.now().isoformat(),
                "template_used": "fallback",
                "personalization_applied": False
            },
            quality_score=0.5,
            personalization_level=0.0,
            suggestions=["Consider adding more context", "Try a more specific question"]
        )
    
    def _plan_response_strategy(self, intent: str, topic: str, patterns: List, docs: List) -> Dict[str, Any]:
        """Plan the optimal response strategy based on available data"""
        strategy = {
            'use_documents': len(docs) > 0,
            'use_patterns': len(patterns) > 0,
            'confidence': 'high' if len(patterns) > 0 else 'medium',
            'approach': 'document-based' if len(docs) > 0 else 'pattern-based' if len(patterns) > 0 else 'template-based'
        }
        return strategy
    
    def _evaluate_response_quality(self, content: str, quality_score: float, personalization: float) -> Dict[str, Any]:
        """Evaluate if response needs improvement"""
        needs_improvement = quality_score < 0.6 or personalization < 0.3
        improvement_areas = []
        
        if quality_score < 0.6:
            improvement_areas.append('quality')
        if personalization < 0.3:
            improvement_areas.append('personalization')
        if len(content) < 50:
            improvement_areas.append('detail')
        
        return {
            'needs_improvement': needs_improvement,
            'improvement_areas': improvement_areas,
            'current_scores': {'quality': quality_score, 'personalization': personalization}
        }
    
    def _refine_response(self, content: str, improvement_areas: List[str]) -> str:
        """Refine response based on evaluation"""
        refined = content
        
        if 'detail' in improvement_areas:
            refined += " I'd be happy to provide more specific guidance if you share more details about your needs."
        
        if 'personalization' in improvement_areas:
            refined = refined.replace("I'm", "I'm dedicated to")
            refined = refined.replace("can", "would love to")
        
        if 'quality' in improvement_areas:
            # Add professional context
            refined = "To provide the best assistance, " + refined.lower()
        
        return refined
    
    async def close(self):
        """Close the content generator"""
        # Session is not used with FastAPI, so nothing to close
        logger.info("[CONTENT] Epsilon AI Content Generator closed")

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Global content generator instance
content_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global content_generator
    content_generator = EpsilonContentGenerator()
    await content_generator.initialize()
    yield
    # Shutdown
    if content_generator:
        await content_generator.close()

app = FastAPI(title="Epsilon AI Content Generator", version="1.0.0", lifespan=lifespan)

class ContentGenerationRequest(BaseModel):
    user_message: str
    context: Dict[str, Any] = {}

class ContentOptimizationRequest(BaseModel):
    content: str
    optimization_goals: List[str] = []

class ContentSummarizationRequest(BaseModel):
    content: str
    max_length: int = 100


@app.post("/generate-response")
async def generate_response(request: ContentGenerationRequest):
    """Generate personalized response content"""
    try:
        content = await content_generator.generate_response(request.user_message, request.context)
        return {
            "content": content.content,
            "metadata": content.metadata,
            "quality_score": content.quality_score,
            "personalization_level": content.personalization_level,
            "suggestions": content.suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-content")
async def optimize_content(request: ContentOptimizationRequest):
    """Optimize content for specific goals"""
    try:
        optimized = await content_generator.optimize_content(request.content, request.optimization_goals)
        return {"optimized_content": optimized}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-content")
async def summarize_content(request: ContentSummarizationRequest):
    """Summarize content to specified length"""
    try:
        summary = await content_generator.summarize_content(request.content, request.max_length)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Epsilon AI Content Generator"}

@app.head("/")
async def root_head():
    """Root endpoint for health checks"""
    return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

