#!/usr/bin/env python3
"""
Epsilon AI - Advanced NLP Processing Service
© 2025 Neural Operation's & Holding's LLC. All rights reserved.

This service provides advanced NLP capabilities for Epsilon AI including:
- Sentiment analysis
- Entity extraction
- Intent classification
- Text summarization
- Language detection
- Content generation
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NLPAnalysis:
    """Structured NLP analysis results"""
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    intent: str
    confidence: float
    language: str
    keywords: List[str]
    summary: Optional[str] = None
    topics: List[str] = None

class EpsilonNLPProcessor:
    """Advanced NLP processing for Epsilon AI"""
    
    def __init__(self):
        self.session = None
        self.models_loaded = False
        
    async def initialize(self):
        """Initialize the NLP processor"""
        try:
            # self.session = aiohttp.ClientSession()  # Not needed - using FastAPI
            await self._load_models()
            logger.info("[NLP] Epsilon AI NLP Processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[NLP] Failed to initialize NLP processor: {e}")
            return False
    
    async def _load_models(self):
        """Load NLP models (placeholder for actual model loading)"""
        # In production, this would load actual NLP models
        # For now, we'll use API-based services
        self.models_loaded = True
        logger.info("[NLP] NLP models loaded")
    
    async def analyze_text(self, text: str, analysis_type: str = "full") -> NLPAnalysis:
        """
        Perform comprehensive NLP analysis on text
        
        Args:
            text: Input text to analyze
            analysis_type: Type of analysis (full, sentiment, entities, intent)
        
        Returns:
            NLPAnalysis object with results
        """
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input, using fallback")
            return self._get_fallback_analysis("")
        if len(text) > 100000:  # Prevent DoS
            logger.warning("Text too long, truncating to 100KB")
            text = text[:100000]
        if not isinstance(analysis_type, str) or analysis_type not in ["full", "sentiment", "entities", "intent"]:
            analysis_type = "full"
        
        try:
            logger.info(f"🔍 Analyzing text: {text[:50]}...")
            
            # Perform different types of analysis
            if analysis_type == "full" or analysis_type == "sentiment":
                sentiment = await self._analyze_sentiment(text)
            else:
                sentiment = {"positive": 0.5, "negative": 0.5, "neutral": 0.5}
            
            if analysis_type == "full" or analysis_type == "entities":
                entities = await self._extract_entities(text)
            else:
                entities = []
            
            if analysis_type == "full" or analysis_type == "intent":
                intent, confidence = await self._classify_intent(text)
            else:
                intent, confidence = "general", 0.5
            
            # Always get language and keywords
            language = await self._detect_language(text)
            keywords = await self._extract_keywords(text)
            
            # Get topics if full analysis
            topics = []
            if analysis_type == "full":
                topics = await self._extract_topics(text)
            
            analysis = NLPAnalysis(
                sentiment=sentiment,
                entities=entities,
                intent=intent,
                confidence=confidence,
                language=language,
                keywords=keywords,
                topics=topics
            )
            
            logger.info(f"[NLP] NLP analysis complete: {intent} ({confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"[NLP] Error in NLP analysis: {e}")
            return self._get_fallback_analysis(text)
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            # Use a simple rule-based sentiment analysis
            # In production, this would use a trained model or API
            positive_words = ["good", "great", "excellent", "amazing", "love", "like", "happy", "satisfied"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "disappointed"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            
            positive_score = min(positive_count / total_words * 3, 1.0)
            negative_score = min(negative_count / total_words * 3, 1.0)
            neutral_score = 1.0 - positive_score - negative_score
            
            # Normalize
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            return {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }
        except Exception as e:
            logger.error(f"[NLP] Sentiment analysis error: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return []
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            # Simple entity extraction based on patterns
            # In production, this would use spaCy, NLTK, or similar
            entities = []
            
            # Email pattern
            import re
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            for email in emails:
                entities.append({
                    "text": email,
                    "label": "EMAIL",
                    "confidence": 0.9
                })
            
            # Phone pattern
            phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
            for phone in phones:
                entities.append({
                    "text": phone,
                    "label": "PHONE",
                    "confidence": 0.8
                })
            
            # Company/business keywords
            business_words = ["company", "business", "corporation", "llc", "inc", "ltd"]
            for word in business_words:
                if word.lower() in text.lower():
                    entities.append({
                        "text": word,
                        "label": "ORGANIZATION",
                        "confidence": 0.7
                    })
            
            return entities
        except Exception as e:
            logger.error(f"[NLP] Entity extraction error: {e}")
            return []
    
    async def _classify_intent(self, text: str) -> tuple[str, float]:
        """Classify user intent"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return "general", 0.5
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            text_lower = text.lower()
            
            # Intent classification patterns
            intents = {
                "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "question": ["what", "how", "why", "when", "where", "who", "?"],
                "request": ["can you", "could you", "please", "help me", "i need"],
                "complaint": ["problem", "issue", "wrong", "not working", "error"],
                "compliment": ["thank you", "thanks", "great", "excellent", "love"],
                "goodbye": ["bye", "goodbye", "see you", "farewell"],
                "business_inquiry": ["website", "automation", "development", "services", "pricing"],
                "technical_support": ["bug", "fix", "support", "technical", "problem"]
            }
            
            best_intent = "general"
            best_confidence = 0.0
            
            for intent, keywords in intents.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                # Division by zero protection
                confidence = matches / max(len(keywords), 1) if keywords else 0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent
            
            # Boost confidence for longer matches
            if best_confidence > 0:
                best_confidence = min(best_confidence * 1.5, 1.0)
            
            return best_intent, best_confidence
        except Exception as e:
            logger.error(f"[NLP] Intent classification error: {e}")
            return "general", 0.5
    
    async def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return "en"
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            # Simple language detection based on common words
            # In production, this would use langdetect or similar
            english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
            spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le"]
            
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if word in text_lower)
            spanish_count = sum(1 for word in spanish_words if word in text_lower)
            
            if english_count > spanish_count:
                return "en"
            elif spanish_count > english_count:
                return "es"
            else:
                return "en"  # Default to English
        except Exception as e:
            logger.error(f"[NLP] Language detection error: {e}")
            return "en"
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return []
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            # Simple keyword extraction
            # In production, this would use TF-IDF, TextRank, or similar
            import re
            
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            keywords = [word for word in words if word not in stop_words]
            
            # Count frequency and return top keywords
            from collections import Counter
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(10)]
        except Exception as e:
            logger.error(f"[NLP] Keyword extraction error: {e}")
            return []
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Safety check: validate inputs
        if not text or not isinstance(text, str):
            return []
        if len(text) > 100000:  # Prevent DoS
            text = text[:100000]
        
        try:
            # Simple topic extraction based on business domains
            topics = []
            text_lower = text.lower()
            
            topic_keywords = {
                "business_automation": ["automation", "workflow", "process", "efficiency"],
                "website_development": ["website", "web", "development", "design", "site"],
                "ai_strategy": ["ai", "artificial intelligence", "machine learning", "strategy"],
                "technical_support": ["support", "help", "technical", "bug", "issue"],
                "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable"],
                "services": ["service", "offer", "provide", "deliver", "solution"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
        except Exception as e:
            logger.error(f"[NLP] Topic extraction error: {e}")
            return []
    
    def _get_fallback_analysis(self, text: str) -> NLPAnalysis:
        """Get fallback analysis when main analysis fails"""
        return NLPAnalysis(
            sentiment={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            entities=[],
            intent="general",
            confidence=0.5,
            language="en",
            keywords=[],
            topics=[]
        )
    
    async def generate_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Generate AI response based on NLP analysis"""
        # Safety check: validate inputs
        if not user_message or not isinstance(user_message, str):
            return "I'm here to help! How can I assist you with your business automation or website development needs?"
        if len(user_message) > 10000:  # Prevent DoS
            logger.warning("User message too long, truncating to 10KB")
            user_message = user_message[:10000]
        if context is not None and not isinstance(context, dict):
            context = {}
        
        try:
            logger.info(f"🤖 Generating response for: {user_message[:50]}...")
            
            # Analyze the user message
            analysis = await self.analyze_text(user_message)
            
            # Generate response based on intent and context
            if analysis.intent == "greeting":
                return "Hello! I'm Epsilon AI, your AI automation specialist. How can I help transform your business today?"
            elif analysis.intent == "business_inquiry":
                return "I'd love to help with your business needs! We specialize in website development, business automation, and AI strategy. What specific challenge can I help you solve?"
            elif analysis.intent == "technical_support":
                return "I'm here to help with any technical issues. Can you describe the problem you're experiencing in more detail?"
            elif analysis.intent == "question":
                return "That's a great question! Let me provide you with a detailed answer. What specific aspect would you like me to focus on?"
            elif analysis.intent == "complaint":
                return "I apologize for any inconvenience. Let me help resolve this issue for you. Can you provide more details about what went wrong?"
            elif analysis.intent == "compliment":
                return "Thank you so much! I'm glad I could help. Is there anything else I can assist you with?"
            elif analysis.intent == "goodbye":
                return "Thank you for chatting with me today! Feel free to reach out anytime you need assistance. Have a great day!"
            else:
                return "I understand you're looking for assistance. As Epsilon AI, I specialize in business automation, website development, and AI strategy. How can I help you today?"
                
        except Exception as e:
            logger.error(f"[NLP] Error generating response: {e}")
            return "I'm here to help! How can I assist you with your business automation or website development needs?"
    
    async def close(self):
        """Close the NLP processor"""
        # Session is not used with FastAPI, so nothing to close
        logger.info("[NLP] Epsilon AI NLP Processor closed")

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Global NLP processor instance
nlp_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global nlp_processor
    nlp_processor = EpsilonNLPProcessor()
    await nlp_processor.initialize()
    yield
    # Shutdown
    if nlp_processor:
        await nlp_processor.close()

app = FastAPI(title="Epsilon AI NLP Processor", version="1.0.0", lifespan=lifespan)

class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "full"

class TextAnalysisResponse(BaseModel):
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    intent: str
    confidence: float
    language: str
    keywords: List[str]
    topics: List[str] = []

class ResponseGenerationRequest(BaseModel):
    user_message: str
    context: Dict[str, Any] = {}


@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text with NLP"""
    # Safety check: validate inputs
    if not request or not request.text:
        raise HTTPException(status_code=400, detail="text is required")
    if not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="text must be a string")
    if len(request.text) > 100000:  # Prevent DoS - 100KB max
        raise HTTPException(status_code=400, detail="text too long (max 100KB)")
    if not isinstance(request.analysis_type, str):
        raise HTTPException(status_code=400, detail="analysis_type must be a string")
    if len(request.analysis_type) > 50:  # Prevent DoS
        raise HTTPException(status_code=400, detail="analysis_type too long (max 50 characters)")
    if request.analysis_type not in ["full", "sentiment", "entities", "intent"]:
        raise HTTPException(status_code=400, detail="analysis_type must be one of: full, sentiment, entities, intent")
    
    try:
        analysis = await nlp_processor.analyze_text(request.text, request.analysis_type)
        return TextAnalysisResponse(
            sentiment=analysis.sentiment,
            entities=analysis.entities,
            intent=analysis.intent,
            confidence=analysis.confidence,
            language=analysis.language,
            keywords=analysis.keywords,
            topics=analysis.topics or []
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-response")
async def generate_response(request: ResponseGenerationRequest):
    """Generate AI response"""
    # Safety check: validate inputs
    if not request or not request.user_message:
        raise HTTPException(status_code=400, detail="user_message is required")
    if not isinstance(request.user_message, str):
        raise HTTPException(status_code=400, detail="user_message must be a string")
    if len(request.user_message) > 10000:  # Prevent DoS - 10KB max
        raise HTTPException(status_code=400, detail="user_message too long (max 10KB)")
    if request.context is not None and not isinstance(request.context, dict):
        raise HTTPException(status_code=400, detail="context must be a dictionary")
    if request.context:
        import sys
        context_size = sys.getsizeof(str(request.context))
        if context_size > 1024 * 1024:  # 1MB max
            raise HTTPException(status_code=400, detail="context too large (max 1MB)")
    
    try:
        response = await nlp_processor.generate_response(request.user_message, request.context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Epsilon AI NLP Processor"}

@app.head("/")
async def root_head():
    """Root endpoint for health checks"""
    return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

