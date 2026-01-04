#!/usr/bin/env python3
"""
Epsilon AI - Advanced Learning & Analytics Service
© 2025 Neural Operation's & Holding's LLC. All rights reserved.

This service provides advanced machine learning and analytics for Epsilon AI including:
- Conversation pattern analysis
- User behavior prediction
- Response quality optimization
- Performance metrics and insights
- A/B testing and experimentation
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationMetrics:
    """Conversation performance metrics"""
    conversation_id: str
    user_satisfaction: float
    response_time: float
    message_count: int
    topics_covered: List[str]
    intent_accuracy: float
    engagement_score: float
    conversion_probability: float

@dataclass
class UserProfile:
    """User behavior profile"""
    user_id: str
    preferred_topics: List[str]
    communication_style: str
    technical_level: str
    engagement_pattern: Dict[str, Any]
    satisfaction_trend: List[float]
    conversion_likelihood: float

@dataclass
class LearningInsight:
    """Learning insight from data analysis"""
    insight_type: str
    description: str
    confidence: float
    actionable: bool
    impact_score: float
    recommendations: List[str]

class EpsilonLearningAnalytics:
    """Advanced learning and analytics for Epsilon AI"""
    
    def __init__(self):
        self.session = None
        self.conversation_data = []
        self.user_profiles = {}
        self.learning_models = {}
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the learning analytics service"""
        try:
            # self.session = aiohttp.ClientSession()  # Not needed - using FastAPI
            await self._load_learning_models()
            logger.info("[ANALYTICS] Epsilon AI Learning Analytics initialized successfully")
            return True
        except Exception as e:
            logger.error(f"[ANALYTICS] Failed to initialize Learning Analytics: {e}")
            return False
    
    async def _load_learning_models(self):
        """Load machine learning models - real statistical analysis models"""
        # Real statistical analysis models for learning
        self.learning_models = {
            "satisfaction_predictor": self._predict_satisfaction,
            "conversion_predictor": self._predict_conversion,
            "response_optimizer": self._optimize_response,
            "topic_recommender": self._recommend_topics
        }
        logger.info("[ANALYTICS] Learning models loaded")
    
    async def analyze_conversation(self, conversation_data: Dict[str, Any]) -> ConversationMetrics:
        """Analyze conversation performance"""
        # Safety check: validate inputs
        if not conversation_data or not isinstance(conversation_data, dict):
            logger.error("[ANALYTICS] Invalid conversation_data: must be a dictionary")
            raise ValueError("conversation_data must be a non-empty dictionary")
        
        try:
            logger.info(f"[ANALYTICS] Analyzing conversation: {conversation_data.get('conversation_id', 'unknown')}")
            
            # Extract metrics with validation
            conversation_id = conversation_data.get('conversation_id', '')
            if not isinstance(conversation_id, str):
                conversation_id = str(conversation_id) if conversation_id else ''
            if len(conversation_id) > 200:  # Prevent DoS
                conversation_id = conversation_id[:200]
            
            messages = conversation_data.get('messages', [])
            if not isinstance(messages, list):
                messages = []
            if len(messages) > 10000:  # Prevent DoS
                messages = messages[:10000]
            
            feedback = conversation_data.get('feedback', [])
            if not isinstance(feedback, list):
                feedback = []
            if len(feedback) > 1000:  # Prevent DoS
                feedback = feedback[:1000]
            
            # Calculate user satisfaction
            user_satisfaction = await self._calculate_satisfaction(feedback, messages)
            
            # Calculate response time
            response_time = await self._calculate_response_time(messages)
            
            # Count messages
            message_count = len(messages)
            
            # Extract topics
            topics_covered = await self._extract_conversation_topics(messages)
            
            # Calculate intent accuracy
            intent_accuracy = await self._calculate_intent_accuracy(messages)
            
            # Calculate engagement score
            engagement_score = await self._calculate_engagement_score(messages, feedback)
            
            # Predict conversion probability
            conversion_probability = await self._predict_conversion_probability(conversation_data)
            
            metrics = ConversationMetrics(
                conversation_id=conversation_id,
                user_satisfaction=user_satisfaction,
                response_time=response_time,
                message_count=message_count,
                topics_covered=topics_covered,
                intent_accuracy=intent_accuracy,
                engagement_score=engagement_score,
                conversion_probability=conversion_probability
            )
            
            # Store for learning
            self.conversation_data.append(metrics)
            
            logger.info(f"[ANALYTICS] Conversation analysis complete: satisfaction={user_satisfaction:.2f}, engagement={engagement_score:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error analyzing conversation: {e}")
            import traceback
            logger.error(f"[ANALYTICS] Traceback: {traceback.format_exc()}")
            # Re-throw error instead of masking with fallback
            raise
    
    async def _calculate_satisfaction(self, feedback: List[Dict], messages: List[Dict]) -> float:
        """Calculate user satisfaction score"""
        try:
            if not feedback:
                return 0.5  # Neutral if no feedback
            
            # Calculate average rating
            ratings = [f.get('rating', 3) for f in feedback if 'rating' in f]
            if ratings:
                avg_rating = statistics.mean(ratings)
                return avg_rating / 5.0  # Normalize to 0-1
            
            # Use sentiment from feedback text
            positive_words = ["good", "great", "excellent", "amazing", "love", "like", "happy", "satisfied"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "disappointed"]
            
            total_sentiment = 0
            for f in feedback:
                text = f.get('feedback_text', '').lower()
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                if positive_count > negative_count:
                    total_sentiment += 0.8
                elif negative_count > positive_count:
                    total_sentiment += 0.2
                else:
                    total_sentiment += 0.5
            
            return total_sentiment / len(feedback) if feedback else 0.5
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error calculating satisfaction: {e}")
            return 0.5
    
    async def _calculate_response_time(self, messages: List[Dict]) -> float:
        """Calculate average response time"""
        try:
            response_times = []
            for i in range(1, len(messages), 2):  # Every other message (Epsilon AI responses)
                if i < len(messages):
                    current_time = messages[i].get('timestamp', 0)
                    previous_time = messages[i-1].get('timestamp', 0)
                    if current_time > previous_time:
                        response_times.append(current_time - previous_time)
            
            return statistics.mean(response_times) if response_times else 0.0
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error calculating response time: {e}")
            return 0.0
    
    async def _extract_conversation_topics(self, messages: List[Dict]) -> List[str]:
        """Extract topics covered in conversation"""
        try:
            topic_keywords = {
                "business_automation": ["automation", "workflow", "process", "efficiency"],
                "website_development": ["website", "web", "development", "design", "site"],
                "ai_strategy": ["ai", "artificial intelligence", "machine learning", "strategy"],
                "technical_support": ["support", "help", "technical", "bug", "issue"],
                "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable"],
                "services": ["service", "offer", "provide", "deliver", "solution"]
            }
            
            topics = set()
            for message in messages:
                text = message.get('content', '').lower()
                for topic, keywords in topic_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        topics.add(topic)
            
            return list(topics)
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error extracting topics: {e}")
            return []
    
    async def _calculate_intent_accuracy(self, messages: List[Dict]) -> float:
        """Calculate intent classification accuracy"""
        try:
            # This would be calculated based on how well Epsilon AI understood user intents
            # For now, we'll use a simple heuristic
            user_messages = [m for m in messages if m.get('sender') == 'user']
            if not user_messages:
                return 0.5
            
            # Check if Epsilon AI's responses were relevant to user messages
            accuracy_scores = []
            for i, user_msg in enumerate(user_messages):
                if i * 2 + 1 < len(messages):  # Check corresponding Epsilon AI response
                    epsilon_response = messages[i * 2 + 1]
                    # Simple relevance check based on keyword overlap
                    user_words = set(user_msg.get('content', '').lower().split())
                    epsilon_words = set(epsilon_response.get('content', '').lower().split())
                    
                    if user_words and epsilon_words:
                        overlap = len(user_words.intersection(epsilon_words))
                        # Division by zero protection
                        accuracy = min(overlap / max(len(user_words), 1), 1.0)
                        accuracy_scores.append(accuracy)
            
            return statistics.mean(accuracy_scores) if accuracy_scores else 0.5
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error calculating intent accuracy: {e}")
            return 0.5
    
    async def _calculate_engagement_score(self, messages: List[Dict], feedback: List[Dict]) -> float:
        """Calculate user engagement score"""
        try:
            # Safety check: validate inputs
            if not messages or not isinstance(messages, list):
                return 0.5  # Default neutral engagement
            if feedback is not None and not isinstance(feedback, list):
                feedback = []
            
            # Factors that indicate engagement:
            # 1. Number of messages
            # 2. Message length
            # 3. Questions asked
            # 4. Positive feedback
            
            user_messages = [m for m in messages if m.get('sender') == 'user']
            message_count = len(user_messages)
            
            # Calculate average message length with division by zero protection
            message_lengths = [len(m.get('content', '')) for m in user_messages]
            avg_message_length = statistics.mean(message_lengths) if message_lengths else 0.0
            
            # Count questions
            questions = sum(1 for m in messages if '?' in m.get('content', '') and m.get('sender') == 'user')
            
            # Positive feedback score
            positive_feedback = sum(1 for f in feedback if f.get('rating', 0) >= 4)
            feedback_score = positive_feedback / len(feedback) if feedback else 0.5
            
            # Calculate engagement score (0-1)
            message_score = min(message_count / 10, 1.0)  # Normalize to 10 messages
            length_score = min(avg_message_length / 100, 1.0)  # Normalize to 100 chars
            question_score = min(questions / 5, 1.0)  # Normalize to 5 questions
            
            engagement = (message_score * 0.3 + length_score * 0.2 + question_score * 0.2 + feedback_score * 0.3)
            return min(engagement, 1.0)
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error calculating engagement score: {e}")
            return 0.5
    
    async def _predict_conversion_probability(self, conversation_data: Dict[str, Any]) -> float:
        """Predict probability of user conversion"""
        try:
            # Factors that indicate conversion likelihood:
            # 1. High engagement
            # 2. Business-related topics
            # 3. Positive sentiment
            # 4. Multiple interactions
            
            messages = conversation_data.get('messages', [])
            feedback = conversation_data.get('feedback', [])
            
            # Business topic score
            business_topics = ["business_automation", "website_development", "ai_strategy", "services", "pricing"]
            topics = await self._extract_conversation_topics(messages)
            business_score = len(set(topics).intersection(set(business_topics))) / len(business_topics)
            
            # Engagement score
            engagement = await self._calculate_engagement_score(messages, feedback)
            
            # Sentiment score
            satisfaction = await self._calculate_satisfaction(feedback, messages)
            
            # Interaction depth
            interaction_depth = min(len(messages) / 20, 1.0)  # Normalize to 20 messages
            
            # Weighted combination
            conversion_prob = (
                business_score * 0.4 +
                engagement * 0.3 +
                satisfaction * 0.2 +
                interaction_depth * 0.1
            )
            
            return min(conversion_prob, 1.0)
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error predicting conversion: {e}")
            return 0.3
    
    async def generate_learning_insights(self, time_period: str = "7d") -> List[LearningInsight]:
        """Generate learning insights from conversation data"""
        # Safety check: validate inputs
        if not isinstance(time_period, str):
            time_period = "7d"
        if len(time_period) > 50:  # Prevent DoS
            time_period = "7d"
        
        try:
            logger.info(f"[ANALYTICS] Generating learning insights for period: {time_period}")
            
            insights = []
            
            # Analyze conversation patterns
            if len(self.conversation_data) > 0:
                # Satisfaction trend insight
                satisfaction_scores = [c.user_satisfaction for c in self.conversation_data]
                avg_satisfaction = statistics.mean(satisfaction_scores)
                
                if avg_satisfaction > 0.7:
                    insights.append(LearningInsight(
                        insight_type="performance",
                        description=f"High user satisfaction: {avg_satisfaction:.2f}",
                        confidence=0.9,
                        actionable=True,
                        impact_score=0.8,
                        recommendations=["Continue current approach", "Document successful patterns"]
                    ))
                elif avg_satisfaction < 0.5:
                    insights.append(LearningInsight(
                        insight_type="improvement",
                        description=f"Low user satisfaction: {avg_satisfaction:.2f}",
                        confidence=0.9,
                        actionable=True,
                        impact_score=0.9,
                        recommendations=["Review response quality", "Improve intent understanding", "Enhance user experience"]
                    ))
                
                # Response time insight
                response_times = [c.response_time for c in self.conversation_data if c.response_time > 0]
                if response_times:
                    avg_response_time = statistics.mean(response_times)
                    if avg_response_time > 5000:  # 5 seconds
                        insights.append(LearningInsight(
                            insight_type="performance",
                            description=f"Slow response times: {avg_response_time:.0f}ms average",
                            confidence=0.8,
                            actionable=True,
                            impact_score=0.7,
                            recommendations=["Optimize response generation", "Cache frequent responses", "Improve system performance"]
                        ))
                
                # Topic popularity insight
                all_topics = []
                for conv in self.conversation_data:
                    all_topics.extend(conv.topics_covered)
                
                if all_topics:
                    topic_counts = Counter(all_topics)
                    most_popular = topic_counts.most_common(1)[0]
                    
                    insights.append(LearningInsight(
                        insight_type="trend",
                        description=f"Most popular topic: {most_popular[0]} ({most_popular[1]} conversations)",
                        confidence=0.9,
                        actionable=True,
                        impact_score=0.6,
                        recommendations=[f"Enhance {most_popular[0]} knowledge", "Create specialized responses", "Develop topic-specific workflows"]
                    ))
            
            logger.info(f"[ANALYTICS] Generated {len(insights)} learning insights")
            return insights
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error generating insights: {e}")
            return []
    
    async def optimize_response_strategy(self, user_profile: Dict[str, Any], conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize response strategy based on user profile and context"""
        # Safety check: validate inputs
        if not user_profile or not isinstance(user_profile, dict):
            user_profile = {}
        if not conversation_context or not isinstance(conversation_context, dict):
            conversation_context = {}
        
        try:
            logger.info("[ANALYTICS] Optimizing response strategy")
            
            # Analyze user profile with validation
            preferred_topics = user_profile.get('preferred_topics', [])
            if not isinstance(preferred_topics, list):
                preferred_topics = []
            if len(preferred_topics) > 100:  # Prevent DoS
                preferred_topics = preferred_topics[:100]
            
            communication_style = user_profile.get('communication_style', 'professional')
            if not isinstance(communication_style, str) or len(communication_style) > 50:
                communication_style = 'professional'
            
            technical_level = user_profile.get('technical_level', 'intermediate')
            if not isinstance(technical_level, str) or len(technical_level) > 50:
                technical_level = 'intermediate'
            
            # Analyze conversation context with validation
            current_topics = conversation_context.get('topics', [])
            if not isinstance(current_topics, list):
                current_topics = []
            if len(current_topics) > 100:  # Prevent DoS
                current_topics = current_topics[:100]
            
            user_sentiment = conversation_context.get('sentiment', 'neutral')
            if not isinstance(user_sentiment, str) or len(user_sentiment) > 50:
                user_sentiment = 'neutral'
            
            conversation_stage = conversation_context.get('stage', 'initial')
            if not isinstance(conversation_stage, str) or len(conversation_stage) > 50:
                conversation_stage = 'initial'
            
            # Generate optimization recommendations
            strategy = {
                "tone": self._optimize_tone(communication_style, user_sentiment),
                "complexity": self._optimize_complexity(technical_level),
                "topics": self._optimize_topics(preferred_topics, current_topics),
                "approach": self._optimize_approach(conversation_stage, user_sentiment),
                "personalization": self._optimize_personalization(user_profile)
            }
            
            logger.info("[ANALYTICS] Response strategy optimized")
            return strategy
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Error optimizing response strategy: {e}")
            return self._get_default_strategy()
    
    def _optimize_tone(self, communication_style: str, sentiment: str) -> str:
        """Optimize communication tone"""
        if sentiment == "negative":
            return "empathetic"
        elif communication_style == "casual":
            return "friendly"
        elif communication_style == "formal":
            return "professional"
        else:
            return "conversational"
    
    def _optimize_complexity(self, technical_level: str) -> str:
        """Optimize technical complexity"""
        if technical_level == "beginner":
            return "simple"
        elif technical_level == "expert":
            return "detailed"
        else:
            return "moderate"
    
    def _optimize_topics(self, preferred_topics: List[str], current_topics: List[str]) -> List[str]:
        """Optimize topic focus"""
        # Combine preferred and current topics, prioritizing preferred
        all_topics = list(set(preferred_topics + current_topics))
        return all_topics[:5]  # Limit to top 5 topics
    
    def _optimize_approach(self, stage: str, sentiment: str) -> str:
        """Optimize conversation approach"""
        if stage == "initial":
            return "welcoming"
        elif stage == "qualification":
            return "inquisitive"
        elif stage == "objection_handling":
            return "persuasive"
        elif sentiment == "negative":
            return "supportive"
        else:
            return "helpful"
    
    def _optimize_personalization(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize personalization elements"""
        return {
            "use_name": user_profile.get('name') is not None,
            "reference_history": user_profile.get('conversation_count', 0) > 1,
            "customize_examples": user_profile.get('industry') is not None,
            "adaptive_style": True
        }
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default response strategy"""
        return {
            "tone": "professional",
            "complexity": "moderate",
            "topics": [],
            "approach": "helpful",
            "personalization": {
                "use_name": False,
                "reference_history": False,
                "customize_examples": False,
                "adaptive_style": True
            }
        }
    
    def _predict_satisfaction(self, conversation_data: Dict[str, Any]) -> float:
        """Predict user satisfaction based on conversation data"""
        try:
            # Simple satisfaction prediction based on conversation metrics
            messages = conversation_data.get('messages', [])
            feedback = conversation_data.get('feedback', [])
            
            if not messages:
                return 0.5  # Default neutral satisfaction
            
            # Calculate satisfaction based on conversation length and feedback
            message_count = len(messages)
            positive_feedback = sum(1 for f in feedback if f.get('sentiment', 'neutral') == 'positive')
            total_feedback = len(feedback)
            
            # Base satisfaction from conversation length (longer = more engaged)
            length_score = min(message_count / 10.0, 1.0)  # Normalize to 0-1
            
            # Feedback score
            feedback_score = positive_feedback / max(total_feedback, 1)
            
            # Combined satisfaction score
            satisfaction = (length_score * 0.6) + (feedback_score * 0.4)
            return min(max(satisfaction, 0.0), 1.0)  # Clamp to 0-1
            
        except Exception as e:
            logger.error(f"Error predicting satisfaction: {e}")
            return 0.5  # Default neutral satisfaction
    
    def _predict_conversion(self, conversation_data: Dict[str, Any]) -> float:
        """Predict conversion probability based on conversation data"""
        try:
            # Simple conversion prediction
            messages = conversation_data.get('messages', [])
            user_profile = conversation_data.get('user_profile', {})
            
            if not messages:
                return 0.1  # Low conversion for no conversation
            
            # Factors that increase conversion probability
            message_count = len(messages)
            has_questions = any('?' in msg.get('content', '') for msg in messages)
            has_positive_sentiment = any(msg.get('sentiment') == 'positive' for msg in messages)
            
            # Base conversion probability
            base_prob = 0.1
            
            # Increase based on engagement
            if message_count > 5:
                base_prob += 0.2
            if has_questions:
                base_prob += 0.2
            if has_positive_sentiment:
                base_prob += 0.3
            
            return min(max(base_prob, 0.0), 1.0)  # Clamp to 0-1
            
        except Exception as e:
            logger.error(f"Error predicting conversion: {e}")
            return 0.1  # Default low conversion
    
    def _optimize_response(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize response strategy based on conversation data"""
        try:
            return {
                "tone": "professional",
                "length": "moderate",
                "complexity": "intermediate",
                "personalization": True,
                "examples": True
            }
        except Exception as e:
            logger.error(f"Error optimizing response: {e}")
            return {"tone": "professional", "length": "moderate"}
    
    def _recommend_topics(self, conversation_data: Dict[str, Any]) -> List[str]:
        """Recommend topics based on conversation data"""
        try:
            # Simple topic recommendation
            messages = conversation_data.get('messages', [])
            topics = []
            
            for message in messages:
                content = message.get('content', '').lower()
                if 'ai' in content or 'artificial intelligence' in content:
                    topics.append('AI and Machine Learning')
                elif 'automation' in content:
                    topics.append('Business Automation')
                elif 'data' in content:
                    topics.append('Data Analytics')
                elif 'security' in content:
                    topics.append('Cybersecurity')
            
            # Remove duplicates and return top topics
            return list(set(topics))[:3]
            
        except Exception as e:
            logger.error(f"Error recommending topics: {e}")
            return ['General Business']
    
    async def close(self):
        """Close the learning analytics service"""
        # Session is not used with FastAPI, so nothing to close
        logger.info("[ANALYTICS] Epsilon AI Learning Analytics closed")

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Global analytics instance
analytics = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global analytics
    analytics = EpsilonLearningAnalytics()
    await analytics.initialize()
    yield
    # Shutdown
    if analytics:
        await analytics.close()

app = FastAPI(title="Epsilon AI Learning Analytics", version="1.0.0", lifespan=lifespan)

class ConversationAnalysisRequest(BaseModel):
    conversation_data: Dict[str, Any]

class LearningInsightsRequest(BaseModel):
    time_period: str = "7d"

class ResponseOptimizationRequest(BaseModel):
    user_profile: Dict[str, Any]
    conversation_context: Dict[str, Any]


@app.post("/analyze-conversation")
async def analyze_conversation(request: ConversationAnalysisRequest):
    """Analyze conversation performance"""
    # Safety check: validate inputs
    if not request or not request.conversation_data:
        raise HTTPException(status_code=400, detail="conversation_data is required")
    if not isinstance(request.conversation_data, dict):
        raise HTTPException(status_code=400, detail="conversation_data must be a dictionary")
    # Check for reasonable size to prevent DoS
    import sys
    data_size = sys.getsizeof(str(request.conversation_data))
    if data_size > 10 * 1024 * 1024:  # 10MB max
        raise HTTPException(status_code=400, detail="conversation_data too large (max 10MB)")
    
    try:
        metrics = await analytics.analyze_conversation(request.conversation_data)
        return asdict(metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-insights")
async def generate_insights(request: LearningInsightsRequest):
    """Generate learning insights"""
    # Safety check: validate inputs
    if not request:
        raise HTTPException(status_code=400, detail="Request is required")
    if not isinstance(request.time_period, str):
        raise HTTPException(status_code=400, detail="time_period must be a string")
    if len(request.time_period) > 50:  # Prevent DoS
        raise HTTPException(status_code=400, detail="time_period too long (max 50 characters)")
    # Validate time_period format (e.g., "7d", "30d", "1m")
    import re
    if not re.match(r'^\d+[dm]$', request.time_period):
        raise HTTPException(status_code=400, detail="time_period must be in format like '7d' or '30d'")
    
    try:
        insights = await analytics.generate_learning_insights(request.time_period)
        return [asdict(insight) for insight in insights]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-response")
async def optimize_response(request: ResponseOptimizationRequest):
    """Optimize response strategy"""
    # Safety check: validate inputs
    if not request:
        raise HTTPException(status_code=400, detail="Request is required")
    if not isinstance(request.user_profile, dict):
        raise HTTPException(status_code=400, detail="user_profile must be a dictionary")
    if not isinstance(request.conversation_context, dict):
        raise HTTPException(status_code=400, detail="conversation_context must be a dictionary")
    # Check for reasonable size to prevent DoS
    import sys
    profile_size = sys.getsizeof(str(request.user_profile))
    context_size = sys.getsizeof(str(request.conversation_context))
    if profile_size > 5 * 1024 * 1024 or context_size > 5 * 1024 * 1024:  # 5MB max each
        raise HTTPException(status_code=400, detail="user_profile or conversation_context too large (max 5MB each)")
    
    try:
        strategy = await analytics.optimize_response_strategy(request.user_profile, request.conversation_context)
        return strategy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Epsilon AI Learning Analytics"}

@app.head("/")
async def root_head():
    """Root endpoint for health checks"""
    return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

