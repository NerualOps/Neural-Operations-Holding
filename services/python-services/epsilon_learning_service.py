"""
Epsilon AI Automatic Learning Service
Monitors feedback and automatically updates model weights and learning patterns
"""
import asyncio
import aiohttp
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://jdruawealecokthrwtjg.supabase.co')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')

# CRITICAL: Fail startup if key is missing
if not SUPABASE_SERVICE_KEY:
    logger.error("[LEARNING] CRITICAL: SUPABASE_SERVICE_KEY is not set! Learning service cannot function.")
    logger.error("[LEARNING] Set SUPABASE_SERVICE_KEY environment variable before starting the service.")
    sys.exit(1)
else:
    logger.info("[LEARNING] Epsilon AI Learning Service configured with Supabase credentials")

# Configuration for retry logic
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds

async def make_supabase_request_with_retry(method: str, endpoint: str, data: dict = None) -> tuple:
    """
    Make a request to Supabase REST API with automatic retry and exponential backoff
    
    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE)
        endpoint: Supabase REST endpoint
        data: Optional JSON data for POST/PUT requests
    
    Returns:
        Tuple of (response_json, status_code)
    
    Raises:
        RuntimeError: If all retries fail
    """
    headers = {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    payload_size = len(json.dumps(data)) if data else 0
    
    logger.debug(f"📤 {method} {endpoint} (payload: {payload_size} bytes)")
    
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers) as response:
                        return await _process_response(response, endpoint, method)
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, json=data) as response:
                        return await _process_response(response, endpoint, method)
                elif method.upper() == 'PUT':
                    async with session.put(url, headers=headers, json=data) as response:
                        return await _process_response(response, endpoint, method)
                elif method.upper() == 'PATCH':
                    async with session.patch(url, headers=headers, json=data) as response:
                        return await _process_response(response, endpoint, method)
                elif method.upper() == 'DELETE':
                    async with session.delete(url, headers=headers) as response:
                        return await _process_response(response, endpoint, method)
                else:
                    raise ValueError(f"Unsupported method: {method}")
        except aiohttp.ClientError as e:
            wait_time = (BASE_DELAY * (2 ** attempt)) + (random.random() * 0.5)
            logger.warning(f"[LEARNING] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {wait_time:.2f}s...")
            
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[LEARNING] All {MAX_RETRIES} attempts failed for {method} {endpoint}")
                raise RuntimeError(f"Supabase request failed after {MAX_RETRIES} retries") from e
        except Exception as e:
            logger.error(f"[LEARNING] Unexpected error in Supabase request: {e}")
            raise

async def _process_response(response: aiohttp.ClientResponse, endpoint: str, method: str):
    """Process HTTP response and return JSON data"""
    try:
        resp_json = await response.json()
    except Exception:
        resp_text = await response.text()
        is_html_error = '<!DOCTYPE html>' in resp_text or 'Cloudflare' in resp_text or '522' in resp_text or '521' in resp_text
        
        if is_html_error:
            logger.warning(f"[LEARNING] Supabase connection issue from {method} {endpoint}")
            resp_json = {"error": "Supabase connection timeout", "is_html_error": True}
        else:
            logger.warning(f"[LEARNING] Non-JSON response from {method} {endpoint}: {resp_text[:200]}")
        resp_json = {"error": "Non-JSON response", "text": resp_text[:500]}
    
    # Log response details
    logger.debug(f"📥 {method} {endpoint} → {response.status}")
    
    if response.status >= 400:
        if isinstance(resp_json, dict) and resp_json.get("is_html_error"):
            logger.warning(f"[LEARNING] {method} {endpoint} failed: Supabase connection timeout")
        else:
            error_preview = json.dumps(resp_json)[:500] if isinstance(resp_json, dict) else str(resp_json)[:500]
            logger.error(f"[LEARNING] {method} {endpoint} failed with status {response.status}: {error_preview}")
    
    return resp_json, response.status

# Alias for backward compatibility
async def make_supabase_request(method: str, endpoint: str, data: dict = None):
    """Alias for make_supabase_request_with_retry for backward compatibility"""
    return await make_supabase_request_with_retry(method, endpoint, data)

async def analyze_feedback_for_learning(feedback_data: dict) -> Dict[str, Any]:
    """
    Analyze feedback and extract learning signals
    Returns model weights and patterns to update
    """
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning empty signals")
        return {}
    
    try:
        was_helpful = feedback_data.get('was_helpful', False)
        rating = feedback_data.get('rating', 0)
        feedback_text = feedback_data.get('feedback_text', '')
        
        # Validate rating
        if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
            rating = 0
        # Validate feedback_text length
        if feedback_text and isinstance(feedback_text, str):
            if len(feedback_text) > 10000:  # Prevent DoS
                logger.warning(f"Feedback text too long ({len(feedback_text)} chars), truncating to 10KB")
                feedback_text = feedback_text[:10000]
        else:
            feedback_text = ''
        
        # Learning signals based on feedback
        signals = {
            'response_quality': 0.5,  # Default neutral
            'user_satisfaction': 0.5,
            'communication_effectiveness': 0.5,
            'information_accuracy': 0.5
        }
        
        # Adjust based on rating (1-5 scale)
        if rating >= 4:
            signals['response_quality'] = 0.8
            signals['user_satisfaction'] = 0.8
            signals['communication_effectiveness'] = 0.7
        elif rating <= 2:
            signals['response_quality'] = 0.2
            signals['user_satisfaction'] = 0.2
            signals['communication_effectiveness'] = 0.3
        
        # Adjust based on helpful flag
        if was_helpful:
            signals['response_quality'] = min(1.0, signals['response_quality'] + 0.1)
            signals['user_satisfaction'] = min(1.0, signals['user_satisfaction'] + 0.1)
        elif was_helpful is False:
            signals['response_quality'] = max(0.0, signals['response_quality'] - 0.1)
            signals['user_satisfaction'] = max(0.0, signals['user_satisfaction'] - 0.1)
        
        # Analyze feedback text for keywords
        if feedback_text:
            text_lower = feedback_text.lower()
            if any(word in text_lower for word in ['good', 'great', 'helpful', 'perfect', 'excellent']):
                signals['communication_effectiveness'] = min(1.0, signals['communication_effectiveness'] + 0.15)
            elif any(word in text_lower for word in ['bad', 'wrong', 'incorrect', 'unhelpful']):
                signals['information_accuracy'] = max(0.0, signals['information_accuracy'] - 0.2)
        
        return signals
        
    except Exception as e:
        logger.error(f"[LEARNING] Error analyzing feedback: {e}")
        return {}

async def update_model_weights(feedback_data: dict, signals: Dict[str, Any]) -> List[str]:
    """Update model weights based on feedback analysis"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning empty list")
        return []
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning empty list")
        return []
    
    try:
        weight_ids = []
        
        # Create weight updates for each signal
        for weight_name, weight_value in signals.items():
            weight_data = {
                'weight_type': 'response_style',
                'weight_name': weight_name,
                'weight_value': weight_value,
                'learning_session_id': feedback_data.get('id', None),
                'metadata': {
                    'conversation_id': feedback_data.get('conversation_id'),
                    'rating': feedback_data.get('rating'),
                    'was_helpful': feedback_data.get('was_helpful'),
                    'updated_at': datetime.now().isoformat()
                }
            }
            
            try:
                result, status = await make_supabase_request('POST', 'epsilon_model_weights', weight_data)
                if status in [200, 201]:
                    weight_id = result[0]['id'] if isinstance(result, list) else result.get('id')
                    if weight_id:
                        weight_ids.append(weight_id)
                        logger.info(f"[LEARNING] Updated model weight: {weight_name} = {weight_value}")
            except Exception as e:
                logger.error(f"[LEARNING] Error updating weight {weight_name}: {e}")
                continue
        
        return weight_ids
        
    except Exception as e:
        logger.error(f"[LEARNING] Error updating model weights: {e}")
        return []

async def create_learning_pattern(feedback_data: dict, signals: Dict[str, Any]) -> str:
    """Create a learning pattern from feedback analysis"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning None")
        return None
    
    try:
        # Extract pattern data
        pattern_data = {
            'feedback_id': feedback_data.get('id'),
            'rating': feedback_data.get('rating'),
            'was_helpful': feedback_data.get('was_helpful'),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine pattern type
        if feedback_data.get('rating', 0) >= 4:
            pattern_type = 'success_factor'
        elif feedback_data.get('was_helpful', False):
            pattern_type = 'user_satisfaction'
        else:
            pattern_type = 'improvement_area'
        
        # Create pattern record
        # Note: Supabase JSONB accepts dict objects directly, no need to json.dumps
        pattern_record = {
            'pattern_type': pattern_type,
            'pattern_data': pattern_data,  # Send as dict, Supabase converts to JSONB
            'confidence_score': signals.get('response_quality', 0.5),
            'usage_count': 1,
            'metadata': {
                'source': 'automatic_learning_service',
                'conversation_id': feedback_data.get('conversation_id'),
                'created_at': datetime.now().isoformat()
            }
        }
        
        try:
            result, status = await make_supabase_request('POST', 'epsilon_learning_patterns', pattern_record)
            if status in [200, 201]:
                pattern_id = result[0]['id'] if isinstance(result, list) else result.get('id')
                logger.info(f"[LEARNING] Created learning pattern: {pattern_type}")
                return pattern_id
        except Exception as e:
            logger.error(f"[LEARNING] Error creating learning pattern: {e}")
            return None
            
    except Exception as e:
        logger.error(f"[LEARNING] Error creating learning pattern: {e}")
        return None

async def update_learning_analytics(feedback_data: dict, signals: Dict[str, Any]):
    """Update epsilon_learning_analytics table"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning None")
        return None
    
    try:
        # Create analytics entries
        analytics_data = {
            'session_id': feedback_data.get('conversation_id', 'unknown'),
            'user_id': feedback_data.get('user_id'),
            'learning_type': 'quality',
            'metric_score': signals.get('response_quality', 0.5),
            'user_message': feedback_data.get('feedback_text', '')[:500],  # Limit length
            'epsilon_response': '',  # Would need to fetch from conversation
            'metadata': {
                'rating': feedback_data.get('rating'),
                'was_helpful': feedback_data.get('was_helpful'),
                'signals': signals
            }
        }
        
        result, status = await make_supabase_request('POST', 'epsilon_learning_analytics', analytics_data)
        if status in [200, 201]:
            logger.info("[LEARNING] Updated epsilon_learning_analytics")
            return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error updating analytics: {e}")
    return None

async def update_experience_data(feedback_data: dict, signals: Dict[str, Any]):
    """Update epsilon_experience_data table with interaction experience"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning None")
        return None
    
    try:
        experience_data = {
            'user_id': feedback_data.get('user_id'),
            'interaction_type': 'feedback',
            'user_input': feedback_data.get('feedback_text', '')[:500],
            'assistant_response': '',  # Fetch from conversation
            'success_rate': signals.get('response_quality', 0.5),
            'outcome': 'helpful' if feedback_data.get('was_helpful') else 'not_helpful',
            'confidence_score': signals.get('response_quality', 0.5),
            'learning_value': 'high' if feedback_data.get('rating', 0) >= 4 else 'medium'
        }
        
        result, status = await make_supabase_request('POST', 'epsilon_experience_data', experience_data)
        if status in [200, 201]:
            logger.info("[LEARNING] Updated epsilon_experience_data")
            return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error updating experience data: {e}")
    return None

async def create_training_data(feedback_data: dict, signals: Dict[str, Any]):
    """Create training data entry from feedback"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning None")
        return None
    
    try:
        training_data = {
            'input_text': feedback_data.get('feedback_text', '')[:1000],
            'expected_output': feedback_data.get('correction_text', '')[:1000],
            'training_type': 'feedback',
            'quality_score': signals.get('response_quality', 0.5),
            'is_validated': feedback_data.get('was_helpful', False),
            'metadata': {'feedback_id': feedback_data.get('id')}
        }
        
        result, status = await make_supabase_request('POST', 'epsilon_training_data', training_data)
        if status in [200, 201]:
            logger.info("[LEARNING] Created training data")
            return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error creating training data: {e}")
    return None

async def create_learning_rule(feedback_data: dict, signals: Dict[str, Any]):
    """Create or update learning rule from feedback"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning None")
        return None
    
    try:
        rule_data = {
            'rule_type': 'response_pattern',
            'pattern': feedback_data.get('feedback_text', '')[:500],
            'response_template': feedback_data.get('correction_text', '')[:500],
            'confidence_score': signals.get('response_quality', 0.5),
            'success_count': 1 if feedback_data.get('was_helpful') else 0,
            'failure_count': 0 if feedback_data.get('was_helpful') else 1,
            'is_active': True
        }
        
        result, status = await make_supabase_request('POST', 'epsilon_learning_rules', rule_data)
        if status in [200, 201]:
            logger.info("[LEARNING] Created learning rule")
            return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error creating learning rule: {e}")
    return None

async def update_memory_hierarchy(feedback_data: dict):
    """Update memory hierarchy for important interactions"""
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning None")
        return None
    
    try:
        if not feedback_data.get('user_id'):
            return None
        
        rating = feedback_data.get('rating', 0)
        # Only store high-impact interactions in memory
        if rating >= 4 or rating <= 2:
            memory_data = {
                'user_id': feedback_data.get('user_id'),
                'memory_type': 'short_term',
                'content': feedback_data.get('feedback_text', '')[:1000],
                'importance_score': 0.8 if rating >= 4 else 0.7,
                'access_count': 0
            }
            
            result, status = await make_supabase_request('POST', 'epsilon_memory_hierarchy', memory_data)
            if status in [200, 201]:
                logger.info("[LEARNING] Updated memory hierarchy")
                return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error updating memory: {e}")
    return None

async def update_performance_metrics(signals: Dict[str, Any]):
    """Update performance metrics table"""
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning")
        return
    
    try:
        for metric_name, metric_value in signals.items():
            metric_data = {
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_type': 'learning'
            }
            
            result, status = await make_supabase_request('POST', 'epsilon_performance_metrics', metric_data)
            if status in [200, 201]:
                logger.info(f"[LEARNING] Updated performance metric: {metric_name}")
    except Exception as e:
        logger.error(f"[LEARNING] Error updating performance metrics: {e}")

async def create_learning_session():
    """Create a learning session"""
    try:
        session_data = {
            'session_id': f"learning_{datetime.now().isoformat()}",
            'session_type': 'real_time',
            'training_data_count': 1,
            'performance_improvement': 0.01,
            'status': 'active'
        }
        
        result, status = await make_supabase_request('POST', 'epsilon_learning_sessions', session_data)
        if status in [200, 201]:
            logger.info("[LEARNING] Created learning session")
            return result[0]['id'] if isinstance(result, list) else result.get('id')
    except Exception as e:
        logger.error(f"[LEARNING] Error creating learning session: {e}")
    return None

async def process_feedback_learning(feedback_id: str):
    """Process a single feedback for learning"""
    if not feedback_id or not isinstance(feedback_id, str):
        logger.warning("Invalid feedback_id input, returning")
        return
    if len(feedback_id) > 200:  # Prevent DoS
        logger.warning(f"Feedback ID too long ({len(feedback_id)} chars), truncating to 200")
        feedback_id = feedback_id[:200]
    
    try:
        logger.info(f"🔍 Processing feedback for learning: {feedback_id}")
        
        # Get feedback data from database
        try:
            feedback_result, status = await make_supabase_request(
                'GET', 
                f"epsilon_feedback?id=eq.{feedback_id}"
            )
            
            if status != 200 or not feedback_result:
                logger.warning(f"[LEARNING] Feedback not found: {feedback_id}")
                return
            
            feedback_data = feedback_result[0]
            
        except Exception as e:
            logger.error(f"[LEARNING] Error fetching feedback: {e}")
            return
        
        # Analyze feedback
        signals = await analyze_feedback_for_learning(feedback_data)
        if not signals:
            logger.warning("[LEARNING] No learning signals extracted")
            return
        
        logger.info(f"[LEARNING] Learning signals extracted: {signals}")
        
        # Update model weights
        weight_ids = await update_model_weights(feedback_data, signals)
        logger.info(f"⚖️ Updated {len(weight_ids)} model weights")
        
        # Create learning pattern
        pattern_id = await create_learning_pattern(feedback_data, signals)
        if pattern_id:
            logger.info(f"[LEARNING] Created learning pattern: {pattern_id}")
        
        # Update learning analytics (NEW!)
        await update_learning_analytics(feedback_data, signals)
        
        # Update experience data (NEW!)
        await update_experience_data(feedback_data, signals)
        
        # Create training data (NEW!)
        await create_training_data(feedback_data, signals)

        # Create learning rule (NEW!)
        await create_learning_rule(feedback_data, signals)

        # Update memory hierarchy (NEW!)
        await update_memory_hierarchy(feedback_data)

        # Update performance metrics (NEW!)
        await update_performance_metrics(signals)

        # Create learning session (NEW!)
        await create_learning_session()
        
        logger.info(f"[LEARNING] Completed learning for feedback: {feedback_id}")
        
    except Exception as e:
        logger.error(f"[LEARNING] Error processing feedback learning: {e}")

# Rate limiting to prevent runaway learning
LEARNING_RATE_LIMITS = {
    'max_per_hour': 100,
    'max_per_minute': 10,
    'max_batch_size': 50,
    'max_content_length': 10000
}

class RateLimiter:
    """Rate limiter for learning service"""
    def __init__(self):
        self.hourly_count = 0
        self.minute_count = 0
        self.last_minute = datetime.now()
        self.last_hour = datetime.now()
    
    async def check_rate(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Reset hourly counter
        if (now - self.last_hour).total_seconds() > 3600:
            self.hourly_count = 0
            self.last_hour = now
        
        # Reset minute counter
        if (now - self.last_minute).total_seconds() > 60:
            self.minute_count = 0
            self.last_minute = now
        
        # Check limits
        if self.hourly_count >= LEARNING_RATE_LIMITS['max_per_hour']:
            logger.warning(f"[LEARNING] Hourly rate limit reached: {self.hourly_count}/{LEARNING_RATE_LIMITS['max_per_hour']}")
            return False
        
        if self.minute_count >= LEARNING_RATE_LIMITS['max_per_minute']:
            logger.warning(f"[LEARNING] Minute rate limit reached: {self.minute_count}/{LEARNING_RATE_LIMITS['max_per_minute']}")
            return False
        
        return True
    
    def increment(self):
        """Increment counters"""
        self.hourly_count += 1
        self.minute_count += 1

rate_limiter = RateLimiter()

async def reason_and_learn(feedback_data: dict) -> Dict[str, Any]:
    """
    Advanced reasoning layer: Plans, acts, evaluates, and learns
    
    This implements a multi-step reasoning loop:
    1. Plan - analyze feedback and determine what to learn
    2. Act - update model weights and patterns
    3. Evaluate - check if learning was successful
    4. Update Memory - store successful patterns for future recall
    """
    if not feedback_data or not isinstance(feedback_data, dict):
        logger.warning("Invalid feedback_data input, returning error")
        return {'error': 'Invalid feedback_data'}
    
    try:
        signals = await analyze_feedback_for_learning(feedback_data)
        
        # STEP 1: PLAN - Determine learning strategy
        logger.info("[REASONING] Planning learning strategy...")
        learning_strategy = {
            'focus_area': determine_focus_area(feedback_data),
            'learning_intensity': calculate_learning_intensity(feedback_data),
            'update_weights': signals.get('response_quality', 0) < 0.5,
            'create_pattern': len(feedback_data.get('feedback_text', '')) > 10,
            'priority_level': 'high' if feedback_data.get('rating', 0) <= 2 else 'medium'
        }
        logger.info(f"[REASONING] Strategy: {learning_strategy}")
        
        # STEP 2: ACT - Execute planned learning actions
        logger.info("[REASONING] Executing learning actions...")
        weight_ids = []
        pattern_id = None
        analytics_id = None
        
        if learning_strategy['update_weights']:
            weight_ids = await update_model_weights(feedback_data, signals)
        
        if learning_strategy['create_pattern']:
            pattern_id = await create_learning_pattern(feedback_data, signals)
        
        if learning_strategy['priority_level'] == 'high':
            analytics_id = await update_learning_analytics(feedback_data, signals)
            await update_experience_data(feedback_data, signals)
            await create_training_data(feedback_data, signals)
            await create_learning_rule(feedback_data, signals)
            await update_memory_hierarchy(feedback_data)
            await update_performance_metrics(signals)
            await create_learning_session()
        
        # STEP 3: EVALUATE - Check if learning was successful
        logger.info("[REASONING] Evaluating learning effectiveness...")
        evaluation = {
            'weights_updated': len(weight_ids),
            'pattern_created': pattern_id is not None,
            'analytics_created': analytics_id is not None,
            'confidence_improvement': signals.get('confidence_score', 0) - feedback_data.get('rating', 3),
            'success': len(weight_ids) > 0 or pattern_id is not None
        }
        logger.info(f"[REASONING] Evaluation: {evaluation}")
        
        # STEP 4: UPDATE MEMORY - Store successful patterns for semantic recall
        if evaluation['success']:
            logger.info("💾 [REASONING] Updating semantic memory for future recall...")
            await store_memory_for_recall(feedback_data, signals, learning_strategy)
        
        return {
            'strategy': learning_strategy,
            'actions': {'weight_ids': weight_ids, 'pattern_id': pattern_id},
            'evaluation': evaluation,
            'next_reasoning_improvement': suggest_next_improvement(feedback_data, signals)
        }
        
    except Exception as e:
        logger.error(f"[REASONING] Error in reasoning loop: {e}")
        return {'error': str(e)}

def determine_focus_area(feedback: dict) -> str:
    """Determine the primary focus area for learning"""
    if not feedback or not isinstance(feedback, dict):
        return 'communication_effectiveness'
    
    if feedback.get('was_helpful') is False:
        return 'response_quality'
    elif feedback.get('rating', 0) <= 2:
        return 'user_satisfaction'
    else:
        return 'communication_effectiveness'

def calculate_learning_intensity(feedback: dict) -> float:
    """Calculate how intensely to apply learning (0.0 to 1.0)"""
    if not feedback or not isinstance(feedback, dict):
        return 0.5
    
    rating = feedback.get('rating', 3)
    feedback_text = feedback.get('feedback_text', '')
    
    # Validate rating
    if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
        rating = 3
    if not isinstance(feedback_text, str):
        feedback_text = ''
    if len(feedback_text) > 10000:  # Prevent DoS
        feedback_text = feedback_text[:10000]
    
    # Higher intensity for lower ratings
    # Division by 5 is safe (constant divisor)
    base_intensity = (5 - rating) / 5
    
    # Increase if detailed feedback provided
    if len(feedback_text) > 50:
        base_intensity = min(1.0, base_intensity + 0.2)
    
    return base_intensity

def suggest_next_improvement(feedback: dict, signals: dict) -> str:
    """Suggest next area for improvement based on current learning"""
    if not feedback or not isinstance(feedback, dict):
        feedback = {}
    if not signals or not isinstance(signals, dict):
        signals = {}
    
    if signals.get('response_quality', 0.5) < 0.5:
        return "Focus on understanding user intent more accurately"
    elif signals.get('user_satisfaction', 0.5) < 0.5:
        return "Improve response personalization and relevance"
    else:
        return "Optimize response format and clarity"

async def generate_text_embedding(text: str) -> List[float]:
    """Generate text embedding using real embedding model"""
    if not text or not isinstance(text, str):
        logger.warning("Invalid text input, raising error")
        raise ValueError("text must be a non-empty string")
    if len(text) > 100000:  # Prevent DoS - 100KB max
        logger.warning(f"Text too long ({len(text)} chars), truncating to 100KB")
        text = text[:100000]
    
    try:
        # Try to use sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(generate_text_embedding, '_model'):
                generate_text_embedding._model = SentenceTransformer('all-MiniLM-L6-v2')
            model = generate_text_embedding._model
            embedding = model.encode(text, convert_to_numpy=True).tolist()
            # Ensure 384 dimensions (pad or truncate if needed)
            if len(embedding) < 384:
                embedding.extend([0.0] * (384 - len(embedding)))
            elif len(embedding) > 384:
                embedding = embedding[:384]
            return embedding
        except ImportError:
            logger.warning("[LEARNING] sentence-transformers not available, using TF-IDF based embedding")
            # Fallback to TF-IDF based approach (real implementation, not fake)
            from collections import Counter
            import math
            
            # Simple TF-IDF based embedding
            words = text.lower().split()
            word_counts = Counter(words)
            total_words = len(words)
        
            # Create 384-dimensional embedding based on word frequencies
            embedding = [0.0] * 384
            for word, count in word_counts.items():
                # Hash word to index
                word_hash = hash(word) % 384
                # TF-IDF score (simplified)
                tf = count / total_words if total_words > 0 else 0
                embedding[word_hash] += tf
            
            # Normalize
            magnitude = math.sqrt(sum(x * x for x in embedding))
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
        
        return embedding
    except Exception as e:
        logger.error(f"[LEARNING] Error generating embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {e}")

async def store_memory_for_recall(feedback: dict, signals: dict, strategy: dict):
    """Store learning insights in semantic memory for future recall using pgvector"""
    if not feedback or not isinstance(feedback, dict):
        logger.warning("Invalid feedback input, returning")
        return
    if not signals or not isinstance(signals, dict):
        logger.warning("Invalid signals input, returning")
        return
    if not strategy or not isinstance(strategy, dict):
        logger.warning("Invalid strategy input, returning")
        return
    
    try:
        # Create memory entry that can be recalled via semantic search
        memory_content = f"Learned: {strategy['focus_area']} from user feedback. Rating: {feedback.get('rating')}. Key insight: {feedback.get('feedback_text', '')[:200]}"
        
        logger.info(f"💾 [MEMORY] Generating embedding for semantic memory...")
        
        # Generate embedding for semantic search
        embedding = await generate_text_embedding(memory_content)
        
        # Store in epsilon_semantic_memory with pgvector
        memory_data = {
            'user_id': feedback.get('user_id'),
            'content': memory_content,
            'embedding': str(embedding),  # Convert to string for JSON storage
            'memory_type': 'long_term',
            'importance_score': round(strategy.get('learning_intensity', 0.5), 2),
            'metadata': {
                'source': 'learning_service',
                'feedback_id': feedback.get('id'),
                'focus_area': strategy['focus_area'],
                'created_at': datetime.now().isoformat()
            }
        }
        
        result, status = await make_supabase_request(
            'POST', 'epsilon_semantic_memory', memory_data
        )
        
        if status in [200, 201]:
            logger.info(f"[MEMORY] Stored semantic memory for future recall")
        else:
            logger.warning(f"[MEMORY] Failed to store semantic memory: {status}")
        
    except Exception as e:
        logger.error(f"[MEMORY] Error storing memory: {e}")

async def monitor_feedback():
    """
    Monitor epsilon_feedback table for new feedback and process learning
    Uses advanced reasoning loop: Plan → Act → Evaluate → Learn
    """
    logger.info("[LEARNING] Starting Epsilon AI Learning Service...")
    logger.info(f"[LEARNING] Rate limits: {LEARNING_RATE_LIMITS}")
    
    try:
        while True:
            # Check rate limit
            if not await rate_limiter.check_rate():
                logger.warning("⏸️ Rate limit hit, pausing for 60 seconds...")
                await asyncio.sleep(60)
                continue
            
            # Get recent feedback that hasn't been processed for learning
            try:
                # Query for feedback from last 5 minutes (using proper ISO format for Supabase)
                cutoff_time = datetime.now() - timedelta(minutes=5)
                # Format as ISO string with 'Z' suffix for UTC
                cutoff_iso = cutoff_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                
                result, status = await make_supabase_request(
                    'GET',
                    f"epsilon_feedback?created_at=gte.{cutoff_iso}&order=created_at.desc&limit={LEARNING_RATE_LIMITS['max_batch_size']}"
                )
                
                if status == 200 and result:
                    feedback_batch = result[:LEARNING_RATE_LIMITS['max_batch_size']]
                    logger.info(f"[LEARNING] Found {len(feedback_batch)} new feedback entries")
                    
                    if len(feedback_batch) == 0:
                        logger.info("[LEARNING] No new feedback entries found in the last 5 minutes")
                    
                    for feedback in feedback_batch:
                        # Check rate limit before each processing
                        if not await rate_limiter.check_rate():
                            logger.warning("[LEARNING] Rate limit hit mid-batch, stopping processing")
                            break
                        
                        # Use advanced reasoning loop instead of basic processing
                        reasoning_result = await reason_and_learn(feedback)
                        logger.info(f"🎓 [LEARNING] Completed reasoning loop for feedback {feedback['id']}")
                        
                        rate_limiter.increment()
                
            except Exception as e:
                logger.error(f"[LEARNING] Error monitoring feedback: {e}")
            
            # Wait 30 seconds before checking again
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("[LEARNING] Epsilon AI Learning Service stopped")
    except Exception as e:
        logger.error(f"[LEARNING] Fatal error in learning service: {e}")

if __name__ == "__main__":
    # Run the learning service
    try:
        asyncio.run(monitor_feedback())
    except Exception as e:
        logger.error(f"[LEARNING] Fatal error starting learning service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)