/**
 * Server-Side Learning Engine Service
 * ====================================
 * CRITICAL: All learning, RAG decisions, and heuristics run server-side
 * Client is thin - only sends requests and displays responses
 * 
 * Security:
 * - No learning logic exposed to client
 * - PII redaction before storage
 * - Rate limiting and budget guardrails
 * - All API keys server-side only
 */

const { createClient } = require('@supabase/supabase-js');
const crypto = require('crypto');

class ServerLearningService {
  constructor() {
    // Validate environment variables
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_KEY) {
      throw new Error('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set');
    }
    
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );
    
    // Rate limiting (per user)
    this.userRateLimits = new Map();
    this.maxRequestsPerMinute = 30;
    this.maxRequestsPerHour = 500;
    
    // Budget guardrails
    this.dailyCostLimit = 100; // $100/day max
    this.currentDailyCost = 0;
    this.costResetTime = this.getNextMidnight();
    
    // PII patterns for redaction
    this.piiPatterns = {
      email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
      phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b/g,
      ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
      creditCard: /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g,
      ipAddress: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g
    };
  }

  /**
   * Redact PII from text before storage
   */
  redactPII(text) {
    if (!text || typeof text !== 'string') {
      return { redacted: text || '', redactions: [] };
    }
    
    let redacted = text;
    const redactions = [];
    
    // Redact emails
    redacted = redacted.replace(this.piiPatterns.email, (match) => {
      const hash = crypto.createHash('sha256').update(match).digest('hex').substring(0, 8);
      redactions.push({ type: 'email', original: match, hash });
      return `[EMAIL:${hash}]`;
    });
    
    // Redact phone numbers
    redacted = redacted.replace(this.piiPatterns.phone, (match) => {
      const hash = crypto.createHash('sha256').update(match).digest('hex').substring(0, 8);
      redactions.push({ type: 'phone', original: match, hash });
      return `[PHONE:${hash}]`;
    });
    
    // Redact SSN
    redacted = redacted.replace(this.piiPatterns.ssn, (match) => {
      const hash = crypto.createHash('sha256').update(match).digest('hex').substring(0, 8);
      redactions.push({ type: 'ssn', original: match, hash });
      return `[SSN:${hash}]`;
    });
    
    // Redact credit cards
    redacted = redacted.replace(this.piiPatterns.creditCard, (match) => {
      const hash = crypto.createHash('sha256').update(match).digest('hex').substring(0, 8);
      redactions.push({ type: 'credit_card', original: match, hash });
      return `[CARD:${hash}]`;
    });
    
    // Redact IP addresses (keep for analytics but hash)
    redacted = redacted.replace(this.piiPatterns.ipAddress, (match) => {
      const hash = crypto.createHash('sha256').update(match).digest('hex').substring(0, 8);
      return `[IP:${hash}]`;
    });
    
    return { redacted, redactions };
  }

  /**
   * Hash user ID for privacy
   */
  hashUserId(userId) {
    if (!userId) return null;
    return crypto.createHash('sha256').update(userId).digest('hex');
  }

  /**
   * Check rate limits
   */
  checkRateLimit(userId) {
    if (!userId || typeof userId !== 'string') {
      return { allowed: false, reason: 'invalid_user_id' };
    }
    
    const now = Date.now();
    const userLimits = this.userRateLimits.get(userId) || {
      requests: [],
      hourlyRequests: []
    };
    
    // Clean old requests
    userLimits.requests = userLimits.requests.filter(t => now - t < 60000); // Last minute
    userLimits.hourlyRequests = userLimits.hourlyRequests.filter(t => now - t < 3600000); // Last hour
    
    // Check limits
    if (userLimits.requests.length >= this.maxRequestsPerMinute) {
      return { allowed: false, reason: 'rate_limit_minute' };
    }
    
    if (userLimits.hourlyRequests.length >= this.maxRequestsPerHour) {
      return { allowed: false, reason: 'rate_limit_hour' };
    }
    
    // Record request
    userLimits.requests.push(now);
    userLimits.hourlyRequests.push(now);
    this.userRateLimits.set(userId, userLimits);
    
    return { allowed: true };
  }

  /**
   * Check budget guardrails
   */
  checkBudget(cost) {
    if (typeof cost !== 'number' || cost < 0 || !isFinite(cost)) {
      return { allowed: false, reason: 'invalid_cost' };
    }
    
    const now = Date.now();
    
    // Reset daily cost at midnight
    if (now >= this.costResetTime) {
      this.currentDailyCost = 0;
      this.costResetTime = this.getNextMidnight();
    }
    
    if (this.currentDailyCost + cost > this.dailyCostLimit) {
      return { allowed: false, reason: 'budget_exceeded', current: this.currentDailyCost, limit: this.dailyCostLimit };
    }
    
    this.currentDailyCost += cost;
    return { allowed: true, remaining: this.dailyCostLimit - this.currentDailyCost };
  }

  getNextMidnight() {
    const tomorrow = new Date();
    tomorrow.setHours(24, 0, 0, 0);
    return tomorrow.getTime();
  }

  /**
   * Store conversation (server-side, with PII redaction)
   */
  async storeConversation(userId, userMessage, epsilonResponse, responseTime = 0, contextData = {}) {
s
    if (!userId || typeof userId !== 'string') {
      console.error('[SERVER LEARNING] Invalid userId');
      return null;
    }
    if (!userMessage || typeof userMessage !== 'string') {
      console.error('[SERVER LEARNING] Invalid userMessage');
      return null;
    }
    if (!epsilonResponse || typeof epsilonResponse !== 'string') {
      console.error('[SERVER LEARNING] Invalid epsilonResponse');
      return null;
    }
    if (typeof responseTime !== 'number' || responseTime < 0 || !isFinite(responseTime)) {
      responseTime = 0;
    }
    if (!contextData || typeof contextData !== 'object' || Array.isArray(contextData)) {
      contextData = {};
    }
    
    try {
      // Redact PII from messages
      const userMessageRedacted = this.redactPII(userMessage);
      const epsilonResponseRedacted = this.redactPII(epsilonResponse);
      
      // Hash user ID
      const hashedUserId = this.hashUserId(userId);
      
      // Prepare conversation data
      const conversationData = {
        session_id: `session_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`,
        user_id: hashedUserId, // Hashed, not raw
        user_message: userMessageRedacted.redacted,
        epsilon_response: epsilonResponseRedacted.redacted,
        response_time_ms: responseTime,
        context_data: {
          timestamp: new Date().toISOString(),
          ...contextData
        },
        learning_metadata: {
          model_version: '1.0.0',
          response_style: this.detectResponseStyle(epsilonResponse),
          topic_category: this.categorizeTopic(userMessage),
          user_intent: this.detectUserIntent(userMessage)
        },
        pii_redactions: {
          user_message: userMessageRedacted.redactions,
          epsilon_response: epsilonResponseRedacted.redactions
        }
      };

      const { data, error } = await this.supabase
        .from('epsilon_conversations')
        .insert(conversationData)
        .select()
        .limit(1).maybeSingle();

      if (error) {
        console.error('[SERVER LEARNING] Failed to store conversation:', error);
        return null;
      }

      if (global.epsilonAICore && global.epsilonAICore.isTrained) {
        try {
          const feedback = contextData.feedback || null;
          global.epsilonAICore.learnFromConversation(
            userMessageRedacted.redacted,
            epsilonResponseRedacted.redacted,
            feedback
          );
        } catch (error) {
          console.warn('[LEARNING] Failed to learn from conversation:', error.message);
        }
      }

      // Note: messages table does NOT have user_id column - user_id is in conversations table
      try {
        const { data: existingConv, error: convError } = await this.supabase
          .from('conversations')
          .select('id')
          .eq('session_id', conversationData.session_id)
          .limit(1).maybeSingle();
        
        let mainConversationId = null;
        if (convError || !existingConv) {
          const { data: newConv, error: newConvError } = await this.supabase
            .from('conversations')
            .insert([{
              user_id: hashedUserId || null,
              session_id: conversationData.session_id,
              conversation_name: null,
              folder_id: null,
              is_deleted: false,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (!newConvError && newConv) {
            mainConversationId = newConv.id;
          }
        } else {
          mainConversationId = existingConv.id;
        }
        
        // Store messages if we have a conversation ID
        if (mainConversationId) {
          await this.supabase
            .from('messages')
            .insert({
              conversation_id: mainConversationId,
              role: 'user',
              text: userMessageRedacted.redacted
            });

          await this.supabase
            .from('messages')
            .insert({
              conversation_id: mainConversationId,
              role: 'epsilon',
              text: epsilonResponseRedacted.redacted
            });
        }
      } catch (mainTableError) {
        console.warn('[SERVER LEARNING] Failed to store in main tables (non-critical):', mainTableError.message);
      }

      return data.id;
    } catch (error) {
      console.error('[SERVER LEARNING] Error storing conversation:', error);
      return null;
    }
  }

  /**
   * Store feedback (server-side, with PII redaction)
   */
  async storeFeedback(userId, conversationId, rating = null, wasHelpful = null, feedbackText = null) {
s
    if (!userId || typeof userId !== 'string') {
      console.error('[SERVER LEARNING] Invalid userId');
      return null;
    }
    if (!conversationId || typeof conversationId !== 'string') {
      console.error('[SERVER LEARNING] Invalid conversationId');
      return null;
    }
    if (rating !== null && (typeof rating !== 'number' || rating < 1 || rating > 5 || !isFinite(rating))) {
      rating = null;
    }
    if (wasHelpful !== null && typeof wasHelpful !== 'boolean') {
      wasHelpful = null;
    }
    if (feedbackText !== null && typeof feedbackText !== 'string') {
      feedbackText = null;
    }
    
    try {
      // Redact PII from feedback
      const feedbackRedacted = feedbackText ? this.redactPII(feedbackText) : { redacted: null, redactions: [] };
      
      // Hash user ID
      const hashedUserId = this.hashUserId(userId);
      
      const feedbackData = {
        conversation_id: conversationId,
        user_id: hashedUserId,
        rating: rating,
        was_helpful: wasHelpful,
        feedback_text: feedbackRedacted.redacted,
        pii_redactions: feedbackRedacted.redactions
      };

      const { data, error } = await this.supabase
        .from('epsilon_feedback')
        .insert(feedbackData)
        .select()
        .limit(1).maybeSingle();

      if (error) {
        console.error('[SERVER LEARNING] Failed to store feedback:', error);
        return null;
      }

      return data.id;
    } catch (error) {
      console.error('[SERVER LEARNING] Error storing feedback:', error);
      return null;
    }
  }

  /**
   * Helper: Detect response style
   */
  detectResponseStyle(response) {
    if (!response || typeof response !== 'string') {
      return 'conversational';
    }
    
    const lower = response.toLowerCase();
    if (lower.includes('i can') || lower.includes('i will')) return 'helpful';
    if (lower.includes('let me') || lower.includes('i\'ll')) return 'proactive';
    if (lower.includes('based on') || lower.includes('according to')) return 'informative';
    return 'conversational';
  }

  /**
   * Helper: Categorize topic
   */
  categorizeTopic(message) {
    if (!message || typeof message !== 'string') {
      return 'general';
    }
    
    const lower = message.toLowerCase();
    if (lower.includes('automation') || lower.includes('workflow')) return 'automation';
    if (lower.includes('website') || lower.includes('web')) return 'web_development';
    if (lower.includes('ai') || lower.includes('intelligence')) return 'ai_strategy';
    if (lower.includes('pricing') || lower.includes('cost')) return 'pricing';
    return 'general';
  }

  /**
   * Helper: Detect user intent
   */
  detectUserIntent(message) {
    if (!message || typeof message !== 'string') {
      return 'general';
    }
    
    const lower = message.toLowerCase();
    if (/^(hi|hello|hey)/.test(lower)) return 'greeting';
    if (lower.includes('how') || lower.includes('what') || lower.includes('why')) return 'question';
    if (lower.includes('help') || lower.includes('assist')) return 'support';
    if (lower.includes('buy') || lower.includes('purchase') || lower.includes('sign up')) return 'purchase';
    return 'general';
  }
}

module.exports = ServerLearningService;


