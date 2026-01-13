/**
 * Epsilon AI Conversational Learning System
 * ===================================
 * Implements:
 * - Reinforcement learning for tone
 * - Reward models for accuracy
 * - Conversational pattern mining
 * - Sales objection classification
 * - Emotional state detection
 * 
 * NO external AI dependencies
 */

class EpsilonConversationalLearning {
  constructor() {
    // Reward model weights
    this.rewardWeights = {
      accuracy: 0.4,
      helpfulness: 0.3,
      tone: 0.2,
      engagement: 0.1
    };
    
    // Pattern storage
    this.conversationPatterns = new Map();
    this.salesObjections = new Map();
    this.emotionalStates = new Map();
    
    // Learning parameters
    this.learningRate = 0.1;
    this.patternThreshold = 0.7;
  }

  /**
   * Reinforcement learning: Update based on reward
   */
  updateWithReward(response, reward, context = {}) {
    // Calculate reward components
    const rewardComponents = {
      accuracy: this._calculateAccuracyReward(response, context),
      helpfulness: this._calculateHelpfulnessReward(response, context),
      tone: this._calculateToneReward(response, context),
      engagement: this._calculateEngagementReward(response, context)
    };
    
    // Weighted total reward
    const totalReward = Object.keys(rewardComponents).reduce((sum, key) => {
      return sum + (rewardComponents[key] * this.rewardWeights[key]);
    }, 0);
    
    // Update patterns based on reward
    if (totalReward > 0.6) {
      // Positive reward - reinforce pattern
      this._reinforcePattern(context, response, totalReward);
    } else if (totalReward < 0.4) {
      // Negative reward - weaken pattern
      this._weakenPattern(context, response, totalReward);
    }
    
    return {
      totalReward,
      components: rewardComponents
    };
  }

  /**
   * Mine conversational patterns
   */
  minePatterns(conversations) {
    const patterns = new Map();
    
    for (const conv of conversations) {
      const { userMessage, epsilonResponse, outcome } = conv;
      
      // Extract patterns
      const userPattern = this._extractPattern(userMessage);
      const responsePattern = this._extractPattern(epsilonResponse);
      
      // Store pattern
      const patternKey = `${userPattern.type}_${responsePattern.type}`;
      
      if (!patterns.has(patternKey)) {
        patterns.set(patternKey, {
          count: 0,
          outcomes: [],
          avgReward: 0
        });
      }
      
      const pattern = patterns.get(patternKey);
      pattern.count++;
      pattern.outcomes.push(outcome || 'neutral');
      
      // Update average reward
      if (outcome && typeof outcome === 'number') {
        pattern.avgReward = (pattern.avgReward * (pattern.count - 1) + outcome) / pattern.count;
      }
    }
    
    // Store high-confidence patterns
    for (const [key, pattern] of patterns.entries()) {
      if (pattern.count >= 3 && pattern.avgReward > this.patternThreshold) {
        this.conversationPatterns.set(key, pattern);
      }
    }
    
    return Array.from(this.conversationPatterns.entries());
  }

  /**
   * Classify sales objections
   */
  classifySalesObjection(message) {
    const lower = message.toLowerCase();
    
    const objectionTypes = {
      price: ['expensive', 'cost', 'price', 'afford', 'cheaper', 'budget', 'too much'],
      time: ['busy', 'time', 'later', 'not now', 'wait', 'delay'],
      authority: ['boss', 'manager', 'decide', 'approval', 'permission'],
      need: ['need', 'necessary', 'want', 'interested', 'relevant'],
      trust: ['trust', 'believe', 'prove', 'guarantee', 'reliable', 'credible']
    };
    
    const scores = {};
    for (const [type, keywords] of Object.entries(objectionTypes)) {
      scores[type] = keywords.filter(kw => lower.includes(kw)).length;
    }
    
    const maxScore = Math.max(...Object.values(scores));
    if (maxScore === 0) return { type: 'none', confidence: 0 };
    
    const type = Object.keys(scores).find(key => scores[key] === maxScore);
    const confidence = maxScore / Math.max(...Object.values(Object.fromEntries(
      Object.entries(objectionTypes).map(([k]) => [k, objectionTypes[k].length])
    )));
    
    // Store objection pattern
    if (!this.salesObjections.has(type)) {
      this.salesObjections.set(type, []);
    }
    this.salesObjections.get(type).push({
      message,
      timestamp: Date.now()
    });
    
    return { type, confidence, scores };
  }

  /**
   * Detect emotional state
   */
  detectEmotionalState(message) {
    const lower = message.toLowerCase();
    
    const emotions = {
      positive: ['happy', 'excited', 'great', 'awesome', 'love', 'amazing', 'wonderful', 'fantastic'],
      negative: ['angry', 'frustrated', 'upset', 'disappointed', 'terrible', 'awful', 'hate', 'worst'],
      neutral: ['okay', 'fine', 'alright', 'sure', 'maybe'],
      urgent: ['urgent', 'asap', 'immediately', 'now', 'quick', 'hurry', 'emergency'],
      confused: ['confused', 'unclear', 'understand', 'explain', 'what', 'how', 'why'],
      satisfied: ['thanks', 'thank you', 'perfect', 'exactly', 'right', 'good', 'helpful']
    };
    
    const scores = {};
    for (const [emotion, keywords] of Object.entries(emotions)) {
      scores[emotion] = keywords.filter(kw => lower.includes(kw)).length;
    }
    
    const maxScore = Math.max(...Object.values(scores));
    if (maxScore === 0) return { state: 'neutral', confidence: 0.5 };
    
    const state = Object.keys(scores).find(key => scores[key] === maxScore);
    const confidence = Math.min(1.0, maxScore / 3);
    
    // Store emotional pattern
    if (!this.emotionalStates.has(state)) {
      this.emotionalStates.set(state, []);
    }
    this.emotionalStates.get(state).push({
      message,
      timestamp: Date.now()
    });
    
    return { state, confidence, scores };
  }

  /**
   * Get response strategy based on patterns
   */
  getResponseStrategy(userMessage, context = {}) {
    const objection = this.classifySalesObjection(userMessage);
    const emotion = this.detectEmotionalState(userMessage);
    
    // Check for matching patterns
    const userPattern = this._extractPattern(userMessage);
    const patternKey = `${userPattern.type}_*`;
    
    let strategy = {
      tone: 'professional',
      approach: 'informative',
      urgency: 'normal'
    };
    
    // Adjust based on objection
    if (objection.type !== 'none') {
      strategy.approach = 'address_objection';
      strategy.objectionType = objection.type;
    }
    
    // Adjust based on emotion
    if (emotion.state === 'negative') {
      strategy.tone = 'empathetic';
      strategy.approach = 'de_escalate';
    } else if (emotion.state === 'urgent') {
      strategy.urgency = 'high';
      strategy.approach = 'direct';
    } else if (emotion.state === 'confused') {
      strategy.approach = 'clarify';
      strategy.tone = 'patient';
    } else if (emotion.state === 'positive') {
      strategy.tone = 'enthusiastic';
    }
    
    // Check conversation patterns
    for (const [patternKey, pattern] of this.conversationPatterns.entries()) {
      if (pattern.avgReward > this.patternThreshold) {
        // Use successful pattern
        strategy.pattern = patternKey;
      }
    }
    
    return strategy;
  }

  /**
   * Extract pattern from text
   */
  _extractPattern(text) {
    const lower = text.toLowerCase();
    
    // Question pattern
    if (text.includes('?') || lower.startsWith('how') || lower.startsWith('what') || lower.startsWith('why')) {
      return { type: 'question', confidence: 0.8 };
    }
    
    // Request pattern
    if (lower.includes('can you') || lower.includes('please') || lower.includes('help')) {
      return { type: 'request', confidence: 0.8 };
    }
    
    // Statement pattern
    if (text.includes('.') && !text.includes('?')) {
      return { type: 'statement', confidence: 0.7 };
    }
    
    return { type: 'general', confidence: 0.5 };
  }

  /**
   * Calculate accuracy reward
   */
  _calculateAccuracyReward(response, context) {
    // Check if response addresses the question
    if (context.userMessage && response) {
      const userWords = new Set(context.userMessage.toLowerCase().split(/\s+/));
      const responseWords = new Set(response.toLowerCase().split(/\s+/));
      const overlap = [...userWords].filter(w => responseWords.has(w)).length;
      return Math.min(1.0, overlap / Math.max(userWords.size, 1));
    }
    return 0.5;
  }

  /**
   * Calculate helpfulness reward
   */
  _calculateHelpfulnessReward(response, context) {
    // Longer, detailed responses are generally more helpful
    const lengthScore = Math.min(1.0, response.length / 200);
    
    // Check for actionable content
    const hasActionWords = /(can|will|should|try|use|do|make|create|build)/i.test(response);
    const actionScore = hasActionWords ? 0.3 : 0;
    
    return lengthScore * 0.7 + actionScore;
  }

  /**
   * Calculate tone reward
   */
  _calculateToneReward(response, context) {
    const lower = response.toLowerCase();
    
    // Positive tone indicators
    const positiveWords = ['help', 'glad', 'happy', 'excited', 'great', 'wonderful'];
    const positiveCount = positiveWords.filter(w => lower.includes(w)).length;
    
    // Professional tone indicators
    const professionalWords = ['professional', 'expert', 'quality', 'reliable'];
    const professionalCount = professionalWords.filter(w => lower.includes(w)).length;
    
    return Math.min(1.0, (positiveCount * 0.3 + professionalCount * 0.2));
  }

  /**
   * Calculate engagement reward
   */
  _calculateEngagementReward(response, context) {
    // Questions in response increase engagement
    const hasQuestion = response.includes('?');
    const questionScore = hasQuestion ? 0.3 : 0;
    
    // Call to action
    const hasCTA = /(let me|I can|would you|can I)/i.test(response);
    const ctaScore = hasCTA ? 0.2 : 0;
    
    return questionScore + ctaScore;
  }

  /**
   * Reinforce successful pattern
   */
  _reinforcePattern(context, response, reward) {
    const pattern = this._extractPattern(context.userMessage || '');
    const patternKey = `pattern_${pattern.type}`;
    
    if (!this.conversationPatterns.has(patternKey)) {
      this.conversationPatterns.set(patternKey, {
        count: 0,
        avgReward: 0,
        responses: []
      });
    }
    
    const storedPattern = this.conversationPatterns.get(patternKey);
    storedPattern.count++;
    storedPattern.avgReward = (storedPattern.avgReward * (storedPattern.count - 1) + reward) / storedPattern.count;
    storedPattern.responses.push(response);
  }

  /**
   * Weaken unsuccessful pattern
   */
  _weakenPattern(context, response, reward) {
    const pattern = this._extractPattern(context.userMessage || '');
    const patternKey = `pattern_${pattern.type}`;
    
    if (this.conversationPatterns.has(patternKey)) {
      const storedPattern = this.conversationPatterns.get(patternKey);
      storedPattern.avgReward = Math.max(0, storedPattern.avgReward - (1 - reward) * this.learningRate);
    }
  }
}

module.exports = EpsilonConversationalLearning;

