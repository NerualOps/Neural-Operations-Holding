/**
 * Epsilon AI Self-Learning System
 * -------------------------
 * - Epsilon AI talks to itself to practice and learn (100 conversations per cycle)
 * - Sets learning objectives and goals
 * - Evaluates its own responses with strict, particular rating system
 * - Continuously improves through self-training (transformer-based learning)
 * - Learns from documents by internalizing patterns, not quoting
 * - Understands document categories and uses learned knowledge naturally
 * - Conversations are varied and dynamic (70% new, 20% varied, 10% repeated to test improvement)
 * - Never quotes documents - uses digested information naturally in conversation
 */

const { createClient } = require('@supabase/supabase-js');

const _silent = () => {};
if (typeof console !== 'undefined') {
  console.log = _silent;
  console.info = _silent;
  console.debug = _silent;
}

class EpsilonSelfLearning {
  constructor(epsilonLanguageEngine) {
    this.languageEngine = epsilonLanguageEngine;
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );
    
    // Configurable learning objectives (can be modified)
    this.learningObjectives = [
      {
        id: 'avoid_quoting',
        name: 'Avoid Quoting Documents',
        description: 'Learn from documents but never quote them directly - digest information naturally',
        target: 0.98, // 98% - stricter requirement
        current: 0.0,
        enabled: true,
        weight: 1.0
      },
      {
        id: 'sales_tone',
        name: 'Master Sales Tone',
        description: 'Use sales training documents to learn HOW to talk (tone, language, style), not WHAT to say',
        target: 0.95, // 95% - stricter requirement
        current: 0.0,
        enabled: true,
        weight: 1.0
      },
      {
        id: 'conversational',
        name: 'Be Conversational',
        description: 'Respond naturally like a human, not a document reader or search engine',
        target: 0.92, // 92% - stricter requirement
        current: 0.0,
        enabled: true,
        weight: 1.0
      },
      {
        id: 'learned_patterns',
        name: 'Use Learned Patterns',
        description: 'Apply patterns learned from training (transformer-based learning), not direct quotes or document searches',
        target: 0.95, // 95% - stricter requirement
        current: 0.0,
        enabled: true,
        weight: 1.0
      },
      {
        id: 'internalize_knowledge',
        name: 'Internalize Knowledge',
        description: 'Digest information during training, use naturally in conversation when asked',
        target: 0.90, // 90% - stricter requirement
        current: 0.0,
        enabled: true,
        weight: 1.0
      }
    ];
    
    this.selfConversations = [];
    this.isRunning = false;
    this.trainingInterval = null;
    this.cycleInProgress = false; // Prevent overlapping cycles
    this.lastLogTime = 0; // Track last log time to reduce console spam
    this.logInterval = 30000;
    
    // Adaptive cycle configuration
    this.cycleConfig = {
      baseConversationsPerCycle: 100,
      minConversationsPerCycle: 50,
      maxConversationsPerCycle: 200,
      adaptiveEnabled: true,
      improvementThreshold: 0.02 // 2% improvement triggers cycle size adjustment
    };
    
    // Cycle performance tracking
    this.cyclePerformance = {
      lastCycleScore: 0,
      averageScore: 0,
      scoreHistory: [],
      improvementRate: 0
    }; // Only log every 30 seconds max
  }

  /**
   * Start continuous self-learning loop
   */
  async startSelfLearning() {
    // DISABLE self-learning in production (Render) - only run locally
    const isProduction = process.env.NODE_ENV === 'production';
    
    if (isProduction) {
      _silent('[SELF-LEARNING] Production mode: Self-learning disabled');
      this.isRunning = false;
      return;
    }
    
    if (this.isRunning) {
      _silent('[SELF-LEARNING] Already running');
      return;
    }

    this.isRunning = true;
    _silent('[SELF-LEARNING] Starting continuous self-learning system (LOCAL ONLY)...');

    // Adaptive interval based on improvement rate
    const getAdaptiveInterval = () => {
      if (!this.cycleConfig.adaptiveEnabled) {
        return 5 * 60 * 1000; // Default 5 minutes
      }
      
      const improvementRate = this.cyclePerformance.improvementRate || 0;
      if (improvementRate > 0.05) {
        // High improvement - train more frequently
        return 3 * 60 * 1000; // 3 minutes
      } else if (improvementRate > 0.02) {
        // Moderate improvement - normal frequency
        return 5 * 60 * 1000; // 5 minutes
      } else {
        // Low improvement - train less frequently
        return 10 * 60 * 1000; // 10 minutes
      }
    };
    
    // Run self-learning with adaptive interval
    const scheduleNextCycle = () => {
      if (this.trainingInterval) {
        clearInterval(this.trainingInterval);
      }
      const interval = getAdaptiveInterval();
      this.trainingInterval = setInterval(async () => {
        if (!this.cycleInProgress) {
          await this.runSelfLearningCycle();
          // Reschedule with updated interval
          scheduleNextCycle();
        }
      }, interval);
    };
    
    scheduleNextCycle();

    // Run initial cycle
    await this.runSelfLearningCycle();
  }

  /**
   * Stop self-learning
   */
  stopSelfLearning() {
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }
    this.isRunning = false;
    _silent('[SELF-LEARNING] Stopped self-learning system');
  }

  /**
   * Run one self-learning cycle
   */
  async runSelfLearningCycle() {
    // CRITICAL: Never allow self-learning in production
    const isProduction = process.env.NODE_ENV === 'production';
    if (isProduction) {
      _silent('🚫 [SELF-LEARNING] BLOCKED: Self-learning is disabled in production mode');
      return;
    }
    
    // Prevent overlapping cycles
    if (this.cycleInProgress) {
      _silent('[SELF-LEARNING] Cycle already in progress, skipping...');
      return;
    }

    this.cycleInProgress = true;
    const startTime = Date.now();

    try {
      const now = Date.now();
      const shouldLog = (now - this.lastLogTime) > this.logInterval;
      
      // Check if Python services are ready (required)
      const isPythonReady = this.languageEngine.isPythonReady();
      
      if (!isPythonReady) {
        if (shouldLog) {
          console.warn('[SELF-LEARNING] Python services not ready - waiting for services to start...');
          this.lastLogTime = now;
        }
        this.cycleInProgress = false;
        return;
      }
      
      // Check if training is in progress - skip self-learning to avoid memory issues
      if (global.epsilonAutomaticTraining && global.epsilonAutomaticTraining.status && global.epsilonAutomaticTraining.status.isTraining) {
        if (shouldLog) {
          _silent('[SELF-LEARNING] Training in progress - skipping self-learning to conserve memory');
          this.lastLogTime = now;
        }
        this.cycleInProgress = false;
        return;
      }
      
      // Check if model is ready (optional - self-learning can help train the model)
      const isModelReady = this.languageEngine.isModelReady();
      
      if (!isModelReady) {
        // Only log once per interval to reduce console spam
        if (shouldLog) {
          _silent('[SELF-LEARNING] Model not trained yet - self-learning will start after initial training completes.');
          _silent('[SELF-LEARNING] Automatic training will train the model first, then self-learning can begin.');
          this.lastLogTime = now;
        }
        this.cycleInProgress = false;
        return;
      }

      // 1. Calculate adaptive conversation count
      let conversationsPerCycle = this.cycleConfig.baseConversationsPerCycle;
      if (this.cycleConfig.adaptiveEnabled) {
        const improvementRate = this.cyclePerformance.improvementRate || 0;
        if (improvementRate > this.cycleConfig.improvementThreshold) {
          // High improvement - increase cycle size
          conversationsPerCycle = Math.min(
            this.cycleConfig.maxConversationsPerCycle,
            Math.floor(this.cycleConfig.baseConversationsPerCycle * 1.5)
          );
        } else if (improvementRate < -0.01) {
          // Declining - reduce cycle size
          conversationsPerCycle = Math.max(
            this.cycleConfig.minConversationsPerCycle,
            Math.floor(this.cycleConfig.baseConversationsPerCycle * 0.75)
          );
        }
      }
      
      _silent(`[SELF-LEARNING] Generating ${conversationsPerCycle} self-conversations for training (adaptive: ${this.cycleConfig.adaptiveEnabled ? 'enabled' : 'disabled'})...`);
      const conversations = await this.generateSelfConversations(conversationsPerCycle);
      _silent(`[SELF-LEARNING] Generated ${conversations.length} conversations`);
      
      if (conversations.length === 0) {
        console.warn('[SELF-LEARNING] No conversations generated - model may not be responding correctly');
        this.cycleInProgress = false;
        return;
      }
      
      // Log sample conversations
      conversations.slice(0, 2).forEach((conv, idx) => {
        _silent(`   Conversation ${idx + 1}:`);
        _silent(`      Q: ${conv.question}`);
        _silent(`      A: ${conv.response.substring(0, 100)}...`);
      });
      
      // 2. Evaluate responses against learning objectives
      _silent('[SELF-LEARNING] Evaluating responses...');
      const evaluations = await this.evaluateResponses(conversations);
      _silent(`[SELF-LEARNING] Evaluated ${evaluations.length} responses`);
      
      // Log evaluation results (Epsilon AI rates her own responses - transformer-based self-evaluation)
      // More strict evaluation - must meet target thresholds, not just 0.8
      evaluations.forEach((evaluation, idx) => {
        const objectiveTargets = this.learningObjectives.reduce((acc, obj) => {
          acc[obj.id] = obj.target;
          return acc;
        }, {});
        
        const passedCount = Object.entries(evaluation.scores).filter(([id, score]) => {
          const target = objectiveTargets[id] || 0.8;
          return score >= target; // Must meet specific target, not just 0.8
        }).length;
        
        const totalObjectives = Object.keys(evaluation.scores).length;
        const rating = (passedCount / totalObjectives * 100).toFixed(1);
        
        // Calculate average score across all objectives
        const avgScore = Object.values(evaluation.scores).reduce((a, b) => a + b, 0) / totalObjectives;
        
        _silent(`   Response ${idx + 1} self-rating: ${rating}% (${passedCount}/${totalObjectives} objectives passed, avg score: ${(avgScore * 100).toFixed(1)}%)`);
        
        if (evaluation.issues.length > 0) {
          _silent(`      Issues identified: ${evaluation.issues.map(i => `${i.objective} (${(i.score * 100).toFixed(1)}% < ${(i.target * 100).toFixed(1)}%)`).join(', ')}`);
        } else {
          _silent(`      All objectives met - response is good`);
        }
      });
      
      // 3. Identify areas for improvement
      const improvements = this.identifyImprovements(evaluations);
      _silent(`[SELF-LEARNING] Identified ${improvements.length} improvements needed`);
      
      // 4. Retrain on improved examples (only if significant improvements found)
      // CRITICAL: This actually updates the model weights through real training
      if (improvements.length > 0) {
        _silent(`[SELF-LEARNING] Found ${improvements.length} responses needing improvement - retraining model...`);
        await this.retrainOnImprovements(improvements);
        _silent('[SELF-LEARNING] Retraining complete - model weights updated');
      } else {
        _silent('[SELF-LEARNING] All responses passed objectives - no retraining needed');
      }
      
      // 5. Update learning objectives progress in Supabase
      // This stores progress so we can track improvement over time
      await this.updateLearningProgress(evaluations);
      
      // Calculate cycle performance metrics
      const avgScore = evaluations.length > 0 
        ? evaluations.reduce((sum, e) => {
            const scores = Object.values(e.scores);
            return sum + (scores.reduce((a, b) => a + b, 0) / scores.length);
          }, 0) / evaluations.length
        : 0;
      
      const previousScore = this.cyclePerformance.lastCycleScore;
      const improvement = previousScore > 0 ? avgScore - previousScore : 0;
      this.cyclePerformance.lastCycleScore = avgScore;
      this.cyclePerformance.scoreHistory.push(avgScore);
      if (this.cyclePerformance.scoreHistory.length > 10) {
        this.cyclePerformance.scoreHistory.shift();
      }
      this.cyclePerformance.averageScore = this.cyclePerformance.scoreHistory.reduce((a, b) => a + b, 0) / this.cyclePerformance.scoreHistory.length;
      this.cyclePerformance.improvementRate = improvement;
      
      // Log progress
      this.learningObjectives.forEach(obj => {
        const progress = (obj.current / obj.target * 100).toFixed(1);
        _silent(`   ${obj.name}: ${(obj.current * 100).toFixed(1)}% (target: ${(obj.target * 100).toFixed(1)}%)`);

      });

      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      _silent(`[SELF-LEARNING] Self-learning cycle complete (${duration}s, avg score: ${(avgScore * 100).toFixed(1)}%, improvement: ${(improvement * 100).toFixed(2)}%)`);
    } catch (error) {
      // Enhanced error handling with context
      const errorContext = {
        timestamp: new Date().toISOString(),
        cycleStartTime: startTime,
        duration: Date.now() - startTime,
        errorMessage: error.message,
        errorStack: error.stack,
        errorCode: error.code,
        pythonReady: this.languageEngine?.isPythonReady?.() || false,
        modelReady: this.languageEngine?.isModelReady?.() || false
      };
      
      console.error('[ERROR] [SELF-LEARNING] Self-learning cycle failed:', error);
      console.error('[ERROR] [SELF-LEARNING] Error context:', JSON.stringify(errorContext, null, 2));
      console.error('   Stack:', error.stack);
      
      // Store error for analysis
      try {
        await this.supabase.from('learning_metrics').insert({
          name: 'self_learning_error',
          value: 1,
          metadata: errorContext
        });
      } catch (dbError) {
        console.warn('[WARN] [SELF-LEARNING] Failed to store error in database:', dbError.message);
      }
    } finally {
      this.cycleInProgress = false;
    }
  }

  /**
   * Generate self-conversations where Epsilon AI talks to herself
   */
  async generateSelfConversations(count = 100) {
    const conversations = [];
    
    // FIRST: Get REAL topics from actual user conversations in Supabase
    // This makes training dynamic and based on what users actually ask
    let realTopics = [];
    try {
      // Get real user questions from messages table
      const { data: realMessages, error: msgError } = await this.supabase
        .from('messages')
        .select('text, role, created_at')
        .eq('role', 'user')
        .order('created_at', { ascending: false })
        .limit(200); // Get recent user questions
      
      if (!msgError && realMessages && realMessages.length) {
        realTopics = realMessages
          .map(msg => msg.text)
          .filter(q => q && q.length > 10 && q.length < 200) // Valid questions
          .slice(0, 50); // Use top 50 real questions
        
        _silent(`[SELF-LEARNING] Found ${realTopics.length} real user questions from conversations`);
      }
      
      // Also get topics from epsilon_conversations
      const { data: epsilonConvs, error: convError } = await this.supabase
        .from('epsilon_conversations')
        .select('user_message')
        .order('created_at', { ascending: false })
        .limit(50);
      
      if (!convError && epsilonConvs && epsilonConvs.length) {
        const epsilonTopics = epsilonConvs
          .map(c => c.user_message)
          .filter(q => q && q.length > 10 && q.length < 200);
        realTopics = [...realTopics, ...epsilonTopics].slice(0, 50);
      }
    } catch (error) {
      console.warn('[SELF-LEARNING] Failed to load real topics, using fallback:', error.message);
    }
    
    // Fallback: Expanded, varied topic pool for dynamic conversations (if no real topics)
    const baseTopics = [
      // Business & Services
      'How can I help with business automation?',
      'What services do you offer?',
      'Tell me about your sales approach',
      'How does your AI work?',
      'What makes you different?',
      'Can you help with website development?',
      'What is your pricing?',
      'How do I get started?',
      // Sales & Communication
      'How do you approach new clients?',
      'What sales techniques do you recommend?',
      'How do you build rapport with customers?',
      'What makes a good sales conversation?',
      'How do you handle objections?',
      'What is your communication style?',
      'How do you explain complex topics simply?',
      'What questions should I ask prospects?',
      // Technical & AI
      'How does machine learning work?',
      'What is artificial intelligence?',
      'How do neural networks learn?',
      'What are the benefits of AI automation?',
      'How do you train an AI model?',
      'What is natural language processing?',
      'How do transformer models work?',
      'What is transformer architecture?',
      // General Conversation
      'What are you working on today?',
      'How can I improve my business?',
      'What trends are you seeing?',
      'What advice do you have?',
      'How do you stay current?',
      'What challenges do you face?',
      'How do you solve problems?',
      'What inspires you?',
      // Case Studies & Examples
      'Can you share a success story?',
      'What results have you achieved?',
      'How do you measure success?',
      'What case studies do you have?',
      'Can you give me an example?',
      'What real-world applications exist?',
      // Process & Methodology
      'What is your process?',
      'How do you approach projects?',
      'What steps do you follow?',
      'How do you ensure quality?',
      'What methodology do you use?',
      'How do you plan your work?',
      // Relationship Building
      'How do you build trust?',
      'What makes a good relationship?',
      'How do you maintain connections?',
      'What is important in communication?',
      'How do you listen effectively?',
      'How do you show empathy?',
      // Problem Solving
      'How do you identify problems?',
      'What is your problem-solving approach?',
      'How do you analyze situations?',
      'What tools do you use?',
      'How do you make decisions?',
      'What is your thinking process?',
      // Value & Benefits
      'What value do you provide?',
      'What benefits can I expect?',
      'What outcomes are possible?',
      'How do you create value?',
      'What ROI can I expect?',
      'What makes you valuable?',
      // Future & Vision
      'Where do you see yourself in the future?',
      'What are your goals?',
      'How do you plan to grow?',
      'What innovations are coming?',
      'What is your vision?',
      'How do you stay ahead?'
    ];
    
    // Combine real topics with base topics (prioritize real ones)
    const allTopics = realTopics.length > 0 
      ? [...realTopics, ...baseTopics] // Use real topics first
      : baseTopics; // Fallback to static if no real topics

    // Track which topics we've used to ensure variety
    const usedTopics = new Set();
    const repeatedTopics = []; // Some topics we'll repeat to test improvement
    
    _silent(`[SELF-LEARNING] Generating ${count} conversations (${realTopics.length} real topics + ${baseTopics.length} base topics)...`);

    for (let i = 0; i < count; i++) {
      let topic;
      
      // Strategy: 60% real topics (if available), 30% new base topics, 10% repeated (to test improvement)
      const strategy = Math.random();
      
      if (strategy < 0.6 && realTopics.length > 0) {
        // Use REAL user questions (60% of the time if available)
        const unusedReal = realTopics.filter(t => !usedTopics.has(t));
        if (unusedReal.length > 0) {
          topic = unusedReal[Math.floor(Math.random() * unusedReal.length)];
          usedTopics.add(topic);
        } else {
          // All real topics used, pick random real topic
          topic = realTopics[Math.floor(Math.random() * realTopics.length)];
        }
      } else if (strategy < 0.9) {
        // New topic - pick from unused base topics first, then random
        const unusedTopics = baseTopics.filter(t => !usedTopics.has(t));
        if (unusedTopics.length > 0) {
          topic = unusedTopics[Math.floor(Math.random() * unusedTopics.length)];
          usedTopics.add(topic);
        } else {
          // All topics used, pick random and add variation
          topic = baseTopics[Math.floor(Math.random() * baseTopics.length)];
          // Add slight variation to make it dynamic
          const variations = ['Tell me more about', 'Can you explain', 'I want to know about', 'Help me understand'];
          topic = `${variations[Math.floor(Math.random() * variations.length)]} ${topic.toLowerCase()}`;
        }
      } else {
        // Repeat a topic to test if Epsilon AI gives better response (transformer-based improvement testing)
        if (repeatedTopics.length > 0 && Math.random() < 0.5) {
          topic = repeatedTopics[Math.floor(Math.random() * repeatedTopics.length)];
          _silent(`   Repeating topic to test improvement: "${topic}"`);
        } else {
          topic = allTopics[Math.floor(Math.random() * allTopics.length)];
          repeatedTopics.push(topic);
        }
      }
      
      try {
        _silent(`   Generating response ${i + 1}/${count} for: "${topic}"`);
        
        // Generate response using current model
        const response = await this.languageEngine.generate({
          userMessage: topic,
          ragContext: [], // Don't pass context - let model use learned patterns
          persona: { mode: 'advisor', tone: 'natural' }
        });

        if (response && response.text) {
          _silent(`   Generated response (${response.text.length} chars)`);
          conversations.push({
            question: topic,
            response: response.text,
            timestamp: new Date().toISOString()
          });
        } else {
          console.warn(`   No response generated for "${topic}" - model may not be ready or returned null`);
        }
      } catch (error) {
        console.error(`[ERROR] Failed to generate response for "${topic}":`, error.message);
        if (error.stack) {
          console.error(`      Stack: ${error.stack.split('\n')[0]}`);
        }
      }
    }

    _silent(`[SELF-LEARNING] Successfully generated ${conversations.length}/${count} conversations`);
    return conversations;
  }

  /**
   * Evaluate responses against learning objectives
   */
  async evaluateResponses(conversations) {
    const evaluations = [];

    for (const conv of conversations) {
      const evaluation = {
        question: conv.question,
        response: conv.response,
        scores: {},
        issues: [],
        passed: true
      };

      // Check each learning objective
      for (const objective of this.learningObjectives) {
        const score = this.evaluateAgainstObjective(conv.response, objective);
        evaluation.scores[objective.id] = score;

        if (score < objective.target) {
          evaluation.issues.push({
            objective: objective.id,
            score,
            target: objective.target,
            message: `Failed ${objective.name}: score ${score.toFixed(2)} < target ${objective.target}`
          });
          evaluation.passed = false;
        }
      }

      evaluations.push(evaluation);
    }

    return evaluations;
  }

  /**
   * Evaluate a response against a specific learning objective
   */
  evaluateAgainstObjective(response, objective) {
    const text = (response || '').toLowerCase();

    switch (objective.id) {
      case 'avoid_quoting': {
        // Stricter check for document quotes, titles, "according to", etc.
        const quotePatterns = [
          /according to (our |the )?knowledge base/i,
          /according to (the |our )?document/i,
          /from (the |our )?document/i,
          /based on (the |our )?(document|knowledge base)/i,
          /the art of mastering/i,
          /the psychology of/i,
          /a study on/i,
          /by [A-Z][a-z]+ [A-Z][a-z]+/i, // Author names
          /©\s*\d{4}/i, // Copyright
          /(document|pdf|file|paper|study|research|article) (says|states|indicates|shows|mentions)/i,
          /(as mentioned|as stated|as noted) in (the |our )?(document|knowledge base)/i,
          /(per|from|via) (the |our )?(document|knowledge base|source)/i
        ];
        const hasQuotes = quotePatterns.some(pattern => pattern.test(text));
        
        // Also check for document-like structure (too formal, structured)
        const isTooStructured = /(first|second|third|finally|in conclusion|to summarize)/i.test(text) && 
                                text.split('.').length > 5; // Long structured response
        
        if (hasQuotes) return 0.0; // Zero score if quotes detected
        if (isTooStructured) return 0.3; // Low score if too document-like
        return 1.0; // Perfect score if natural
      }

      case 'sales_tone': {
        // Stricter check if response sounds natural and sales-oriented
        const salesIndicators = [
          /\b(help|assist|support|guide|show|explain|let me|here's|that's)\b/i,
          /\b(you|your|we|our|I|me|my)\b/i,
          /\b(can|will|would|should|might)\b/i,
          /\b(think|believe|feel|know|understand)\b/i
        ];
        const salesCount = salesIndicators.filter(pattern => pattern.test(text)).length;
        const hasSalesTone = salesCount >= 2; // Must have at least 2 indicators
        
        const isTooFormal = /(according to|based on|the document|the study|furthermore|moreover|in addition)/i.test(text);
        const isTooRobotic = /(please note|it should be noted|it is important to|one must)/i.test(text);
        const hasNaturalFlow = /(i|we|you|let me|here's|that's|so|well|actually)/i.test(text);
        
        if (hasSalesTone && hasNaturalFlow && !isTooFormal && !isTooRobotic) return 1.0;
        if (hasSalesTone && !isTooFormal) return 0.7;
        if (hasNaturalFlow) return 0.5;
        return 0.3; // Low score if too formal/robotic
      }

      case 'conversational': {
        // Stricter check if response sounds natural, not robotic
        const hasDocumentLanguage = /(according to|based on|the document|the study|the art of|the psychology|per the|as per)/i.test(text);
        const hasConversationalFlow = /(i|we|you|your|let me|here's|that's|so|well|actually|yeah|sure)/i.test(text);
        const hasQuestions = /\?/.test(text); // Questions make it more conversational
        const hasContractions = /(i'm|you're|we're|that's|here's|it's|can't|won't|don't)/i.test(text);
        const isTooLong = text.length > 500; // Very long responses are less conversational
        const isTooShort = text.length < 20; // Too short is also not good
        
        // Check for variety in sentence structure
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const hasVariety = sentences.length >= 2 && sentences.length <= 8; // Good variety
        
        if (hasDocumentLanguage) return 0.2; // Very low if document language detected
        if (!hasConversationalFlow) return 0.3; // Low if no natural flow
        if (isTooLong || isTooShort) return 0.6; // Medium if length issues
        if (!hasVariety) return 0.7; // Medium if no sentence variety
        
        let score = 0.8; // Base score
        if (hasQuestions) score += 0.1;
        if (hasContractions) score += 0.1;
        
        return Math.min(1.0, score);
      }

      case 'learned_patterns': {
        // Stricter check if response uses learned patterns vs direct quotes
        const hasDirectQuotes = /(according to|based on|the document|the study|by [A-Z]|from the|per the)/i.test(text);
        const hasSubstance = text.length > 30;
        const usesFirstPerson = /(i|we|my|our)/i.test(text);
        const usesNaturalLanguage = /(think|believe|feel|know|understand|help|can|will)/i.test(text);
        const hasLearnedInsights = !/(search|find|look up|retrieve|document|knowledge base|database)/i.test(text);
        
        if (hasDirectQuotes) return 0.1; // Very low if direct quotes
        if (!hasSubstance) return 0.2; // Low if no substance
        if (!hasLearnedInsights) return 0.3; // Low if searching language
        
        let score = 0.6; // Base score
        if (usesFirstPerson) score += 0.2;
        if (usesNaturalLanguage) score += 0.2;
        
        return Math.min(1.0, score);
      }

      case 'internalize_knowledge': {
        // Stricter check if Epsilon AI is using digested knowledge naturally vs searching/quoting
        const hasSearchLanguage = /(search|find|look up|retrieve|document|knowledge base|database|query|fetch)/i.test(text);
        const hasNaturalUsage = /(i|we|you|let me|here's|that's|can help|think|believe|know)/i.test(text);
        const hasPersonalInsight = /(i've|i have|i learned|i understand|i know|i think|i believe)/i.test(text);
        const hasApplication = /(can|will|would|should|might|could)/i.test(text) && 
                              /(help|assist|support|guide|show|explain|do|make|create)/i.test(text);
        
        if (hasSearchLanguage) return 0.2; // Very low if searching language
        if (!hasNaturalUsage) return 0.4; // Low if not natural
        
        let score = 0.7; // Base score
        if (hasPersonalInsight) score += 0.15;
        if (hasApplication) score += 0.15;
        
        return Math.min(1.0, score);
      }

      default:
        return 0.5;
    }
  }

  /**
   * Identify improvements needed
   */
  identifyImprovements(evaluations) {
    const improvements = [];

    for (const evaluation of evaluations) {
      if (!evaluation.passed) {
        // Create improved version
        const improved = this.improveResponse(evaluation.response, evaluation.issues);
        improvements.push({
          original: evaluation.response,
          improved: improved,
          question: evaluation.question,
          issues: evaluation.issues
        });
      }
    }

    return improvements;
  }

  /**
   * Improve a response based on identified issues
   */
  improveResponse(response, issues) {
    let improved = response;

    for (const issue of issues) {
      switch (issue.objective) {
        case 'avoid_quoting':
          // Remove all document references
          improved = improved
            .replace(/according to (our |the )?knowledge base[:,-]?\s*/gi, '')
            .replace(/based on (our |the )?knowledge base[:,-]?\s*/gi, '')
            .replace(/according to (the |our )?document[:,-]?\s*/gi, '')
            .replace(/from (the |our )?document[:,-]?\s*/gi, '')
            .replace(/^(the art of mastering|the psychology of|a study on)[:,\s]*/gi, '')
            .replace(/\b(by|author|published|document|pdf|study|paper)\b[:\s]*[A-Z][^.]*\./gi, '')
            .replace(/\b©\s*\d{4}[^.]*\./gi, '');
          
          // Rewrite in first person
          improved = improved
            .replace(/\bwe\b/gi, 'I')
            .replace(/\bour\b/gi, 'my')
            .replace(/\bclients?\b/gi, 'customers');
          break;

        case 'sales_tone':
          // Make it more conversational
          if (!improved.toLowerCase().startsWith('i ') && !improved.toLowerCase().startsWith('here')) {
            improved = `I ${improved.charAt(0).toLowerCase()}${improved.slice(1)}`;
          }
          break;

        case 'conversational':
          // Add natural flow
          if (!/(i|we|you|let me|here's)/i.test(improved)) {
            improved = `Here's how I think about that: ${improved}`;
          }
          break;

        case 'internalize_knowledge':
          // Remove any document search language, make it natural
          improved = improved
            .replace(/\b(search|find|look up|retrieve|document|knowledge base)\b/gi, '')
            .replace(/\b(according to|based on|from the document)\b/gi, '');
          break;
      }
    }

    return improved.trim();
  }

  /**
   * Retrain model on improved examples
   * CRITICAL: This actually updates the model weights through real training
   */
  async retrainOnImprovements(improvements) {
    if (improvements.length === 0) return;

    _silent(`[SELF-LEARNING] Retraining on ${improvements.length} improved examples...`);

    // Convert improvements to training samples
    // These samples will be used to actually train the transformer model
    const samples = improvements.map(imp => ({
      text: imp.improved,
      category: 'general',
      tone: 'natural',
      weight: 1.2, // Higher weight for self-improved examples
      signals: {
        source: 'self_learning',
        original_issue: imp.issues.map(i => i.objective).join(','),
        improved: true
      }
    }));

    // Add to training data and retrain (fine-tuning only - no pre-training needed)
    // Self-learning uses incremental fine-tuning, not full pre-training
    // This calls trainNow() which sends samples to Python service /train endpoint
    // Python service actually trains the transformer model and updates weights
    try {
      _silent(`[SELF-LEARNING] Starting incremental fine-tuning on ${samples.length} improved examples...`);
      _silent(`   Sample preview: ${samples[0]?.text?.substring(0, 100)}...`);
      
      const trainingResult = await this.languageEngine.trainNow({
        reason: 'self_learning_improvement',
        additionalSamples: samples,
        trainingMode: 'fine_tune' // Only fine-tune, skip pre-training for self-learning
      });
      
      if (trainingResult && trainingResult.success) {
        _silent(`[SELF-LEARNING] Incremental fine-tuning complete - model weights updated`);
        _silent(`   Training stats:`, trainingResult.stats);
      } else {
        console.warn(`[SELF-LEARNING] Training completed but success flag not set`);
      }
    } catch (error) {
      console.error('[ERROR]  [SELF-LEARNING] Retraining failed:', error);
      console.error('   Stack:', error.stack);
    }
  }

  /**
   * Update learning progress
   */
  async updateLearningProgress(evaluations) {
    // Calculate average scores for each objective
    for (const objective of this.learningObjectives) {
      const scores = evaluations.map(evaluation => evaluation.scores[objective.id] || 0);
      const avgScore = scores.length > 0 
        ? scores.reduce((a, b) => a + b, 0) / scores.length 
        : 0;
      
      objective.current = avgScore;
    }

    // Store progress in Supabase (tracks learning objectives over time)
    try {
      const { data, error } = await this.supabase
        .from('epsilon_learning_objectives')
        .upsert({
          objective_id: 'self_learning',
          objectives: this.learningObjectives,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'objective_id'
        });
      
      if (error) {
        console.warn('[SELF-LEARNING] Failed to store progress in Supabase:', error.message);
      } else {
        _silent('[SELF-LEARNING] Learning progress stored in Supabase');
      }
    } catch (error) {
      console.warn('[SELF-LEARNING] Failed to store progress:', error.message);
    }
  }

  /**
   * Get current learning progress
   */
  getLearningProgress() {
    return {
      objectives: this.learningObjectives,
      isRunning: this.isRunning,
      totalConversations: this.selfConversations.length
    };
  }

  /**
   * Manually trigger self-learning cycle
   */
  async triggerSelfLearning() {
    return await this.runSelfLearningCycle();
  }
}

module.exports = EpsilonSelfLearning;


