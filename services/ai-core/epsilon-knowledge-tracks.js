/**
 * Epsilon AI 3-Track Knowledge System
 * ============================
 * Separates knowledge into 3 distinct tracks:
 * 1. Factual Knowledge (facts, data, information)
 * 2. Procedural Knowledge (how-to, processes, steps)
 * 3. Tone/Style Knowledge (conversation style, sales tone, personality)
 * 
 * Each track has:
 * - Separate vector storage
 * - Separate classifiers
 * - Separate weighting filters
 * - Confidence scoring
 */

const { createClient } = require('@supabase/supabase-js');

class EpsilonKnowledgeTracks {
  constructor() {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_KEY) {
      throw new Error('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set');
    }
    
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );
    
    // Track classifiers
    this.factualKeywords = [
      'data', 'fact', 'information', 'statistic', 'number', 'percentage',
      'research', 'study', 'analysis', 'report', 'finding', 'evidence',
      'definition', 'concept', 'theory', 'principle', 'law', 'rule'
    ];
    
    this.proceduralKeywords = [
      'step', 'process', 'procedure', 'method', 'how to', 'guide',
      'instruction', 'tutorial', 'workflow', 'algorithm', 'sequence',
      'first', 'then', 'next', 'finally', 'after', 'before', 'when'
    ];
    
    this.toneKeywords = [
      'tone', 'style', 'voice', 'personality', 'attitude', 'manner',
      'friendly', 'professional', 'casual', 'formal', 'empathetic',
      'enthusiastic', 'confident', 'helpful', 'supportive', 'sales'
    ];
  }

  /**
   * Classify text into knowledge track
   */
  classifyKnowledge(text) {
    if (!text || typeof text !== 'string') {
      return { track: 'factual', confidence: 0.5, scores: { factual: 0.33, procedural: 0.33, tone: 0.34 } };
    }
    
    const lower = text.toLowerCase();
    
    let factualScore = 0;
    let proceduralScore = 0;
    let toneScore = 0;
    
    // Score for factual
    for (const keyword of this.factualKeywords) {
      if (lower.includes(keyword)) factualScore++;
    }
    
    // Score for procedural
    for (const keyword of this.proceduralKeywords) {
      if (lower.includes(keyword)) proceduralScore++;
    }
    
    // Score for tone
    for (const keyword of this.toneKeywords) {
      if (lower.includes(keyword)) toneScore++;
    }
    
    // Normalize scores
    const total = factualScore + proceduralScore + toneScore;
    if (total === 0) {
      return { track: 'factual', confidence: 0.5, scores: { factual: 0.33, procedural: 0.33, tone: 0.34 } };
    }
    
    const scores = {
      factual: factualScore / total,
      procedural: proceduralScore / total,
      tone: toneScore / total
    };
    
    // Determine primary track
    let track = 'factual';
    let maxScore = scores.factual;
    
    if (scores.procedural > maxScore) {
      track = 'procedural';
      maxScore = scores.procedural;
    }
    
    if (scores.tone > maxScore) {
      track = 'tone';
      maxScore = scores.tone;
    }
    
    return {
      track,
      confidence: maxScore,
      scores
    };
  }

  /**
   * Store knowledge in appropriate track with embedding
   */
  async storeKnowledge(documentId, text, metadata = {}) {
    if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
      throw new Error('Invalid documentId provided to storeKnowledge()');
    }
    
    const docIdStr = String(documentId).trim();
    if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
      throw new Error(`Invalid document_id: ${documentId}`);
    }
    
    if (!text || typeof text !== 'string') {
      throw new Error('Invalid text provided to storeKnowledge()');
    }
    
    const classification = this.classifyKnowledge(text);
    
    // Generate embedding for vector search
    const embedding = await this._generateEmbedding(text);
    
    // Store in track-specific table
    const trackData = {
      document_id: docIdStr,
      text: text,
      track: classification.track,
      confidence: classification.confidence,
      scores: classification.scores,
      embedding: embedding, // Vector embedding for semantic search
      metadata: {
        ...metadata,
        classification
      }
    };
    
    // Store in Supabase
    const { data, error } = await this.supabase
      .from('epsilon_knowledge_tracks')
      .insert(trackData)
      .select()
      .limit(1).maybeSingle();
    
    if (error) {
      const errorStr = error?.message || error?.toString() || '';
      const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                         errorStr.includes('Cloudflare') || 
                         errorStr.includes('522') || 
                         errorStr.includes('521');
      if (isHtmlError) {
        console.warn('[WARN] [KNOWLEDGE TRACKS] Supabase connection issue while storing knowledge');
      } else {
        console.error('[KNOWLEDGE TRACKS] Failed to store knowledge:', error.message || error);
      }
      return null;
    }
    
    return data;
  }
  
  /**
   * Generate embedding for knowledge text
   */
  async _generateEmbedding(text) {
    if (!text || typeof text !== 'string') {
      return this._hashEmbedding('');
    }
    
    // Use Epsilon AI's embedding engine
    const EpsilonEmbeddings = require('./epsilon-embeddings');
    const EpsilonTokenizer = require('./epsilon-tokenizer');
    
    // Get or create embedding engine instance
    if (!global.epsilonEmbeddingEngine) {
      global.epsilonEmbeddingEngine = new EpsilonEmbeddings(384);
      global.epsilonTokenizer = new EpsilonTokenizer();
    }
    
    // Generate embedding
    if (global.epsilonEmbeddingEngine && global.epsilonEmbeddingEngine.isTrained) {
      return global.epsilonEmbeddingEngine.getTextEmbedding(text, global.epsilonTokenizer);
    } else {
      // Fallback hash embedding until training complete
      return this._hashEmbedding(text);
    }
  }
  
  /**
   * Hash-based embedding fallback
   */
  _hashEmbedding(text) {
    if (!text || typeof text !== 'string') {
      text = '';
    }
    
    const embedding = new Array(384).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    
    for (let i = 0; i < words.length && i < 384; i++) {
      const hash = this._hashString(words[i]);
      embedding[i] = (hash % 200 - 100) / 100; // Normalize to [-1, 1]
    }
    
    // Normalize vector
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      return embedding.map(val => val / magnitude);
    }
    
    return embedding;
  }
  
  /**
   * Simple string hash
   */
  _hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Retrieve knowledge from specific track using vector search
   */
  async retrieveKnowledge(query, track = null, topK = 5) {
    if (!query || typeof query !== 'string') {
      return [];
    }
    if (track && typeof track !== 'string') {
      track = null;
    }
    if (typeof topK !== 'number' || topK < 1 || topK > 100) {
      topK = 5;
    }
    
    try {
      // Generate query embedding
      const queryEmbedding = await this._generateEmbedding(query);
      if (!queryEmbedding) {
        // Fallback to text search if embedding generation fails
        return await this._textSearch(query, track, topK);
      }
      
      // Use vector similarity search via match_knowledge_tracks function
      const { data: results, error } = await this.supabase.rpc('match_knowledge_tracks', {
        query_embedding: queryEmbedding, // Array of 384 numbers - auto-converted to vector(384)
        p_track: track || null,
        match_threshold: 0.7,
        match_count: topK
      });
      
      if (error) {
        console.warn('[KNOWLEDGE TRACKS] Vector search failed, using text search:', error.message);
        return await this._textSearch(query, track, topK);
      }
      
      return results || [];
    } catch (error) {
      console.error('[KNOWLEDGE TRACKS] Error retrieving knowledge:', error);
      // Fallback to text search
      return await this._textSearch(query, track, topK);
    }
  }
  
  /**
   * Text-based search fallback
   */
  async _textSearch(query, track = null, topK = 5) {
    if (!query || typeof query !== 'string') {
      return [];
    }
    if (track && typeof track !== 'string') {
      track = null;
    }
    if (typeof topK !== 'number' || topK < 1 || topK > 100) {
      topK = 5;
    }
    
    // Sanitize query to prevent potential issues (Supabase handles SQL injection, but be safe)
    const sanitizedQuery = query.replace(/[%_]/g, '\\$&'); // Escape SQL LIKE wildcards
    
    let queryBuilder = this.supabase
      .from('epsilon_knowledge_tracks')
      .select('id, track, text, metadata, created_at, updated_at');
    
    if (track) {
      queryBuilder = queryBuilder.eq('track', track);
    }
    
    queryBuilder = queryBuilder.ilike('text', `%${sanitizedQuery}%`)
      .order('confidence', { ascending: false })
      .limit(topK);
    
    const { data, error } = await queryBuilder;
    
    if (error) {
      console.error('[KNOWLEDGE TRACKS] Text search failed:', error);
      return [];
    }
    
    return data || [];
  }

  /**
   * Get weighted knowledge for response generation
   */
  async getWeightedKnowledge(query, context = {}) {
    if (!query || typeof query !== 'string') {
      return [];
    }
    if (!context || typeof context !== 'object') {
      context = {};
    }
    
    const results = {
      factual: [],
      procedural: [],
      tone: []
    };
    
    // Retrieve from each track
    results.factual = await this.retrieveKnowledge(query, 'factual', 3);
    results.procedural = await this.retrieveKnowledge(query, 'procedural', 3);
    results.tone = await this.retrieveKnowledge(query, 'tone', 2);
    
    // Weight by context
    const weights = this._computeWeights(context);
    
    // Combine weighted results
    const combined = [];
    
    for (const item of results.factual) {
      combined.push({
        ...item,
        weight: weights.factual * item.confidence
      });
    }
    
    for (const item of results.procedural) {
      combined.push({
        ...item,
        weight: weights.procedural * item.confidence
      });
    }
    
    for (const item of results.tone) {
      combined.push({
        ...item,
        weight: weights.tone * item.confidence
      });
    }
    
    combined.sort((a, b) => b.weight - a.weight);
    
    return combined.slice(0, 5); // Top 5
  }

  /**
   * Compute weights based on context
   */
  _computeWeights(context) {
    // Default weights
    let factual = 0.4;
    let procedural = 0.3;
    let tone = 0.3;
    
    // Adjust based on context
    if (context.intent === 'question' || context.intent === 'information') {
      factual = 0.6;
      procedural = 0.2;
      tone = 0.2;
    } else if (context.intent === 'how_to' || context.intent === 'procedure') {
      factual = 0.2;
      procedural = 0.6;
      tone = 0.2;
    } else if (context.intent === 'conversation' || context.intent === 'sales') {
      factual = 0.2;
      procedural = 0.2;
      tone = 0.6;
    }
    
    return { factual, procedural, tone };
  }
}

module.exports = EpsilonKnowledgeTracks;


