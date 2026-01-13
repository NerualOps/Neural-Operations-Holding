/**
 * Epsilon AI Memory Engine
 * ===================
 * Multi-layered, weighted, time-decaying memory system
 * 
 * Memory Types:
 * - Long-term knowledge (facts, concepts)
 * - User-specific memory (preferences, history)
 * - Task memory (current conversation context)
 * - Short-term working memory (immediate context)
 * 
 * Features:
 * - Confidence scoring
 * - Time decay
 * - Weighted retrieval
 * - Memory consolidation
 */

class EpsilonMemoryEngine {
  constructor() {
    // Memory layers
    this.longTermMemory = new Map(); // Persistent knowledge
    this.userMemory = new Map(); // Per-user memory
    this.taskMemory = new Map(); // Current task context
    this.workingMemory = []; // Short-term buffer (last N items)
    
    // Memory configuration
    this.workingMemorySize = 10;
    this.decayRate = 0.95; // Decay per time step
    this.minConfidence = 0.3; // Minimum confidence to keep
  }

  /**
   * Store memory
   */
  storeMemory(type, key, content, metadata = {}) {
    // Safety check: validate inputs
    if (!type || typeof type !== 'string') {
      throw new Error('Invalid type provided to storeMemory()');
    }
    if (!key || typeof key !== 'string') {
      throw new Error('Invalid key provided to storeMemory()');
    }
    if (content === undefined || content === null) {
      throw new Error('Invalid content provided to storeMemory()');
    }
    if (metadata && typeof metadata !== 'object') {
      metadata = {};
    }
    
    const memory = {
      type,
      key,
      content,
      metadata: {
        ...metadata,
        timestamp: Date.now(),
        accessCount: 0,
        lastAccessed: Date.now(),
        confidence: metadata.confidence || 0.8
      }
    };
    
    // Store in appropriate layer
    switch (type) {
      case 'long_term':
        this.longTermMemory.set(key, memory);
        break;
      case 'user':
        if (!this.userMemory.has(metadata.userId)) {
          this.userMemory.set(metadata.userId, new Map());
        }
        this.userMemory.get(metadata.userId).set(key, memory);
        break;
      case 'task':
        this.taskMemory.set(key, memory);
        break;
      case 'working':
        this.workingMemory.push(memory);
        if (this.workingMemory.length > this.workingMemorySize) {
          this.workingMemory.shift(); // Remove oldest
        }
        break;
    }
    
    return memory;
  }

  /**
   * Retrieve memory with weighted scoring
   */
  retrieveMemory(query, context = {}) {
    // Safety check: validate inputs
    if (!query || typeof query !== 'string') {
      return [];
    }
    if (context && typeof context !== 'object') {
      context = {};
    }
    
    const results = [];
    
    // Search all layers
    results.push(...this._searchLayer(this.longTermMemory, query, 1.0));
    results.push(...this._searchLayer(this.taskMemory, query, 0.8));
    
    // User-specific memory
    if (context.userId && this.userMemory.has(context.userId)) {
      results.push(...this._searchLayer(this.userMemory.get(context.userId), query, 0.9));
    }
    
    // Working memory
    for (const memory of this.workingMemory) {
      const score = this._computeRelevance(memory, query);
      if (score > 0.3) {
        results.push({
          ...memory,
          relevanceScore: score * 0.7 // Lower weight for working memory
        });
      }
    }
    
    // Apply time decay
    const now = Date.now();
    for (const result of results) {
      const age = (now - result.metadata.timestamp) / (1000 * 60 * 60); // Hours
      const decay = Math.pow(this.decayRate, age);
      result.finalScore = result.relevanceScore * result.metadata.confidence * decay;
    }
    
    // Sort by final score
    results.sort((a, b) => b.finalScore - a.finalScore);
    
    // Filter low confidence
    return results.filter(r => r.finalScore >= this.minConfidence);
  }

  /**
   * Search memory layer
   */
  _searchLayer(layer, query, weight) {
    // Safety check: validate inputs
    if (!layer || !(layer instanceof Map) || !query || typeof query !== 'string') {
      return [];
    }
    if (typeof weight !== 'number' || weight < 0 || weight > 1) {
      weight = 1.0;
    }
    
    const results = [];
    const queryLower = query.toLowerCase();
    
    for (const [key, memory] of layer.entries()) {
      const relevanceScore = this._computeRelevance(memory, query);
      if (relevanceScore > 0.3) {
        results.push({
          ...memory,
          relevanceScore: relevanceScore * weight
        });
      }
    }
    
    return results;
  }

  /**
   * Compute relevance score
   */
  _computeRelevance(memory, query) {
    // Safety check: validate inputs
    if (!memory || !query || typeof query !== 'string') {
      return 0;
    }
    
    const queryLower = query.toLowerCase();
    const contentLower = (memory.content || '').toLowerCase();
    const keyLower = (memory.key || '').toLowerCase();
    
    // Exact match
    if (keyLower === queryLower) return 1.0;
    if (contentLower.includes(queryLower)) return 0.8;
    
    // Keyword overlap
    const queryWords = new Set(queryLower.split(/\s+/));
    const contentWords = new Set(contentLower.split(/\s+/));
    const overlap = [...queryWords].filter(w => contentWords.has(w)).length;
    const overlapScore = overlap / Math.max(queryWords.size, 1);
    
    return overlapScore * 0.6;
  }

  /**
   * Consolidate memory (merge similar memories)
   */
  consolidateMemory() {
    
    // Consolidate long-term memory
    const consolidated = new Map();
    const toRemove = [];
    
    for (const [key, memory] of this.longTermMemory.entries()) {
      // Check for similar memories
      let merged = false;
      
      for (const [existingKey, existingMemory] of consolidated.entries()) {
        const similarity = this._computeSimilarity(memory, existingMemory);
        
        if (similarity > 0.8) {
          // Merge memories
          existingMemory.content += '\n' + memory.content;
          existingMemory.metadata.confidence = Math.max(
            existingMemory.metadata.confidence,
            memory.metadata.confidence
          );
          existingMemory.metadata.accessCount += memory.metadata.accessCount;
          merged = true;
          break;
        }
      }
      
      if (!merged) {
        consolidated.set(key, memory);
      }
    }
    
    this.longTermMemory = consolidated;
    
    // Clean up low-confidence memories
    this._cleanupLowConfidence();
    
  }

  /**
   * Compute similarity between memories
   */
  _computeSimilarity(mem1, mem2) {
    // Safety check: validate inputs
    if (!mem1 || !mem2 || !mem1.content || !mem2.content) {
      return 0;
    }
    
    const content1 = (mem1.content || '').toLowerCase();
    const content2 = (mem2.content || '').toLowerCase();
    
    // Simple word overlap
    const words1 = new Set(content1.split(/\s+/).filter(w => w.length > 0));
    const words2 = new Set(content2.split(/\s+/).filter(w => w.length > 0));
    const intersection = [...words1].filter(w => words2.has(w)).length;
    const union = new Set([...words1, ...words2]).size;
    
    // Safety check: prevent division by zero
    if (union === 0) {
      return 0;
    }
    
    return intersection / union;
  }

  /**
   * Clean up low-confidence memories
   */
  _cleanupLowConfidence() {
    for (const [key, memory] of this.longTermMemory.entries()) {
      const age = (Date.now() - memory.metadata.timestamp) / (1000 * 60 * 60 * 24); // Days
      const decayedConfidence = memory.metadata.confidence * Math.pow(this.decayRate, age);
      
      if (decayedConfidence < this.minConfidence && memory.metadata.accessCount < 3) {
        this.longTermMemory.delete(key);
      }
    }
  }

  /**
   * Update memory access
   */
  updateAccess(memory) {
    // Safety check: validate input
    if (!memory || !memory.metadata) {
      return;
    }
    
    memory.metadata.accessCount++;
    memory.metadata.lastAccessed = Date.now();
    
    // Boost confidence on frequent access
    if (memory.metadata.accessCount > 5) {
      memory.metadata.confidence = Math.min(1.0, memory.metadata.confidence + 0.05);
    }
  }

  /**
   * Get memory statistics
   */
  getStats() {
    return {
      longTerm: this.longTermMemory.size,
      user: Array.from(this.userMemory.values()).reduce((sum, map) => sum + map.size, 0),
      task: this.taskMemory.size,
      working: this.workingMemory.length
    };
  }
}

module.exports = EpsilonMemoryEngine;


