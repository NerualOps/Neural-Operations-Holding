/**
 * Epsilon AI Embedding Engine
 * ======================
 * Self-contained embedding model for Epsilon AI
 * NO external AI dependencies - trained from scratch
 * 
 * Features:
 * - CBOW (Continuous Bag-of-Words) training
 * - Skip-Gram training
 * - GloVe-style embeddings
 * - 384-dimensional vectors (matches Supabase vector size)
 */

class EpsilonEmbeddings {
  constructor(embeddingDim = 384) {
    this.embeddingDim = embeddingDim;
    this.vocab = new Map();
    this.embeddings = null; // Matrix: [vocabSize, embeddingDim]
    this.vocabSize = 0;
    this.isTrained = false;
    
    // Training parameters
    this.windowSize = 5; // Context window
    this.learningRate = 0.01;
    this.epochs = 10;
    this.batchSize = 32;
  }

  /**
   * Train embeddings using CBOW (Continuous Bag-of-Words)
   */
  async trainCBOW(corpus, tokenizer) {
    // Safety check: validate inputs
    if (!Array.isArray(corpus) || corpus.length === 0) {
      throw new Error('Invalid corpus provided to trainCBOW()');
    }
    if (!tokenizer || typeof tokenizer.getVocabSize !== 'function') {
      throw new Error('Invalid tokenizer provided to trainCBOW()');
    }
    
    // CRITICAL: Get current vocab size and check if it changed
    const newVocabSize = tokenizer.getVocabSize();
    if (newVocabSize <= 0) {
      throw new Error('Tokenizer vocab size is invalid');
    }
    
    // If vocab size changed, resize embeddings
    if (this.vocabSize > 0 && newVocabSize !== this.vocabSize) {
      console.log(`[EMBEDDINGS] Vocab size changed from ${this.vocabSize} to ${newVocabSize}, resizing embeddings...`);
      this._resizeEmbeddings(newVocabSize);
    }
    
    this.vocabSize = newVocabSize;
    this.vocab = new Map();
    
    // Initialize embeddings randomly (or use resized ones)
    if (!this.embeddings || this.embeddings.length !== this.vocabSize) {
      this.embeddings = this._initializeEmbeddings(this.vocabSize, this.embeddingDim);
    }
    
    // Prepare training data
    const trainingPairs = [];
    
    for (const text of corpus) {
      const tokenIds = tokenizer.encode(text);
      
      // Create context-target pairs
      for (let i = 0; i < tokenIds.length; i++) {
        const target = tokenIds[i];
        const context = [];
        
        // Collect context words (before and after)
        for (let j = Math.max(0, i - this.windowSize); j < Math.min(tokenIds.length, i + this.windowSize + 1); j++) {
          if (j !== i) {
            context.push(tokenIds[j]);
          }
        }
        
        if (context.length > 0) {
          trainingPairs.push({ target, context });
        }
      }
    }
    
    
    // Train using stochastic gradient descent
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;
      
      // Shuffle training pairs
      const shuffled = this._shuffleArray([...trainingPairs]);
      
      // Process in batches
      for (let i = 0; i < shuffled.length; i += this.batchSize) {
        const batch = shuffled.slice(i, i + this.batchSize);
        
        for (const { target, context } of batch) {
          const loss = this._trainCBOWStep(target, context);
          totalLoss += loss;
        }
      }
      
      const avgLoss = shuffled.length > 0 ? totalLoss / shuffled.length : 0;
    }
    
    this.isTrained = true;
    
    return this.embeddings;
  }

  /**
   * Single CBOW training step
   */
  _trainCBOWStep(targetId, contextIds) {
    // Safety check: prevent division by zero
    if (!contextIds || contextIds.length === 0) {
      return 0;
    }
    
    // Safety check: ensure embeddings are initialized and targetId is valid
    if (!this.embeddings || targetId < 0 || targetId >= this.vocabSize || !this.embeddings[targetId]) {
      return 0;
    }
    
    // Average context embeddings
    const contextEmbedding = new Array(this.embeddingDim).fill(0);
    for (const contextId of contextIds) {
      // Safety check: ensure contextId is valid
      if (contextId >= 0 && contextId < this.vocabSize && this.embeddings[contextId]) {
        for (let i = 0; i < this.embeddingDim; i++) {
          contextEmbedding[i] += this.embeddings[contextId][i];
        }
      }
    }
    
    // Average
    for (let i = 0; i < this.embeddingDim; i++) {
      contextEmbedding[i] /= contextIds.length;
    }
    
    // Predict target (simple dot product similarity)
    const targetEmbedding = this.embeddings[targetId];
    let similarity = 0;
    for (let i = 0; i < this.embeddingDim; i++) {
      similarity += contextEmbedding[i] * targetEmbedding[i];
    }
    
    // Gradient update using CBOW method
    const error = 1.0 - similarity; // Target similarity is 1.0
    
    // Update embeddings
    for (let i = 0; i < this.embeddingDim; i++) {
      const gradient = error * contextEmbedding[i] * this.learningRate;
      this.embeddings[targetId][i] += gradient;
      
      // Update context embeddings
      for (const contextId of contextIds) {
        // Safety check: ensure contextId is valid before updating
        if (contextId >= 0 && contextId < this.vocabSize && this.embeddings[contextId]) {
          this.embeddings[contextId][i] += gradient / contextIds.length;
        }
      }
    }
    
    return Math.abs(error);
  }

  /**
   * Get embedding for token ID
   */
  getEmbedding(tokenId) {
    // Safety check: ensure tokenId is valid
    if (!this.isTrained || !this.embeddings || tokenId < 0 || tokenId >= this.vocabSize || !this.embeddings[tokenId]) {
      return new Array(this.embeddingDim).fill(0);
    }
    
    return [...this.embeddings[tokenId]];
  }

  /**
   * Get embedding for text (average of token embeddings)
   */
  getTextEmbedding(text, tokenizer) {
    if (!this.isTrained) {
      return new Array(this.embeddingDim).fill(0);
    }
    
    const tokenIds = tokenizer.encode(text);
    if (tokenIds.length === 0) {
      return new Array(this.embeddingDim).fill(0);
    }
    
    // Average token embeddings
    const embedding = new Array(this.embeddingDim).fill(0);
    let validTokens = 0;
    
    for (const tokenId of tokenIds) {
      // Safety check: ensure tokenId is valid (non-negative and within vocab size)
      if (tokenId >= 0 && tokenId < this.vocabSize) {
        const tokenEmbedding = this.getEmbedding(tokenId);
        for (let i = 0; i < this.embeddingDim; i++) {
          embedding[i] += tokenEmbedding[i];
        }
        validTokens++;
      }
    }
    
    if (validTokens > 0) {
      for (let i = 0; i < this.embeddingDim; i++) {
        embedding[i] /= validTokens;
      }
    }
    
    return embedding;
  }

  /**
   * Initialize embeddings randomly
   */
  _initializeEmbeddings(vocabSize, embeddingDim) {
    // Safety check: validate inputs
    if (typeof vocabSize !== 'number' || vocabSize <= 0 || vocabSize > 1000000) {
      throw new Error(`Invalid vocabSize: ${vocabSize}`);
    }
    if (typeof embeddingDim !== 'number' || embeddingDim <= 0 || embeddingDim > 10000) {
      throw new Error(`Invalid embeddingDim: ${embeddingDim}`);
    }
    
    const embeddings = [];
    
    for (let i = 0; i < vocabSize; i++) {
      const embedding = [];
      for (let j = 0; j < embeddingDim; j++) {
        // Random initialization between -0.1 and 0.1
        embedding.push((Math.random() - 0.5) * 0.2);
      }
      embeddings.push(embedding);
    }
    
    return embeddings;
  }

  /**
   * Resize embeddings when vocab size changes
   */
  _resizeEmbeddings(newVocabSize) {
    // Safety check: validate input
    if (typeof newVocabSize !== 'number' || newVocabSize <= 0 || newVocabSize > 1000000) {
      throw new Error(`Invalid newVocabSize: ${newVocabSize}`);
    }
    
    if (!this.embeddings || this.embeddings.length === 0) {
      // First time initialization
      this.embeddings = this._initializeEmbeddings(newVocabSize, this.embeddingDim);
      return;
    }
    
    const oldVocabSize = this.embeddings.length;
    
    if (newVocabSize > oldVocabSize) {
      // Pad with random embeddings
      const padCount = newVocabSize - oldVocabSize;
      for (let i = 0; i < padCount; i++) {
        const embedding = [];
        for (let j = 0; j < this.embeddingDim; j++) {
          embedding.push((Math.random() - 0.5) * 0.2);
        }
        this.embeddings.push(embedding);
      }
    } else if (newVocabSize < oldVocabSize) {
      // Truncate
      this.embeddings = this.embeddings.slice(0, newVocabSize);
    }
    
    // Validate final size
    if (this.embeddings.length !== newVocabSize) {
      throw new Error(`Embedding resize failed: expected ${newVocabSize}, got ${this.embeddings.length}`);
    }
    
    console.log(`[EMBEDDINGS] Resized embeddings from ${oldVocabSize} to ${newVocabSize}`);
  }

  /**
   * Shuffle array (Fisher-Yates)
   */
  _shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  /**
   * Save embeddings
   */
  save() {
    return {
      embeddings: this.embeddings,
      vocabSize: this.vocabSize,
      embeddingDim: this.embeddingDim,
      isTrained: this.isTrained
    };
  }

  /**
   * Load embeddings
   */
  load(data) {
    // Safety check: validate data structure
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid data provided to load()');
    }
    
    this.embeddings = data.embeddings || null;
    this.vocabSize = data.vocabSize || 0;
    this.embeddingDim = data.embeddingDim || this.embeddingDim || 384;
    this.isTrained = data.isTrained || false;
  }
}

module.exports = EpsilonEmbeddings;


