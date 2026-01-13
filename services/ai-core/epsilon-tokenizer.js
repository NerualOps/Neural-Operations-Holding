/**
 * Epsilon AI Tokenizer
 * ==============
 * Self-contained WordPiece tokenizer for Epsilon AI
 * NO external AI dependencies - built from scratch
 * 
 * Features:
 * - WordPiece encoding (subword tokenization)
 * - Vocabulary building from corpus
 * - Encoding/decoding
 * - Special tokens (PAD, UNK, CLS, SEP, MASK)
 */

class EpsilonTokenizer {
  constructor() {
    this.vocab = new Map();
    this.invVocab = new Map();
    this.wordPieces = new Map();
    this.maxVocabSize = 40000; // Matches transformer vocab_size
    this.unkToken = '[UNK]';
    this.padToken = '[PAD]';
    this.clsToken = '[CLS]';
    this.sepToken = '[SEP]';
    this.maskToken = '[MASK]';
    
    // Initialize with special tokens
    this.specialTokens = [
      this.unkToken,
      this.padToken,
      this.clsToken,
      this.sepToken,
      this.maskToken
    ];
    
    this.vocabSize = 0;
    this.isTrained = false;
  }

  /**
   * Build vocabulary from corpus
   */
  buildVocab(corpus, minFrequency = 2) {
    // Safety check: validate input
    if (!Array.isArray(corpus) || corpus.length === 0) {
      throw new Error('Invalid corpus provided to buildVocab()');
    }
    if (typeof minFrequency !== 'number' || minFrequency < 1) {
      minFrequency = 2;
    }
    
    // Limit corpus size to prevent DoS
    const maxCorpusSize = 1000000; // 1M texts max
    if (corpus.length > maxCorpusSize) {
      console.warn(`[TOKENIZER] Corpus too large (${corpus.length} texts), limiting to ${maxCorpusSize}`);
      corpus = corpus.slice(0, maxCorpusSize);
    }
    
    // Count word frequencies
    const wordFreq = new Map();
    const words = [];
    
    for (const text of corpus) {
      // Validate text
      if (typeof text !== 'string') {
        continue;
      }
      // Limit text length to prevent DoS
      const maxTextLength = 100000; // 100KB per text
      const safeText = text.length > maxTextLength ? text.substring(0, maxTextLength) : text;
      
      const tokens = this._basicTokenize(safeText);
      for (const token of tokens) {
        // Limit token length
        if (token.length > 200) {
          continue; // Skip extremely long tokens
        }
        wordFreq.set(token, (wordFreq.get(token) || 0) + 1);
        words.push(token);
      }
    }
    
    // Filter by frequency
    const filteredWords = Array.from(wordFreq.entries())
      .filter(([word, freq]) => freq >= minFrequency)
      .sort((a, b) => b[1] - a[1]); // Sort by frequency
    
    // Add special tokens first
    let tokenId = 0;
    for (const specialToken of this.specialTokens) {
      this.vocab.set(specialToken, tokenId);
      this.invVocab.set(tokenId, specialToken);
      tokenId++;
    }
    
    // Store old vocab size for resizing check
    const oldVocabSize = this.vocabSize;
    
    // Add most frequent words
    for (const [word, freq] of filteredWords) {
      if (tokenId >= this.maxVocabSize) break;
      
      this.vocab.set(word, tokenId);
      this.invVocab.set(tokenId, word);
      tokenId++;
    }
    
    // Build WordPiece subwords
    this._buildWordPieces();
    
    this.vocabSize = tokenId;
    this.isTrained = true;
    
    // Notify if vocab size changed (for transformer resizing)
    if (oldVocabSize > 0 && this.vocabSize !== oldVocabSize) {
      console.log(`[TOKENIZER] Vocabulary size changed from ${oldVocabSize} to ${this.vocabSize}`);
    }
    
    return this.vocabSize;
  }

  /**
   * Build WordPiece subwords
   */
  _buildWordPieces() {
    // Simple WordPiece: split words into subwords
    // Format: word -> [subword1, subword2, ...]
    for (const [word, tokenId] of this.vocab.entries()) {
      if (this.specialTokens.includes(word)) continue;
      
      // Split into character-level subwords
      const subwords = [];
      if (word.length > 3) {
        // Split long words: "running" -> ["run", "##ning"]
        const splitPoint = Math.floor(word.length / 2);
        if (splitPoint > 0 && splitPoint < word.length) {
          const prefix = word.substring(0, splitPoint);
          const suffix = '##' + word.substring(splitPoint);
          subwords.push(prefix, suffix);
        } else {
          subwords.push(word);
        }
      } else {
        subwords.push(word);
      }
      
      this.wordPieces.set(word, subwords);
    }
  }

  /**
   * Basic tokenization (split on whitespace and punctuation)
   */
  _basicTokenize(text) {
    if (!text || typeof text !== 'string') return [];
    
    // Normalize text
    text = text.toLowerCase()
      .replace(/[^\w\s]/g, ' $& ') // Separate punctuation
      .replace(/\s+/g, ' ')
      .trim();
    
    return text.split(/\s+/).filter(t => t.length > 0);
  }

  /**
   * Encode text to token IDs
   */
  encode(text, maxLength = null) {
    if (!text || typeof text !== 'string') return [];
    
    const tokens = this._basicTokenize(text);
    const tokenIds = [];
    
    for (const token of tokens) {
      // Try exact match first
      if (this.vocab.has(token)) {
        tokenIds.push(this.vocab.get(token));
      } else {
        // Try WordPiece subwords
        const subwords = this.wordPieces.get(token) || [token];
        let found = false;
        
        for (const subword of subwords) {
          if (this.vocab.has(subword)) {
            tokenIds.push(this.vocab.get(subword));
            found = true;
          }
        }
        
        // If no match, use UNK
        if (!found) {
          const unkId = this.vocab.get(this.unkToken);
          if (unkId !== undefined) {
            tokenIds.push(unkId);
          } else {
            // Fallback if vocab not initialized
            tokenIds.push(0);
          }
        }
      }
    }
    
    // Truncate or pad to maxLength
    if (maxLength !== null) {
      if (tokenIds.length > maxLength) {
        tokenIds.splice(maxLength);
      } else {
        const padId = this.vocab.get(this.padToken);
        const padTokenId = padId !== undefined ? padId : 1; // Fallback to 1 if not initialized
        while (tokenIds.length < maxLength) {
          tokenIds.push(padTokenId);
        }
      }
    }
    
    return tokenIds;
  }

  /**
   * Decode token IDs to text
   */
  decode(tokenIds) {
    // Safety check: validate input
    if (!Array.isArray(tokenIds)) {
      return '';
    }
    
    const tokens = [];
    
    for (const tokenId of tokenIds) {
      if (this.invVocab.has(tokenId)) {
        const token = this.invVocab.get(tokenId);
        if (token !== this.padToken && token !== this.clsToken && token !== this.sepToken) {
          tokens.push(token);
        }
      }
    }
    
    // Reconstruct text
    let text = tokens.join(' ');
    
    // Clean up WordPiece markers
    text = text.replace(/##/g, '');
    
    return text;
  }

  /**
   * Get vocabulary size
   */
  getVocabSize() {
    return this.vocabSize || this.vocab.size;
  }

  /**
   * Save tokenizer to JSON
   */
  save() {
    return {
      vocab: Array.from(this.vocab.entries()),
      wordPieces: Array.from(this.wordPieces.entries()),
      vocabSize: this.vocabSize,
      isTrained: this.isTrained
    };
  }

  /**
   * Load tokenizer from JSON
   */
  load(data) {
    // Safety check: validate input
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid data provided to load()');
    }
    
    // Validate vocab structure
    if (!Array.isArray(data.vocab)) {
      throw new Error('Invalid vocab structure in load()');
    }
    
    this.vocab = new Map(data.vocab || []);
    this.wordPieces = new Map(data.wordPieces || []);
    this.vocabSize = data.vocabSize || this.vocab.size;
    this.isTrained = data.isTrained || false;
    
    // Rebuild inverse vocab
    this.invVocab = new Map();
    for (const [word, tokenId] of this.vocab.entries()) {
      this.invVocab.set(tokenId, word);
    }
  }
}

module.exports = EpsilonTokenizer;


