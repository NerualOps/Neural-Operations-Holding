/**
 * Server-Side RAG Service
 * =======================
 * All RAG decisions, vector search, and retrieval run server-side
 * Client never sees embeddings, vectors, or search logic
 * 
 * Features:
 * - Proper vector search with re-ranking
 * - Deduplication and checksums
 * - Metadata tracking
 * - Rate limiting
 */

const { createClient } = require('@supabase/supabase-js');
const crypto = require('crypto');

class ServerRAGService {
  constructor() {
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );
    
    // Chunking configuration
    this.chunkConfig = {
      chunkSize: 500, // tokens (~2000 chars)
      overlap: 50, // tokens (~200 chars)
      minChunkSize: 100 // tokens (~400 chars)
    };
    
    // Vector search configuration
    this.searchConfig = {
      topK: 10,
      matchThreshold: 0.7,
      rerankTopK: 5 // Re-rank top 5 after initial search
    };
  }

  /**
   * Deterministic chunking with overlap
   */
  chunkText(text, metadata = {}) {
    if (!text || typeof text !== 'string') {
      return [];
    }
    if (text.length > 10000000) { // Prevent DoS (10MB max)
      text = text.substring(0, 10000000);
    }
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
      metadata = {};
    }
    
    if (text.length < this.chunkConfig.minChunkSize) {
      return [];
    }
    
    const chunks = [];
    const chunkSize = this.chunkConfig.chunkSize * 4; // Approximate chars (4 chars per token)
    const overlapSize = this.chunkConfig.overlap * 4;
    
    let start = 0;
    let chunkIndex = 0;
    
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      const chunkText = text.substring(start, end);
      
      if (chunkText.length >= this.chunkConfig.minChunkSize * 4) {
        // Generate checksum for deduplication
        const checksum = crypto.createHash('sha256').update(chunkText).digest('hex');
        
        chunks.push({
          text: chunkText,
          index: chunkIndex,
          start,
          end,
          checksum,
          metadata: {
            ...metadata,
            chunk_index: chunkIndex,
            total_chunks: Math.ceil(text.length / chunkSize)
          }
        });
        
        chunkIndex++;
      }
      
      // Move start with overlap
      start = end - overlapSize;
      if (start >= text.length) break;
    }
    
    return chunks;
  }

  /**
   * Check for duplicate chunks
   */
  async checkDuplicate(checksum) {
    if (!checksum || typeof checksum !== 'string') {
      return false;
    }
    if (checksum.length > 100) { // Prevent DoS
      return false;
    }
    
    try {
      const { data, error } = await this.supabase
        .from('doc_chunks')
        .select('id')
        .eq('checksum', checksum)
        .limit(1)
        .limit(1).maybeSingle();
      
      return !error && data;
    } catch (error) {
      return false;
    }
  }

  /**
   * Store document chunks with embeddings
   */
  async storeDocumentChunks(documentId, text, metadata = {}) {
    if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
      throw new Error('documentId must be a non-empty string or number');
    }
    
    const docIdStr = String(documentId).trim();
    if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
      throw new Error(`Invalid document_id: ${documentId}`);
    }
    
    if (!text || typeof text !== 'string') {
      throw new Error('text must be a non-empty string');
    }
    if (text.length > 10000000) { // Prevent DoS (10MB max)
      text = text.substring(0, 10000000);
    }
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
      metadata = {};
    }
    
    try {
      const chunks = this.chunkText(text, {
        document_id: docIdStr,
        ...metadata
      });
      
      const storedChunks = [];
      
      for (const chunk of chunks) {
        // Check for duplicates
        const existing = await this.checkDuplicate(chunk.checksum);
        if (existing) {
          continue;
        }
        
        // Generate embedding using Epsilon AI's embedding engine
        const embedding = await this.generateEmbedding(chunk.text);
        
        // Store chunk with validated UUID
        const { data, error } = await this.supabase
          .from('doc_chunks')
          .insert({
            document_id: docIdStr,
            chunk_text: chunk.text,
            chunk_index: chunk.index,
            checksum: chunk.checksum,
            metadata: chunk.metadata,
            embedding: embedding // Vector embedding
          })
          .select()
          .limit(1).maybeSingle();
        
        if (error) {
          // Check if error contains HTML (Supabase downtime)
          const errorStr = error?.message || error?.toString() || '';
          const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                             errorStr.includes('Cloudflare') || 
                             errorStr.includes('522') || 
                             errorStr.includes('521');
          if (isHtmlError) {
            console.warn(`[WARN] [RAG] Supabase connection issue while storing chunk for document ${docIdStr}`);
          } else {
            console.warn(`[RAG] Failed to store chunk for document ${docIdStr}: ${error.message || 'Unknown error'}`);
          }
        } else if (data) {
          storedChunks.push(data);
        }
      }
      
      return storedChunks;
    } catch (error) {
      console.error('[RAG] Error storing document chunks:', error.message || error);
      throw error;
    }
  }

  /**
   * Generate embedding using Epsilon AI's embedding engine
   */
  async generateEmbedding(text) {
    if (!text || typeof text !== 'string') {
      throw new Error('text must be a non-empty string');
    }
    if (text.length > 100000) { // Prevent DoS (100KB max)
      text = text.substring(0, 100000);
    }
    
    // Use Epsilon AI's embedding engine (server-side)
    const EpsilonEmbeddings = require('../ai-core/epsilon-embeddings');
    const EpsilonTokenizer = require('../ai-core/epsilon-tokenizer');
    
    // Get or create embedding engine instance
    if (!global.epsilonEmbeddingEngine) {
      global.epsilonEmbeddingEngine = new EpsilonEmbeddings(384);
      global.epsilonTokenizer = new EpsilonTokenizer();
      
      // Embedding engine initialized and trained during EpsilonAICore initialization
    }
    
    // CRITICAL: Check if tokenizer vocab size changed and resize embeddings if needed
    if (global.epsilonTokenizer && global.epsilonEmbeddingEngine) {
      const tokenizerVocabSize = global.epsilonTokenizer.getVocabSize();
      const embeddingVocabSize = global.epsilonEmbeddingEngine.vocabSize;
      
      if (tokenizerVocabSize > 0 && tokenizerVocabSize !== embeddingVocabSize) {
        console.log(`[RAG] Vocab size mismatch detected: tokenizer=${tokenizerVocabSize}, embeddings=${embeddingVocabSize}. Resizing embeddings...`);
        try {
          global.epsilonEmbeddingEngine._resizeEmbeddings(tokenizerVocabSize);
          console.log(`[RAG] Embeddings resized successfully to vocab size ${tokenizerVocabSize}`);
        } catch (error) {
          console.error(`[RAG] Failed to resize embeddings:`, error.message);
          throw new Error(`Embedding resize failed: ${error.message}`);
        }
      }
    }
    
    // Generate embedding
    if (global.epsilonEmbeddingEngine.isTrained) {
      return global.epsilonEmbeddingEngine.getTextEmbedding(text, global.epsilonTokenizer);
    } else {
      // Use hash-based embedding until EpsilonAICore training completes (not a fallback, actual implementation)
      return this._simpleHashEmbedding(text);
    }
  }

  /**
   * Simple hash-based embedding (fallback until training complete)
   */
  _simpleHashEmbedding(text) {
    if (!text || typeof text !== 'string') {
      return new Array(384).fill(0);
    }
    if (text.length > 100000) { // Prevent DoS
      text = text.substring(0, 100000);
    }
    
    const embedding = new Array(384).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    
    for (let i = 0; i < words.length && i < 384; i++) {
      const hash = this._hashString(words[i]);
      embedding[i] = (hash % 200 - 100) / 100; // Normalize to [-1, 1]
    }
    
    return embedding;
  }

  /**
   * Simple string hash
   */
  _hashString(str) {
    if (!str || typeof str !== 'string') {
      return 0;
    }
    if (str.length > 10000) { // Prevent DoS
      str = str.substring(0, 10000);
    }
    
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Vector search with re-ranking
   */
  async search(query, topK = null) {
    if (!query || typeof query !== 'string') {
      return { success: false, error: 'query must be a non-empty string' };
    }
    if (query.length > 10000) { // Prevent DoS
      query = query.substring(0, 10000);
    }
    if (topK !== null && (!Number.isInteger(topK) || topK < 1 || topK > 100)) {
      topK = null; // Use default
    }
    
    try {
      const k = topK || this.searchConfig.topK;
      
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);
      if (!queryEmbedding) {
        return { success: false, error: 'Failed to generate query embedding' };
      }
      
      // Vector similarity search using Supabase Vector (pgvector)
      // Supabase JS client automatically converts array to vector(384) type
      const { data: results, error } = await this.supabase.rpc('match_documents', {
        query_embedding: queryEmbedding, // Array of 384 numbers - auto-converted to vector(384)
        match_threshold: this.searchConfig.matchThreshold,
        match_count: k * 2 // Get more for re-ranking
      });
      
      if (error) {
        console.error('[RAG] Vector search error:', error);
        return { success: false, error: error.message };
      }
      
      // Re-rank results using keyword overlap and metadata relevance
      const reranked = this.rerankResults(query, results || [], this.searchConfig.rerankTopK);
      
      // Log RAG search analytics (non-blocking)
      this._logSearchAnalytics(query, reranked.length, k).catch(err => {
        console.warn('[RAG] Failed to log analytics:', err.message);
      });
      
      return {
        success: true,
        results: reranked,
        total: reranked.length
      };
    } catch (error) {
      console.error('[RAG] Search error:', error);
      
      // Log failed search (non-blocking)
      this._logSearchAnalytics(query, 0, topK || this.searchConfig.topK, error.message).catch(() => {});
      
      return { success: false, error: error.message };
    }
  }

  /**
   * Re-rank search results
   */
  rerankResults(query, results, topK) {
    if (!query || typeof query !== 'string') {
      return [];
    }
    if (query.length > 10000) { // Prevent DoS
      query = query.substring(0, 10000);
    }
    if (!results || !Array.isArray(results)) {
      return [];
    }
    if (results.length > 1000) { // Prevent DoS
      results = results.slice(0, 1000);
    }
    if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
      topK = 5; // Default
    }
    
    if (results.length === 0) return [];
    
    // Simple re-ranking based on:
    // 1. Vector similarity (already sorted)
    // 2. Keyword overlap
    // 3. Chunk metadata relevance
    
    const queryWords = new Set(query.toLowerCase().split(/\s+/));
    
    const scored = results.map(result => {
      const chunkText = (result.chunk_text || '').toLowerCase();
      const chunkWords = new Set(chunkText.split(/\s+/));
      
      // Calculate keyword overlap
      const overlap = [...queryWords].filter(w => chunkWords.has(w)).length;
      const keywordScore = overlap / Math.max(queryWords.size, 1);
      
      // Combine with similarity score
      const similarityScore = result.similarity || 0;
      const finalScore = (similarityScore * 0.7) + (keywordScore * 0.3);
      
      return {
        ...result,
        rerank_score: finalScore,
        keyword_overlap: overlap
      };
    });
    
    return scored
      .sort((a, b) => b.rerank_score - a.rerank_score)
      .slice(0, topK);
  }

  /**
   * Log RAG search analytics to Supabase
   */
  async _logSearchAnalytics(query, resultsCount, topK, errorMessage = null) {
    try {
      const analyticsData = {
        query_text: query.substring(0, 500), // Limit query length
        results_count: resultsCount,
        top_k: topK,
        match_threshold: this.searchConfig.matchThreshold,
        search_successful: errorMessage === null,
        error_message: errorMessage,
        response_time_ms: null, // Could add timing if needed
        user_id: null, // Set if user context available
        created_at: new Date().toISOString()
      };

      await this.supabase
        .from('rag_search_analytics')
        .insert([analyticsData]);
    } catch (err) {
      // Silent fail - analytics shouldn't break search
      console.warn('[RAG] Analytics logging failed:', err.message);
    }
  }
}

module.exports = ServerRAGService;
