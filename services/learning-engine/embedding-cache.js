/**
 * Embedding Cache
 * ===============
 * Caches embeddings for performance optimization
 */

class EmbeddingCache {
  constructor(maxSize = 10000, ttl = 24 * 60 * 60 * 1000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl; // 24 hours default
    this.hits = 0;
    this.misses = 0;
    this.accessTimes = new Map();
  }

  /**
   * Generate cache key from text
   */
  generateKey(text) {
    if (!text || typeof text !== 'string') {
      return null;
    }
    // Normalize text for consistent hashing
    const normalized = text.toLowerCase().trim().replace(/\s+/g, ' ');
    return this.hashString(normalized);
  }

  /**
   * Hash string to consistent key
   */
  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return `emb_${Math.abs(hash)}`;
  }

  /**
   * Get embedding from cache
   */
  get(text) {
    const key = this.generateKey(text);
    if (!key) return null;

    const cached = this.cache.get(key);
    if (!cached) {
      this.misses++;
      return null;
    }

    // Check TTL
    const age = Date.now() - cached.timestamp;
    if (age > this.ttl) {
      this.cache.delete(key);
      this.accessTimes.delete(key);
      this.misses++;
      return null;
    }

    // Update access time for LRU
    this.accessTimes.set(key, Date.now());
    this.hits++;
    return cached.embedding;
  }

  /**
   * Store embedding in cache
   */
  set(text, embedding) {
    const key = this.generateKey(text);
    if (!key || !embedding) return false;

    // Evict if cache is full (LRU)
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    this.cache.set(key, {
      embedding,
      timestamp: Date.now()
    });
    this.accessTimes.set(key, Date.now());

    return true;
  }

  /**
   * Evict least recently used entry
   */
  evictLRU() {
    if (this.accessTimes.size === 0) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
      }
      return;
    }

    // Find least recently used
    let lruKey = null;
    let lruTime = Infinity;

    for (const [key, time] of this.accessTimes.entries()) {
      if (time < lruTime) {
        lruTime = time;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
      this.accessTimes.delete(lruKey);
    }
  }

  /**
   * Clear cache
   */
  clear() {
    this.cache.clear();
    this.accessTimes.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const total = this.hits + this.misses;
    const hitRate = total > 0 ? (this.hits / total) * 100 : 0;

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: hitRate.toFixed(2) + '%',
      ttl: this.ttl
    };
  }

  /**
   * Clean expired entries
   */
  cleanExpired() {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, cached] of this.cache.entries()) {
      const age = now - cached.timestamp;
      if (age > this.ttl) {
        this.cache.delete(key);
        this.accessTimes.delete(key);
        cleaned++;
      }
    }

    return cleaned;
  }
}

// Export singleton instance
let instance = null;

function getEmbeddingCache(maxSize, ttl) {
  if (!instance) {
    instance = new EmbeddingCache(maxSize, ttl);
  }
  return instance;
}

module.exports = { EmbeddingCache, getEmbeddingCache };

