/**
 * Epsilon AI Inference Client
 * Single source of truth for model inference calls
 * © 2025 Neural Operation's & Holding's LLC. All rights reserved.
 */

const axios = require('axios');

class InferenceClient {
  constructor() {
    // Get inference URL from environment or default to local
    this.inferenceUrl = process.env.INFERENCE_URL || 'http://127.0.0.1:8005';
    this.timeout = parseInt(process.env.INFERENCE_TIMEOUT || '30000', 10); // 30s default - faster timeout
    this.ready = false;
    this.modelInfo = null;
  }

  /**
   * Check if inference service is healthy and ready
   */
  async checkHealth() {
    try {
      const response = await axios.get(`${this.inferenceUrl}/health`, {
        timeout: 2000  // Faster health check - 2 seconds max
      });
      
      if (response.data && response.data.model_loaded === true) {
        this.ready = true;
        return true;
      }
      
      this.ready = false;
      return false;
    } catch (error) {
      this.ready = false;
      // Only log if it's not a connection error (to reduce noise)
      if (!error.code || (error.code !== 'ECONNREFUSED' && error.code !== 'ETIMEDOUT')) {
        console.warn('[INFERENCE CLIENT] Health check failed:', error.message);
      }
      return false;
    }
  }

  /**
   * Get model information
   */
  async getModelInfo() {
    try {
      const response = await axios.get(`${this.inferenceUrl}/model-info`, {
        timeout: 10000
      });
      
      this.modelInfo = response.data;
      return response.data;
    } catch (error) {
      console.warn('[INFERENCE CLIENT] Failed to get model info:', error.message);
      return null;
    }
  }

  /**
   * Generate text from prompt
   * 
   * @param {Object} options - Generation options
   * @param {string} options.prompt - Input prompt
   * @param {number} options.max_new_tokens - Maximum tokens to generate (default: 256)
   * @param {number} options.temperature - Sampling temperature (default: 0.7)
   * @param {number} options.top_p - Top-p sampling (default: 0.9)
   * @param {string[]} options.stop - Stop sequences (optional)
   * @returns {Promise<Object|null>} Response with text, model_id, and tokens
   */
  async generate(options = {}) {
    if (!this.ready) {
      // Try to check health first
      const isHealthy = await this.checkHealth();
      if (!isHealthy) {
        console.warn('[INFERENCE CLIENT] Service not ready, cannot generate');
        return null;
      }
    }

    const {
      prompt,
      max_new_tokens = 75,  // Default to faster responses
      temperature = 0.7,
      top_p = 0.9,
      stop = null,
      repetition_penalty = 1.3
    } = options;

    if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
      console.warn('[INFERENCE CLIENT] Invalid prompt provided');
      return null;
    }

    try {
      const requestPayload = {
        prompt: prompt.trim(),
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty
      };

      if (stop && Array.isArray(stop) && stop.length > 0) {
        requestPayload.stop = stop;
      }

      const response = await axios.post(
        `${this.inferenceUrl}/generate`,
        requestPayload,
        {
          timeout: this.timeout,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.data && response.data.text) {
        return {
          text: response.data.text,
          model_id: response.data.model_id || 'unknown',
          tokens: response.data.tokens || { prompt: 0, completion: 0 }
        };
      }

      console.warn('[INFERENCE CLIENT] Invalid response format from inference service');
      return null;
    } catch (error) {
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        console.warn(`[INFERENCE CLIENT] Generation timed out after ${this.timeout}ms`);
      } else if (error.response) {
        const status = error.response.status || 'unknown';
        const detail = error.response.data?.detail || error.response.statusText || 'unknown error';
        console.warn(`[INFERENCE CLIENT] Generation failed: ${status} - ${detail}`);
      } else {
        console.warn(`[INFERENCE CLIENT] Generation failed: ${error.message}`);
      }
      
      return null;
    }
  }

  /**
   * Initialize client - check health and load model info
   */
  async initialize() {
    console.log('[INFERENCE CLIENT] Initializing inference client...');
    console.log(`[INFERENCE CLIENT] Inference URL: ${this.inferenceUrl}`);
    
    const isHealthy = await this.checkHealth();
    if (isHealthy) {
      await this.getModelInfo();
      console.log('[INFERENCE CLIENT] Inference client ready');
    } else {
      console.warn('[INFERENCE CLIENT] Inference service not available, will retry on first request');
    }
    
    return isHealthy;
  }
}

// Singleton instance
let inferenceClientInstance = null;

/**
 * Get or create inference client instance
 */
function getInferenceClient() {
  if (!inferenceClientInstance) {
    inferenceClientInstance = new InferenceClient();
  }
  return inferenceClientInstance;
}

module.exports = {
  InferenceClient,
  getInferenceClient
};

