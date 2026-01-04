/**
 * Epsilon AI Language Engine
 * --------------------
 * - Collects training samples from Supabase (documents, chunks, conversations)
 * - Communicates with the internal Python language model service
 * - Provides persona-aware generation for Epsilon AI's responses
 */

const axios = require('axios');
const { createClient } = require('@supabase/supabase-js');
const EpsilonSelfLearning = require('./epsilon-self-learning');

// Use local logger instead of overriding global console
const _silent = () => {};
const _log = (message) => {
  // Only log if explicitly enabled via environment variable
  if (process.env.EPSILON_VERBOSE_LOGGING === '1') {
    console.log(message);
  }
};

function debounce(fn, delay) {
  let timer = null;
  return function debounced(...args) {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

class EpsilonLanguageEngine {
  constructor() {
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      {
        auth: { persistSession: false }
      }
    );

    this.pythonManager = null;
    this.modelStatus = {
      ready: false,
      lastTrainedAt: null,
      stats: {},
      pendingReason: null,
      version: '1.0.0',
      checkpointPath: null
    };

    // Training removed - all training is local-only in ml_local/
    
    // Initialize self-learning system
    this.selfLearning = new EpsilonSelfLearning(this);
  }

  attachPythonManager(manager) {
    this.pythonManager = manager;
  }

  isPythonReady() {
    return Boolean(
      this.pythonManager &&
      this.pythonManager.isServiceReady &&
      this.pythonManager.isServiceReady('language_model')
    );
  }

  isModelReady() {
    return this.modelStatus.ready;
  }

  enqueueTraining(reason = 'scheduled', context = {}) {
    // Training is now local-only in ml_local/
    console.warn('[EPSILON AI LANGUAGE ENGINE] Training is local-only. Use ml_local/train/pretrain.py for training.');
    return { success: false, error: 'Training is local-only. Use ml_local/train/pretrain.py' };
  }

  async trainNow(options = {}) {
    // Training is now local-only in ml_local/
    throw new Error('Training is local-only. Use ml_local/train/pretrain.py for training. This endpoint is disabled.');
  }

  async generate(options = {}) {
    // Use inference client for all generation
    const { getInferenceClient } = require('../runtime/inference_client');
    const inferenceClient = getInferenceClient();
    
    const prompt = (options.userMessage || '').trim();
    if (!prompt) {
      return null;
    }

    // Build persona hint for temperature adjustment (model has learned patterns)
    const personaHint = options.persona || this.buildPersonaHint(options.userMessage, []);
    
    try {
      const result = await inferenceClient.generate({
        prompt,
        max_new_tokens: Math.min(options.maxLength || 256, 256),
        temperature: personaHint?.mode === 'sales' ? 0.95 : 0.7,
        top_p: 0.9,
        stop: null // Can add stop sequences if needed
      });

      if (result && result.text) {
        let responseText = result.text;
        
        // Clean up malformed text patterns (common generation errors)
        responseText = responseText
          .replace(/\bwould love to I\b/gi, 'can I')
          .replace(/\bwould love to\s+([a-z]+)\s+I\b/gi, 'can I $1')
          .replace(/\bHow would love to\b/gi, 'How can')
          .replace(/\bWhat would love to\b/gi, 'What can')
          .trim();
        
        // Validate response quality
        const genericPatterns = [
          /^(what|how|why|when|where|who)\s+(is|are|do|does|can|will)/i,
          /^(i|i'm|i am)\s+(not sure|unsure|not certain)/i,
          /^(that|this)\s+(is|are)\s+(a|an)\s+(good|great|interesting)\s+(question|point)/i
        ];
        
        const isGeneric = genericPatterns.some(pattern => pattern.test(responseText.trim()));
        const isTooShort = responseText.length < 50;
        const hasNoSubstance = !responseText.match(/[.!?]/) || responseText.split(/[.!?]/).length < 2;
        
        if ((isGeneric || (isTooShort && hasNoSubstance)) && responseText.length < 100) {
          console.warn('[EPSILON AI LANGUAGE ENGINE] Generated response is too generic or lacks substance');
          return null;
        }
        
        return {
          text: responseText,
          meta: {
            model_id: result.model_id,
            tokens: result.tokens
          }
        };
      }
    } catch (error) {
      console.warn(`[EPSILON AI LANGUAGE ENGINE] Generation failed: ${error.message}`);
    }

    return null;
  }

  buildPersonaHint(userMessage = '', ragContext = []) {
    // ragContext ignored - model uses learned patterns
    // Persona is built from user message only - model has already learned tone/style from training
    const normalized = userMessage.toLowerCase();
    
    const persona = {
      mode: 'advisor',
      tone: 'neutral',
      energy: 'steady',
      stage: null, // Learned from training, not retrieved
      audience: null, // Learned from training, not retrieved
      urgency: 'normal',
      keyword_hits: []
    };

    // Detect sales intent from user message, not documents
    const hasSalesIntent = /(sales|sell|pitch|client|customer|prospect|close|deal)/i.test(userMessage);
    const hasCaseIntent = /(case|example|story|result|outcome)/i.test(userMessage);
    const hasTechIntent = /(api|integration|architecture|technical|code|implementation)/i.test(userMessage);

    if (normalized.includes('pitch') || normalized.includes('close') || normalized.includes('pricing') || hasSalesIntent) {
      persona.mode = 'sales';
      persona.tone = 'confident';
      persona.energy = 'upbeat';
      persona.call_to_action = 'Let me turn this into language you can use with the buyer.';
    } else if (hasTechIntent) {
      persona.mode = 'technical';
      persona.tone = 'precise';
      persona.energy = 'structured';
      persona.prefersBullets = true;
      persona.call_to_action = 'I can sketch the implementation plan or outline integration steps.';
    } else if (hasCaseIntent) {
      persona.mode = 'credibility';
      persona.tone = 'assured';
      persona.energy = 'steady';
      persona.call_to_action = 'Want me to package this into a quick story or follow-up note?';
    } else if (normalized.includes('onboarding')) {
      persona.mode = 'advisor';
      persona.tone = 'supportive';
      persona.energy = 'steady';
      persona.call_to_action = 'I can outline the rollout tasks or prep a kickoff note.';
    } else if (normalized.includes('support') || normalized.includes('incident')) {
      persona.mode = 'advisor';
      persona.tone = 'calm';
      persona.energy = 'steady';
      persona.stage = 'support';
      persona.call_to_action = 'Let me gather what happened and map the next steps.';
    }

    // Audience is determined from user message intent, not documents
    persona.audience = persona.audience || (
      persona.mode === 'technical' ? 'technical' :
      persona.mode === 'sales' ? 'executive' :
      'general'
    );

    return persona;
  }


  // Rollback to previous model version
  async rollbackToVersion(version) {
    if (!this.modelVersions.has(version)) {
      throw new Error(`Version ${version} not found`);
    }
    
    const versionData = this.modelVersions.get(version);
    this.currentVersion = version;
    this.modelStatus.version = version;
    this.modelStatus.checkpointPath = versionData.checkpointPath;
    this.lastCheckpoint = {
      checkpointPath: versionData.checkpointPath,
      version: version,
      timestamp: Date.now()
    };
    
    this.rollbackHistory.push({
      from: this.currentVersion,
      to: version,
      timestamp: new Date().toISOString()
    });
    
    _silent(`[EPSILON AI LANGUAGE ENGINE] Rolled back to version ${version}`);
    return { success: true, version: version };
  }

  // Get all model versions
  getModelVersions() {
    return Array.from(this.modelVersions.entries()).map(([version, data]) => ({
      version,
      ...data
    }));
  }

  // Increment version number (semantic versioning)
  _incrementVersion(currentVersion) {
    const parts = currentVersion.split('.');
    const major = parseInt(parts[0]) || 1;
    const minor = parseInt(parts[1]) || 0;
    const patch = parseInt(parts[2]) || 0;
    return `${major}.${minor}.${patch + 1}`;
  }

  // Removed: _collectTrainingDataset - training is local-only in ml_local/

  async _collectPreTrainingTexts(limit = null) {
    /**
     * Collect raw documents for transformer-based pre-training.
     * Pre-training uses self-supervised learning (predict next token) on raw text.
     * This teaches the model language patterns, grammar, and general knowledge.
     * NO LIMIT by default - collects ALL documents for large-scale training (2GB+)
     */
    const preTrainTexts = [];
    
    try {
      const preTrainQuery = this.supabase
        .from('knowledge_documents')
        .select('id, content, title, learning_status, is_chunked, total_chunks')
        .in('learning_status', ['learned', 'pending', 'processing'])
        .order('created_at', { ascending: false });
      
      if (limit) {
        preTrainQuery.limit(limit);
      }
      
      let documentRows = null;
      let documentError = null;
      
      try {
        // Add timeout wrapper to prevent hanging
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Query timeout after 30s')), 30000)
        );
        
        const result = await Promise.race([preTrainQuery, timeoutPromise]);
        documentRows = result.data;
        documentError = result.error;
      } catch (timeoutError) {
        const errorStr = timeoutError?.message || timeoutError?.toString() || '';
        const isFetchError = errorStr.includes('fetch failed') || errorStr.includes('TypeError');
        
        if (isFetchError) {
          console.warn(`[EPSILON AI LANGUAGE ENGINE] Supabase connection issue while loading documents for pre-training (likely downtime)`);
          documentError = { message: 'Connection timeout or fetch failed' };
        } else {
          documentError = timeoutError;
        }
      }
      
      if (documentError) {
        if (!documentError.message?.includes('Connection timeout')) {
        console.warn(`[EPSILON AI LANGUAGE ENGINE] Failed to load documents for pre-training:`, documentError.message);
        }
      } else if (documentRows && documentRows.length) {
        _silent(`[EPSILON AI LANGUAGE ENGINE] Found ${documentRows.length} documents for pre-training`);
        
        for (let idx = 0; idx < documentRows.length; idx++) {
          const row = documentRows[idx];
          let text = (row.content || '').toString().trim();
          
          if (row.is_chunked && row.id) {
            try {
              const { fetchChunksInBatches } = require('../utils/chunk-fetcher');
              const chunkRows = await fetchChunksInBatches(this.supabase, row.id, {
                batchSize: 20,
                silent: true,
                maxRetries: 5
              });
              
              if (chunkRows && chunkRows.length > 0) {
                const MAX_STRING_LENGTH = 50 * 1024 * 1024;
                const parts = [];
                let totalLength = 0;
                for (const chunk of chunkRows) {
                  const chunkText = chunk.chunk_text || '';
                  if (totalLength + chunkText.length + 2 > MAX_STRING_LENGTH) {
                    const remaining = MAX_STRING_LENGTH - totalLength;
                    if (remaining > 100) {
                      parts.push(chunkText.slice(0, remaining));
                    }
                    break;
                  }
                  parts.push(chunkText);
                  totalLength += chunkText.length + 2;
                }
                text = parts.join('\n\n');
                if (text.length > MAX_STRING_LENGTH) {
                  text = text.substring(0, MAX_STRING_LENGTH);
                  console.warn(`   [EPSILON AI LANGUAGE ENGINE] WARNING: Final joined string exceeded limit, truncated to ${MAX_STRING_LENGTH} chars`);
                }
                _silent(`   Reconstructed chunked document "${row.title || 'unknown'}" from ${parts.length}/${chunkRows.length} chunks (${text.length} chars) for pre-training`);
              }
            } catch (chunkFetchError) {
              const errorStr = chunkFetchError?.message || chunkFetchError?.toString() || '';
              const isFetchError = errorStr.includes('fetch failed') || errorStr.includes('TypeError');
              
              if (isFetchError) {
                console.warn(`[EPSILON AI LANGUAGE ENGINE] Supabase connection issue while fetching chunks for document ${row.id} (likely downtime)`);
              } else {
              console.warn(`[EPSILON AI LANGUAGE ENGINE] Error fetching chunks for document ${row.id}:`, chunkFetchError.message);
              }
            }
          }
          
          if (text && text.includes(':') && text.split(':').length === 3) {
            try {
              const { decrypt } = require('../runtime/encryption');
              const decrypted = decrypt(text);
              if (decrypted) {
                text = decrypted;
              }
            } catch (e) {
            }
          }
          
          if (text && text.length >= 100) {
            preTrainTexts.push(text);
            
            if (idx < 3) {
              _silent(`   Added "${row.title || 'unknown'}" (${text.length} chars) for pre-training`);
            }
          }
          
          if (idx % 10 === 0) {
            await new Promise(resolve => setImmediate(resolve));
          }
        }
      }
      
      _silent(`[EPSILON AI LANGUAGE ENGINE] Collected ${preTrainTexts.length} texts for pre-training`);
      
    } catch (error) {
      // Check if error contains HTML (Supabase downtime)
      const errorStr = error?.message || error?.toString() || JSON.stringify(error) || '';
      const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                         errorStr.includes('Cloudflare') || 
                         errorStr.includes('Error code 522') || 
                         errorStr.includes('Error code 521');
      if (isHtmlError) {
        console.warn('[WARN] [EPSILON AI LANGUAGE ENGINE] Supabase connection issue while collecting pre-training texts');
      } else {
        console.error(`[ERROR] [EPSILON AI LANGUAGE ENGINE] Error collecting pre-training texts:`, error.message || 'Unknown error');
      }
    }
    
    return preTrainTexts;
  }

  _weightForCategory(category) {
    // Sales training documents are weighted higher because they teach Epsilon AI HOW to talk
        // (tone, language, style) for communication ability
    switch (category) {
      case 'sales':
      case 'sales_training':
      case 'communication_guide':
        return 1.35; // Higher weight - these teach communication style
      case 'case_study':
        return 1.15; // Learn from examples
      case 'technical':
        return 1.05; // Technical knowledge
      default:
        return 1.0;
    }
  }

  async _processQueue() {
    // Training is now local-only - queue processing disabled
    // Clear any queued jobs
    this.trainQueue = [];
    return;
    
    // OLD CODE BELOW - DISABLED
    /*
    if (!this.trainQueue.length || this.trainingInFlight) {
      return;
    }

    if (!this.isPythonReady()) {
      return;
    }

    const nextJob = this.trainQueue.shift();
    if (!nextJob) {
      return;
    }

    this.modelStatus.pendingReason = nextJob.reason;
    try {
      await this.trainNow({ reason: nextJob.reason });
    } catch (error) {
      // Check if error contains HTML (Supabase downtime)
      const errorStr = error?.message || error?.toString() || JSON.stringify(error) || '';
      const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                         errorStr.includes('Cloudflare') || 
                         errorStr.includes('Error code 522') || 
                         errorStr.includes('Error code 521');
      if (isHtmlError) {
        console.warn('[WARN] [EPSILON AI LANGUAGE ENGINE] Supabase connection issue during training job');
      } else {
        console.warn('[EPSILON AI LANGUAGE ENGINE] Training job failed:', error.message || 'Unknown error');
      }
    }
    */
  }

  // Self-learning methods
  startSelfLearning() {
    return this.selfLearning.startSelfLearning();
  }

  stopSelfLearning() {
    return this.selfLearning.stopSelfLearning();
  }

  getSelfLearningProgress() {
    return this.selfLearning.getLearningProgress();
  }

  async triggerSelfLearning() {
    return await this.selfLearning.triggerSelfLearning();
  }

  /**
   * Estimate training time based on data size, mode, and historical performance
   * @param {Object} params - Training parameters
   * @param {number} params.samples - Number of training samples
   * @param {number} params.preTrainTexts - Number of pre-training texts
   * @param {string} params.trainingMode - 'full' or 'fine_tune'
   * @param {Object} params.stats - Training statistics
   * @param {Array} params.historicalData - Previous training durations
   * @returns {Object} - { milliseconds, formatted, mode, confidence }
   */
  _estimateTrainingTime({ samples, preTrainTexts, trainingMode, stats, historicalData, actualSampleChars, actualPreTrainChars }) {
    const AVG_CHARS_PER_TOKEN = 4;
    
    // Calculate tokens from ACTUAL data sizes
    let estimatedSampleTokens, estimatedPreTrainTokens;
    
    if (actualSampleChars !== undefined && actualSampleChars > 0) {
      estimatedSampleTokens = Math.round(actualSampleChars / AVG_CHARS_PER_TOKEN);
    } else {
      const AVG_SAMPLE_CHARS = 1500;
      estimatedSampleTokens = Math.round((AVG_SAMPLE_CHARS / AVG_CHARS_PER_TOKEN) * samples);
    }
    
    if (actualPreTrainChars !== undefined && actualPreTrainChars > 0) {
      estimatedPreTrainTokens = Math.round(actualPreTrainChars / AVG_CHARS_PER_TOKEN);
    } else {
      const AVG_PRETRAIN_CHARS = 25000;
      estimatedPreTrainTokens = Math.round((AVG_PRETRAIN_CHARS / AVG_CHARS_PER_TOKEN) * preTrainTexts);
    }
    
    const SECONDS_PER_1000_TOKENS_PER_EPOCH = 1.0;
    const PRETRAIN_EPOCHS = 4;
    const FINETUNE_EPOCHS = 4;
    let preTrainTimeSeconds = 0;
    let fineTuneTimeSeconds = 0;
    
    if (trainingMode === 'full' && preTrainTexts > 0 && estimatedPreTrainTokens > 0) {
      preTrainTimeSeconds = (estimatedPreTrainTokens / 1000) * SECONDS_PER_1000_TOKENS_PER_EPOCH * PRETRAIN_EPOCHS;
      fineTuneTimeSeconds = (estimatedSampleTokens / 1000) * SECONDS_PER_1000_TOKENS_PER_EPOCH * FINETUNE_EPOCHS;
    } else {
      fineTuneTimeSeconds = (estimatedSampleTokens / 1000) * SECONDS_PER_1000_TOKENS_PER_EPOCH * FINETUNE_EPOCHS;
    }
    
    let baseEstimate = (preTrainTimeSeconds + fineTuneTimeSeconds) * 1000;
    
    
    // Factor in data complexity
    let complexityFactor = 1.0;
    if (stats) {
      // More categories = slightly more complex
      const categoryCount = Object.keys(stats.categories || {}).length;
      if (categoryCount > 5) complexityFactor += 0.1;
      if (categoryCount > 10) complexityFactor += 0.1;
      
      // More chunks = more data to process
      if (stats.chunks > 100) complexityFactor += 0.15;
      if (stats.chunks > 500) complexityFactor += 0.15;
    }
    
    baseEstimate *= complexityFactor;
    
    // Use historical data if available (more accurate)
    let finalEstimate = baseEstimate;
    let confidence = 'medium';
    let mode = 'calculated';
    
    if (historicalData && historicalData.length > 0) {
      // Find similar historical training runs
      const similarRuns = historicalData.filter(h => {
        const sampleDiff = Math.abs(h.samples - samples) / Math.max(samples, 1);
        const preTrainDiff = Math.abs((h.preTrainTexts || 0) - preTrainTexts) / Math.max(preTrainTexts, 1);
        return sampleDiff < 0.3 && preTrainDiff < 0.3; // Within 30% similarity
      });
      
      if (similarRuns.length > 0) {
        // Use average of similar runs
        const avgHistoricalTime = similarRuns.reduce((sum, h) => sum + (h.duration || 0), 0) / similarRuns.length;
        // Weighted average: 70% historical, 30% calculated
        finalEstimate = (avgHistoricalTime * 0.7) + (baseEstimate * 0.3);
        confidence = 'high';
        mode = 'historical';
      } else if (historicalData.length > 0) {
        // Use overall average with scaling
        const avgHistoricalTime = historicalData.reduce((sum, h) => sum + (h.duration || 0), 0) / historicalData.length;
        const avgHistoricalSamples = historicalData.reduce((sum, h) => sum + (h.samples || 0), 0) / historicalData.length;
        
        if (avgHistoricalSamples > 0) {
          // Scale based on sample ratio
          const sampleRatio = samples / avgHistoricalSamples;
          finalEstimate = avgHistoricalTime * sampleRatio;
          confidence = 'medium';
          mode = 'scaled_historical';
        }
      }
    }
    
    // Add overhead (model loading, checkpointing, vocabulary building, etc.)
    // Transformer initialization, embedding layer setup, optimizer initialization
    finalEstimate += 300000; // 5 minutes overhead for model initialization and checkpointing
    
    // Format time estimate
    const seconds = Math.round(finalEstimate / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    const remainingSeconds = seconds % 60;
    
    let formatted;
    if (hours > 0) {
      formatted = `${hours}h ${remainingMinutes}m`;
    } else if (minutes > 0) {
      formatted = `${minutes}m ${remainingSeconds}s`;
    } else {
      formatted = `${seconds}s`;
    }
    
    return {
      milliseconds: finalEstimate,
      formatted,
      mode,
      confidence,
      samples,
      preTrainTexts,
      trainingMode
    };
  }
}

module.exports = new EpsilonLanguageEngine();


