// api/supabase-proxy.js - Secure proxy for Supabase operations
const express = require('express');
const { createClient } = require('@supabase/supabase-js');
const router = express.Router();
const { sanitizeText, validateUserInput } = require('../runtime/sanitize');
const { decrypt } = require('../runtime/encryption');

const _silent = () => {};

// Validate environment variables
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_KEY) {
  console.error('ERROR: Missing required Supabase environment variables');
  throw new Error('Supabase configuration is incomplete');
}

// Initialize Supabase client - no cookies to prevent "header too large" errors
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
      detectSessionInUrl: false
    },
    global: {
      headers: {
        'X-Client-Info': 'neuralops-proxy',
        'x-connection-timeout': '30000' // 30 second connection timeout
      },
      fetch: (url, options = {}) => {
        const timeout = options.timeout || 30000; // 30 seconds default
        return fetch(url, {
          ...options,
          signal: AbortSignal.timeout(timeout),
          keepalive: true
        });
      }
    }
  }
);

// CORS headers - NO WILDCARDS
const headers = {
  'Access-Control-Allow-Origin': process.env.FRONTEND_URL || 'https://neuralops.biz',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, Cookie, X-CSRF-Token',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Credentials': 'true',
  'Content-Type': 'application/json'
};

// Helper function to extract the most relevant sections from documents
function extractRelevantSections(documents, query, keyTerms) {
  if (!documents || documents.length === 0) return [];
  
  const sections = [];
  
  documents.forEach(doc => {
    if (!doc.content) return;
    
    // Split content into paragraphs
    const paragraphs = doc.content
      .split(/\n\n+/)
      .filter(p => p.trim().length > 20 && p.trim().length < 500); // Filter out very short or long paragraphs
    
    // Score each paragraph based on relevance to query
    const scoredParagraphs = paragraphs.map(paragraph => {
      const lowerParagraph = paragraph.toLowerCase();
      let score = 0;
      
      // Check for exact query match
      if (lowerParagraph.includes(query.toLowerCase())) {
        score += 10;
      }
      
      // Check for key terms
      keyTerms.forEach(term => {
        const termRegex = new RegExp(`\\b${term}\\b`, 'gi');
        const matches = lowerParagraph.match(termRegex);
        if (matches) {
          score += matches.length * 2;
        }
      });
      
      // Bonus for paragraphs with sentences that directly answer questions
      if (query.toLowerCase().startsWith('how') && 
          (lowerParagraph.includes('process') || lowerParagraph.includes('steps') || 
           lowerParagraph.includes('way to') || lowerParagraph.includes('method'))) {
        score += 5;
      }
      
      if (query.toLowerCase().startsWith('what') && 
          (lowerParagraph.includes('is a') || lowerParagraph.includes('refers to') || 
           lowerParagraph.includes('defined as') || lowerParagraph.includes('means'))) {
        score += 5;
      }
      
      if (query.toLowerCase().includes('cost') || query.toLowerCase().includes('price') || 
          query.toLowerCase().includes('how much')) {
        if (lowerParagraph.match(/\$\d+|\d+ dollars|cost|price|pricing|package/g)) {
          score += 8;
        }
      }
      
      if (query.toLowerCase().includes('timeline') || query.toLowerCase().includes('how long')) {
        if (lowerParagraph.match(/week|month|day|hour|time|duration|schedule/g)) {
          score += 8;
        }
      }
      
      return { paragraph, score };
    });
    
    scoredParagraphs.sort((a, b) => b.score - a.score);
    const topParagraphs = scoredParagraphs.slice(0, 2);
    
    // Only include paragraphs with a minimum score
    topParagraphs.forEach(item => {
      if (item.score >= 3) {
        sections.push({
          docId: doc.id,
          docTitle: doc.title,
          content: item.paragraph,
          score: item.score
        });
      }
    });
  });
  
  sections.sort((a, b) => b.score - a.score);
  return sections.slice(0, 3);
}

const EMBEDDING_DIMENSION = 384;

function normalizeEmbeddingText(text = '') {
  if (!text) return '';
  return text
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s.,!?;:-]/g, '')
    .trim()
    .substring(0, 2000);
}

function generateHashEmbedding(text = '') {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const embedding = new Array(EMBEDDING_DIMENSION).fill(0);

  words.forEach(word => {
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = ((hash << 5) - hash + word.charCodeAt(i)) & 0xffffffff;
    }
    const index = Math.abs(hash) % EMBEDDING_DIMENSION;
    embedding[index] += 1;
  });

  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return embedding.map(val => (magnitude > 0 ? val / magnitude : 0));
}

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
    return 0;
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;
  return dotProduct / denominator;
}

function normalizeEmbeddingArray(value) {
  if (!value) return null;
  if (Array.isArray(value)) {
    const normalized = value.map(num => {
      const parsed = Number(num);
      return Number.isFinite(parsed) ? parsed : 0;
    });
    return normalized.length ? normalized : null;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    try {
      if ((trimmed.startsWith('[') && trimmed.endsWith(']')) || (trimmed.startsWith('{') && trimmed.endsWith('}'))) {
        const jsonLike = trimmed.startsWith('{')
          ? `[${trimmed.slice(1, -1)}]`
          : trimmed;
        const parsed = JSON.parse(jsonLike);
        if (Array.isArray(parsed)) {
          return parsed.map(num => {
            const parsedNum = Number(num);
            return Number.isFinite(parsedNum) ? parsedNum : 0;
          });
        }
      }
    } catch (err) {
      console.warn('Failed to parse embedding string, ignoring value');
    }
    return null;
  }

  if (typeof value === 'object' && value !== null) {
    if (Array.isArray(value.data)) {
      return value.data.map(num => {
        const parsed = Number(num);
        return Number.isFinite(parsed) ? parsed : 0;
      });
    }
  }

  return null;
}

function parseEmbeddingVector(row) {
  if (!row) return null;

  let vector = normalizeEmbeddingArray(row.embedding);
  if (!vector) {
    vector = normalizeEmbeddingArray(row.embedding_data);
  }

  if (!vector) return null;

  let processedVector = vector.slice(0, EMBEDDING_DIMENSION);
  if (processedVector.length < EMBEDDING_DIMENSION) {
    processedVector = processedVector.concat(
      new Array(EMBEDDING_DIMENSION - processedVector.length).fill(0)
    );
  }

  const magnitude = Math.sqrt(processedVector.reduce((sum, val) => sum + val * val, 0));
  if (magnitude > 0) {
    processedVector = processedVector.map(val => val / magnitude);
  }

  return processedVector;
}

function clampScore(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

const CATEGORY_KEYWORDS = {
  sales: ['price', 'pricing', 'plan', 'package', 'deal', 'close', 'conversion', 'roi', 'revenue', 'sell', 'purchase', 'discount', 'proposal', 'quote'],
  case_study: ['case study', 'customer', 'client', 'success story', 'results', 'implementation', 'deployment', 'before and after', 'outcome'],
  technical: ['api', 'integration', 'architecture', 'workflow', 'endpoint', 'deployment', 'schema', 'database', 'technical', 'engineering', 'roadmap', 'implementation', 'stack'],
  enablement: ['objection', 'script', 'talk track', 'pitch deck', 'battlecard', 'positioning', 'competitive', 'enablement', 'rebuttal'],
  onboarding: ['onboarding', 'kickoff', 'rollout', 'implementation plan', 'training plan', 'adoption', 'launch checklist'],
  retention: ['renewal', 'expansion', 'upsell', 'cross-sell', 'churn', 'retention', 'health score', 'success plan'],
  support: ['support', 'issue', 'bug', 'escalation', 'trouble', 'downtime', 'sla', 'incident'],
  testimonial: ['testimonial', 'review', 'feedback', 'quote', 'endorsement', 'reference call'],
  general: []
};

const STAGE_KEYWORDS = {
  discovery: ['discovery', 'awareness', 'top of funnel', 'initial', 'assessment', 'intro'],
  evaluation: ['evaluation', 'consideration', 'compare', 'proof of concept', 'poc', 'pilot', 'scorecard'],
  decision: ['decision', 'negotiation', 'contract', 'signature', 'closing'],
  onboarding: ['onboarding', 'kickoff', 'implementation', 'adoption'],
  renewal: ['renewal', 'retention', 'expansion', 'upsell'],
  support: ['support', 'escalation', 'ticket', 'issue', 'incident']
};

const AUDIENCE_KEYWORDS = {
  executive: ['executive', 'c-suite', 'c level', 'ceo', 'cfo', 'founder', 'board'],
  operations: ['operations', 'ops', 'manager', 'director', 'lead'],
  technical: ['developer', 'engineer', 'architect', 'technical', 'it', 'product'],
  sales: ['sales', 'ae', 'account executive', 'sdR', 'bizdev'],
  marketing: ['marketing', 'demand gen', 'growth', 'brand'],
  customer_success: ['customer success', 'cs', 'account manager', 'success manager']
};

const URGENCY_KEYWORDS = ['urgent', 'asap', 'immediately', 'right away', 'deadline', 'today', 'critical', 'priority'];

function inferQueryIntent(query = '') {
  const normalized = (query || '').toLowerCase();
  const keywordHits = new Set();
  const categoryWeights = { general: 1 };
  let primaryCategory = 'general';

  Object.entries(CATEGORY_KEYWORDS).forEach(([category, keywords]) => {
    let score = 0;
    keywords.forEach(keyword => {
      if (!keyword) return;
      if (normalized.includes(keyword)) {
        score += keyword.trim().split(/\s+/).length > 1 ? 1.2 : 1;
        keywordHits.add(keyword);
      }
    });

    if (score > 0) {
      const weight = 1.15 + score * 0.08;
      categoryWeights[category] = weight;
      if ((categoryWeights[primaryCategory] || 1) < weight) {
        primaryCategory = category;
      }
    }
  });

  if (normalized.includes('results') || normalized.includes('success story') || normalized.includes('improvements')) {
    categoryWeights.case_study = Math.max(categoryWeights.case_study || 1.2, 1.35);
    primaryCategory = 'case_study';
  }

  if (!categoryWeights.sales && (normalized.includes('pricing') || normalized.includes('quote'))) {
    categoryWeights.sales = 1.32;
    primaryCategory = 'sales';
  }

  if (!categoryWeights.technical && (normalized.includes('api') || normalized.includes('integration'))) {
    categoryWeights.technical = 1.28;
    primaryCategory = 'technical';
  }

  let stage = null;
  Object.entries(STAGE_KEYWORDS).forEach(([stageKey, keywords]) => {
    keywords.forEach(keyword => {
      if (!keyword) return;
      if (normalized.includes(keyword)) {
        stage = stageKey;
      }
    });
  });

  let audience = null;
  Object.entries(AUDIENCE_KEYWORDS).forEach(([audKey, keywords]) => {
    keywords.forEach(keyword => {
      if (!keyword) return;
      if (normalized.includes(keyword)) {
        audience = audKey;
      }
    });
  });

  const urgency = URGENCY_KEYWORDS.some(keyword => normalized.includes(keyword)) ? 'high' : 'normal';

  let persona = 'advisor';
  if (primaryCategory === 'sales' || stage === 'decision') {
    persona = 'sales';
  } else if (primaryCategory === 'technical') {
    persona = 'technical';
  } else if (primaryCategory === 'case_study' || primaryCategory === 'testimonial') {
    persona = 'credibility';
  }

  const tone = persona === 'sales'
    ? 'salesy'
    : persona === 'technical'
      ? 'technical'
      : persona === 'credibility'
        ? 'credibility'
        : 'neutral';

  return {
    primaryCategory,
    categoryWeights,
    tone,
    persona,
    stage,
    audience,
    urgency,
    keywordHits: Array.from(keywordHits)
  };
}

function getCategoryWeight(intent, category) {
  if (!category) return intent.categoryWeights.general || 1;
  const normalized = category.toLowerCase();
  return intent.categoryWeights[normalized] || intent.categoryWeights.general || 1;
}

function getToneWeight(intent, tone) {
  if (!tone) return 1;
  const normalizedTone = tone.toLowerCase();
  if (intent.tone === 'salesy' && (normalizedTone.includes('sales') || normalizedTone.includes('commercial'))) return 1.12;
  if (intent.tone === 'technical' && normalizedTone.includes('technical')) return 1.08;
  if (intent.tone === 'credibility' && (normalizedTone.includes('case') || normalizedTone.includes('formal'))) return 1.07;
  if (intent.tone === 'neutral' && normalizedTone === 'neutral') return 1.05;
  return 1;
}

function getStageWeight(intent, metadata) {
  const stageValue =
    metadata?.sales_stage ||
    metadata?.stage ||
    metadata?.funnel_stage ||
    metadata?.pipeline_stage ||
    metadata?.customer_journey_stage ||
    '';

  const normalizedStage = stageValue ? stageValue.toString().toLowerCase() : '';
  if (!normalizedStage) return 1;

  let weight = 1;
  if (intent.stage && normalizedStage.includes(intent.stage)) {
    weight += 0.14;
  } else if (intent.stage === 'evaluation' && /negotiation|decision|closing/.test(normalizedStage)) {
    weight += 0.08;
  } else if (intent.persona === 'sales' && /renewal|upsell|expansion/.test(normalizedStage)) {
    weight += 0.06;
  } else if (intent.persona === 'technical' && /implementation|onboarding|kickoff/.test(normalizedStage)) {
    weight += 0.06;
  }

  return clampScore(weight, 0.9, 1.25);
}

function getAudienceWeight(intent, metadata) {
  const audienceValue =
    metadata?.audience ||
    metadata?.target_persona ||
    metadata?.persona ||
    metadata?.role ||
    metadata?.stakeholder ||
    '';

  const normalizedAudience = audienceValue ? audienceValue.toString().toLowerCase() : '';
  if (!normalizedAudience) return 1;

  let weight = 1;
  if (intent.audience && normalizedAudience.includes(intent.audience)) {
    weight += 0.12;
  } else if (intent.persona === 'technical' && /developer|engineer|architect|technical/.test(normalizedAudience)) {
    weight += 0.08;
  } else if (intent.persona === 'sales' && /executive|c level|decision|vp/.test(normalizedAudience)) {
    weight += 0.08;
  }

  return clampScore(weight, 0.92, 1.2);
}

function getSignalWeight(metadata) {
  if (!metadata) return 1;
  const signals = metadata.signals || {};
  let weight = 1;

  if (signals.containsOutcomeLanguage) weight += 0.05;
  if (signals.containsCallToAction) weight += 0.05;
  if (signals.containsTimeline || metadata.timeline) weight += 0.03;

  const comparative = metadata.comparative_snapshot;
  if (comparative && typeof comparative.improvement_word_percent === 'number') {
    if (comparative.improvement_word_percent > 10) weight += 0.04;
    if (comparative.improvement_word_percent < -5) weight -= 0.03;
  }

  return clampScore(weight, 0.9, 1.18);
}

function computeRecencyWeight(createdAt) {
  if (!createdAt) return 1;
  const createdDate = new Date(createdAt);
  if (Number.isNaN(createdDate.getTime())) {
    return 1;
  }
  const daysDiff = (Date.now() - createdDate.getTime()) / (1000 * 60 * 60 * 24);
  if (daysDiff <= 0) return 1.2;
  if (daysDiff < 7) return 1.15;
  if (daysDiff < 30) return 1.08;
  if (daysDiff < 90) return 1.02;
  return 0.95;
}

function parseMetadata(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') {
    return { ...raw };
  }
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw);
    } catch (error) {
      console.warn('Unable to parse metadata JSON string. Ignoring metadata.');
      return {};
    }
  }
  return {};
}

// Handle OPTIONS requests for CORS
router.options('/', (req, res) => {
  res.set(headers);
  res.status(200).end();
});

// Utility: check Supabase table connectivity
async function checkTableConnection(tableName, columns = 'id') {
  try {
    const { count, error } = await supabase
      .from(tableName)
      .select(columns, { count: 'exact', head: true });
    
    if (error) {
      // Check if error contains HTML (Supabase downtime)
      const errorStr = error?.message || error?.toString() || JSON.stringify(error) || '';
      const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                         errorStr.includes('Cloudflare') || 
                         errorStr.includes('Error code 522') || 
                         errorStr.includes('Error code 521');
      if (isHtmlError) {
        console.warn(`[WARN] [SUPABASE PROXY] Table check failed for ${tableName}: Supabase connection issue (likely downtime)`);
      } else {
        console.error(`[SUPABASE PROXY] Table check failed for ${tableName}:`, error.message || error);
      }
      return {
        table: tableName,
        connected: false,
        rowCount: 0,
        error: error.message || 'Connection issue'
      };
    }
    
    return {
      table: tableName,
      connected: true,
      rowCount: typeof count === 'number' ? count : null,
      error: null
    };
  } catch (err) {
    // Check if error contains HTML (Supabase downtime)
    const errorStr = err?.message || err?.toString() || JSON.stringify(err) || '';
    const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                       errorStr.includes('Cloudflare') || 
                       errorStr.includes('Error code 522') || 
                       errorStr.includes('Error code 521');
    if (isHtmlError) {
      console.warn(`[WARN] [SUPABASE PROXY] Unexpected error checking table ${tableName}: Supabase connection issue (likely downtime)`);
    } else {
      console.error(`[SUPABASE PROXY] Unexpected error checking table ${tableName}:`, err.message || err);
    }
    return {
      table: tableName,
      connected: false,
      rowCount: 0,
      error: err.message || 'Connection issue'
    };
  }
}

// Handle Epsilon AI response generation (uses EpsilonAICore)
async function handleGetEpsilonResponse(body) {
  // Use inference client for all generation
  const { getInferenceClient } = require('../runtime/inference_client');
  const inferenceClient = getInferenceClient();
  
  const { user_message, session_id, context_data, user_id } = body;
  
  if (!user_message) {
    throw new Error('User message is required');
  }
  
  _silent('[PROXY EPSILON] Generating response for:', user_message.substring(0, 50) + '...');
  
  // Quick health check - single attempt, no delays
  let isReady = await inferenceClient.checkHealth();
  
  // Only retry once if not ready (for startup scenarios)
  if (!isReady) {
    _silent('[PROXY EPSILON] Model not ready, checking once more...');
    await new Promise(resolve => setTimeout(resolve, 500)); // Minimal delay
    isReady = await inferenceClient.checkHealth();
  }
  
  if (!isReady) {
    // Try to trigger a reload once if not ready
    try {
      const axios = require('axios');
      const inferenceUrl = process.env.INFERENCE_URL || 'http://127.0.0.1:8005';
      _silent('[PROXY EPSILON] Model not ready, attempting reload...');
      await axios.post(`${inferenceUrl}/reload-model`, {}, { timeout: 10000 });
      await new Promise(resolve => setTimeout(resolve, 1000));
      isReady = await inferenceClient.checkHealth();
    } catch (reloadError) {
      _silent('[PROXY EPSILON] Reload attempt failed:', reloadError.message);
    }
    
    if (!isReady) {
      const isProduction = process.env.NODE_ENV === 'production';
      if (isProduction) {
        throw new Error('AI model is currently loading. Please try again in a few seconds.');
      } else {
        throw new Error('AI model is not ready. Please wait a moment and try again.');
      }
    }
  }
  
  // Get conversation history if available
  let conversationHistory = [];
  if (session_id) {
    try {
      const { data: recentConvs } = await supabase
        .from('epsilon_conversations')
        .select('user_message, epsilon_response')
        .eq('session_id', session_id)
        .order('created_at', { ascending: false })
        .limit(10); // Last 10 exchanges for context
      
      if (recentConvs && recentConvs.length > 0) {
        // Build conversation history array (oldest first)
        recentConvs.reverse().forEach(conv => {
          if (conv.user_message) {
            conversationHistory.push({ role: 'user', content: conv.user_message });
          }
          if (conv.epsilon_response) {
            conversationHistory.push({ role: 'assistant', content: conv.epsilon_response });
          }
        });
      }
    } catch (historyError) {
      _silent('[PROXY EPSILON] Could not fetch conversation history:', historyError.message);
    }
  }
  
  // Format prompt for Epsilon AI with Harmony format
  // The model uses Harmony response format automatically via chat template
  // Pass messages in conversation format - the inference service handles Harmony formatting
  const formattedPrompt = user_message;
  
  // Generate response using inference client
  try {
    // Generation parameters for quality responses
    const result = await inferenceClient.generate({
      prompt: formattedPrompt,
      max_new_tokens: 512,   // Increased for better responses
      temperature: 0.7,     // Balanced creativity and coherence
      top_p: 0.9,          // Nucleus sampling
      repetition_penalty: 1.3,  // Prevent repetition loops
      conversation_history: conversationHistory
    });
    
    if (!result || !result.text) {
      throw new Error('Inference service returned invalid response - no text generated');
    }
    
    const cleanedResponse = String(result.text || '').trim();
    
    _silent('[PROXY EPSILON] Response generated via inference client');
    return {
      success: true,
      response: cleanedResponse,
      session_id: session_id,
      meta: {
        source: 'inference_service',
        confidence: 0.8,
        model_ready: true,
        model_id: result.model_id,
        tokens: result.tokens
      }
    };
  } catch (error) {
    console.error('[ERROR] [PROXY EPSILON] Inference generation failed:', error.message);
    
    // Provide more helpful error messages
    if (error.message && error.message.includes('timeout')) {
      throw new Error('Request timed out. The AI service may be busy. Please try again.');
    } else if (error.message && error.message.includes('ECONNREFUSED')) {
      throw new Error('AI service is not available. Please try again in a few moments.');
    } else if (error.message && error.message.includes('503')) {
      throw new Error('AI model is still loading. Please try again in a few seconds.');
    } else {
      throw new Error(`Failed to generate response: ${error.message}`);
    }
  }
}

// Helper function to check if table exists
async function checkTableExists(tableName) {
  try {
    const { error } = await supabase
      .from(tableName)
      .select('id', { count: 'exact', head: true });
    
    return !error;
  } catch (err) {
    return false;
  }
}

// Main proxy endpoint
router.post('/', async (req, res) => {
  // Set CORS headers
  res.set(headers);
  
  try {
    const { action, data } = req.body;
    
    if (!action) {
      return res.status(400).json({ error: 'Action is required' });
    }
    
    // CSRF validation - check if token is present and valid
    // SECURITY: Reject requests with failed CSRF validation
    const isDevelopment = process.env.NODE_ENV !== 'production';
    
    if (req.csrfValid === false) {
      // Parse cookies manually to avoid module path issues
      const parseCookies = (cookieHeader) => {
        const cookies = {};
        if (cookieHeader) {
          cookieHeader.split(';').forEach(cookie => {
            const [name, value] = cookie.trim().split('=');
            if (name && value) {
              cookies[name] = decodeURIComponent(value);
            }
          });
        }
        return cookies;
      };
      const cookies = parseCookies(req.headers.cookie || '');
      const cookieToken = cookies.csrfToken;
      const headerToken = req.headers['x-csrf-token'] || req.headers['X-CSRF-Token'];
      
      // In development: Allow if header token OR cookie token exists (don't require both)
      // In production: Require both cookie and header token
      if (isDevelopment) {
        // Development: Allow if we have ANY token
        if (headerToken || cookieToken) {
          // Valid token exists, allow request
          req.csrfValid = true;
        } else {
          // No tokens at all - only block in production
          // In development, allow but don't log (to reduce noise)
        }
      } else {
        // Production: Require both tokens
        if (!cookieToken && !headerToken) {
          // Only block if BOTH tokens are missing
        try {
            const { logSecurityEvent } = require('../runtime/logging');
          logSecurityEvent('CSRF_VALIDATION_FAILED', {
            path: req.path,
            ip: req.ip,
            method: req.method,
              action: action || 'unknown',
              hasCookieToken: !!cookieToken,
              hasHeaderToken: !!headerToken
          }, 'warn');
        } catch (logError) {
            // Logging module not available, continue anyway
        }
        
        return res.status(403).json({ 
          success: false,
          error: 'CSRF validation failed' 
        });
        }
      }
    }
    
    // Log successful CSRF token presence (for debugging)
    if (req.headers['x-csrf-token']) {
      _silent(`CSRF token found for ${req.path}: ${req.headers['x-csrf-token']?.substring(0, 8)}...`);
    } else if (req.csrfValid === true) {
      // Token validated via middleware, no header needed
      _silent(`CSRF validated via middleware for ${req.path}`);
    }
    
    _silent(`Supabase proxy action: ${action}`);
    
    // SECURITY: Enhanced input sanitization
    const sanitizedData = data ? JSON.parse(JSON.stringify(data)) : {};
    Object.keys(sanitizedData).forEach(key => {
      if (typeof sanitizedData[key] === 'string') {
        // Use enhanced validation for user inputs
        sanitizedData[key] = validateUserInput(sanitizedData[key], {
          maxLength: 10000,
          allowHTML: false,
          allowSpecialChars: true
        });
      } else if (Array.isArray(sanitizedData[key])) {
        // Sanitize array elements
        sanitizedData[key] = sanitizedData[key].map(item => {
          if (typeof item === 'string') {
            return validateUserInput(item, { maxLength: 5000, allowHTML: false });
          }
          return item;
        });
      }
    });
    
    let result = { success: false, error: 'No action handler found' };
    
    // Handle different actions
    switch (action) {
      case 'track-page-visit':
        try {
          const { session_id, page } = sanitizedData;
          
          if (!session_id || !page) {
            throw new Error('Session ID and page are required');
          }
          
          // Validate page URL
          const pageUrl = sanitizeText(page);
          if (!pageUrl || pageUrl.includes('<') || pageUrl.includes('>')) {
            throw new Error('Invalid page URL');
          }
          
          _silent('Tracking page visit:', pageUrl);
          
          // page_visits table is required
          const tableExists = await checkTableExists('page_visits');
          if (!tableExists) {
            throw new Error('page_visits table is required but not available');
          }
          
          const { data: pageData, error } = await supabase
            .from('page_visits')
            .insert([{
              session_id,
              page: pageUrl,
              user_agent: req.headers['user-agent'] || null
            }]);
            
          if (error) {
            throw error;
          }
          
          _silent('Page visit tracked successfully');
          result = { success: true };
        } catch (error) {
          throw new Error(`Page visit tracking failed: ${error.message}`);
        }
        break;
        
      case 'get-similar-responses':
        try {
          const { query_text, match_limit = 5 } = sanitizedData;
          
          if (!query_text) {
            throw new Error('Query text is required');
          }
          
          // First get similar user messages
          const { data: similarMessages, error } = await supabase
            .from('epsilon_conversations')
            .select('user_message, epsilon_response, created_at')
            .not('user_message', 'is', null)
            .or(`user_message.ilike.%${query_text}%,epsilon_response.ilike.%${query_text}%`)
            .order('created_at', { ascending: false })
            .limit(match_limit);
            
          if (error) throw error;
          
          const results = [];
          
          if (similarMessages && similarMessages.length > 0) {
            for (const msg of similarMessages) {
              // Since we already have the conversation data, no need for separate query
              // Just use the message data directly
              if (msg.user_message && msg.epsilon_response) {
                results.push({
                  user_message: msg.user_message,
                  message: msg.epsilon_response,
                  timestamp: msg.created_at
                });
              }
            }
          }
          
          result = { success: true, results };
        } catch (error) {
          throw error;
        }
        break;
        
      case 'get-learning-insights':
        try {
          // Get top intents (if intents table exists)
          let topIntents = [];
          try {
            const intentsTableExists = await checkTableExists('intents');
            if (intentsTableExists) {
              const { data: intentsData, error: intentsError } = await supabase
                .from('intents')
                .select('intent, count')
                .order('count', { ascending: false })
                .limit(10);
            
          if (intentsError) throw intentsError;
              topIntents = intentsData || [];
            }
          } catch (intentsErr) {
            // Intents table is optional - continue without it
            topIntents = [];
          }
          
          // Get average feedback rating
          const { data: avgFeedback, error: feedbackError } = await supabase
            .from('epsilon_feedback')
            .select('avg(rating)')
            .limit(1).maybeSingle();
            
          if (feedbackError) throw feedbackError;
          
          // Get most visited pages
          const { data: topPages, error: pagesError } = await supabase
            .from('page_visits')
            .select('page, count(*)')
            .group('page')
            .order('count', { ascending: false })
            .limit(5);
            
          if (pagesError) throw pagesError;
          
          result = {
            success: true,
            insights: {
              top_intents: topIntents || [],
              average_feedback: avgFeedback?.avg || 0,
              top_pages: topPages || []
            }
          };
        } catch (error) {
          throw error;
        }
        break;
        
      case 'search-knowledge':
        try {
          const { query_text, doc_type, match_limit = 3 } = sanitizedData;
          
          _silent(`Searching knowledge documents for: "${query_text}"`);
          
          if (!query_text || query_text.trim().length < 3) {
            console.warn('Search query too short or empty');
            result = { success: true, results: [] };
            break;
          }
          
          // Extract key terms for better search
          const keyTerms = query_text
            .toLowerCase()
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(term => term.length > 3 && !['what', 'when', 'where', 'which', 'how', 'why', 'does', 'will', 'should', 'could', 'would'].includes(term));
          
          _silent('🔑 Key search terms:', keyTerms.join(', '));
          
          // Get user ID from request if available (for tenant isolation)
          const userId = req.user?.id;
          
          // Use the search_documents function for better results
          let searchQuery = {
            search_query: query_text,
            limit_count: match_limit || 3
          };
          
          // Add owner_id if available for tenant isolation
          if (userId) {
            searchQuery.owner_id = userId;
          }
          
          const { data: documents, error } = await supabase.rpc(
            'search_documents',
            searchQuery
          );
          
          if (error) {
            console.error('RPC function search_documents failed:', error);
            throw new Error(`RPC function search_documents failed: ${error.message}`);
          }
          
          _silent(`Found ${documents?.length || 0} matching documents`);
          
          // Decrypt content for each result from RPC call
          const decryptedRpcResults = (documents || []).map(doc => {
            try {
              if (doc.content) {
                const isEncrypted = typeof doc.content === 'string' && doc.content.includes(':') && doc.content.split(':').length === 3;
                
                if (isEncrypted) {
                  const decryptedContent = decrypt(doc.content);
                  if (!decryptedContent) {
                    console.error(`[SUPABASE PROXY] Decryption failed for document ${doc.id} in RPC search`);
                    console.error(`[SUPABASE PROXY] Encrypted content preview: ${doc.content.substring(0, 100)}...`);
                  }
                  return {
                    ...doc,
                    content: decryptedContent || '' // Return empty string if decryption fails, never return encrypted content
                  };
                } else {
                  // Content is not encrypted, return as-is
                  return {
                    ...doc,
                    content: doc.content
                  };
                }
              }
              return doc;
            } catch (decryptError) {
              console.error(`[SUPABASE PROXY] Error decrypting document ${doc.id} in RPC search:`, decryptError);
              return {
                ...doc,
                content: '' // Return empty string if decryption fails, never return encrypted content
              };
            }
          });
          
          // Extract the most relevant sections from each document
          const relevantSections = extractRelevantSections(decryptedRpcResults || [], query_text, keyTerms);
          
          result = { 
            success: true, 
            results: decryptedRpcResults || [],
            relevantSections: relevantSections
          };
        } catch (error) {
          console.error('Knowledge search error:', error);
          // Return empty results to prevent frontend errors
          result = { success: true, results: [], relevantSections: [] };
        }
        break;
        
      // Epsilon AI Learning System Actions
      case 'store-epsilon-conversation':
        try {
          const { session_id, user_id, user_message, epsilon_response, response_time_ms, context_data, learning_metadata } = sanitizedData;
          
          _silent('[EXTENSIVE LOG] Received data:', {
            session_id: session_id,
            user_id: user_id || 'null',
            user_message_length: user_message?.length || 0,
            epsilon_response_length: epsilon_response?.length || 0,
            response_time_ms: response_time_ms || 0,
            has_context_data: !!context_data,
            has_learning_metadata: !!learning_metadata,
            context_data_keys: context_data ? Object.keys(context_data) : [],
            learning_metadata_keys: learning_metadata ? Object.keys(learning_metadata) : []
          });
          
          if (!session_id || !user_message || !epsilon_response) {
            const missing = [];
            if (!session_id) missing.push('session_id');
            if (!user_message) missing.push('user_message');
            if (!epsilon_response) missing.push('epsilon_response');
            throw new Error(`Missing required fields: ${missing.join(', ')}`);
          }
          
          _silent('[EXTENSIVE LOG] All required fields present, proceeding...');
          
          // Store directly in epsilon_conversations table (RPC function doesn't exist)
          try {
            _silent('[EXTENSIVE LOG] Storing conversation directly in epsilon_conversations table...');
            
            const conversationData = {
              session_id: session_id,
              user_id: user_id || null,
              user_message: user_message,
              epsilon_response: epsilon_response,
              response_time_ms: response_time_ms || 0,
              context_data: context_data || {},
              learning_metadata: learning_metadata || {}
            };
            
            _silent('[EXTENSIVE LOG] Conversation data:', {
              session_id: conversationData.session_id,
              user_id: conversationData.user_id,
              user_message_length: conversationData.user_message?.length || 0,
              epsilon_response_length: conversationData.epsilon_response?.length || 0,
              response_time_ms: conversationData.response_time_ms,
              context_data_size: JSON.stringify(conversationData.context_data).length,
              learning_metadata_size: JSON.stringify(conversationData.learning_metadata).length
            });
            
            const { data: insertedData, error } = await supabase
              .from('epsilon_conversations')
              .insert([conversationData])
              .select('id')
              .limit(1).maybeSingle();
            
            if (error) {
              console.error('[EXTENSIVE LOG] Insert failed:', error);
              throw new Error(`Failed to store conversation: ${error.message}`);
            }
            
            _silent('[EXTENSIVE LOG] Conversation stored successfully, id:', insertedData?.id);
            const conversationId = insertedData?.id;
            
            // ALSO store in main conversations/messages tables for training collection
            let mainConversationId = null;
            try {
              // Get or create conversation
              const { data: existingConv, error: convError } = await supabase
                .from('conversations')
                .select('id')
                .eq('session_id', session_id)
                .limit(1).maybeSingle();
              
              if (convError || !existingConv) {
                // Create new conversation
                const { data: newConv, error: newConvError } = await supabase
                  .from('conversations')
                  .insert([{
                    user_id: user_id || null,
                    session_id: session_id,
                    conversation_name: null,
                    folder_id: null,
                    is_deleted: false,
                    created_at: new Date().toISOString(),
                    updated_at: new Date().toISOString()
                  }])
                  .select('id')
                  .limit(1).maybeSingle();
                
                if (!newConvError && newConv) {
                  mainConversationId = newConv.id;
                  
                  // Log conversation creation
                  try {
                    await supabase.from('conversation_changes').insert([{
                      conversation_id: mainConversationId,
                      user_id: user_id || null,
                      change_type: 'create',
                      old_value: null,
                      new_value: session_id,
                      metadata: { session_id: session_id },
                      created_at: new Date().toISOString()
                    }]);
                  } catch (logError) {
                    console.warn('Failed to log conversation creation (non-critical):', logError.message);
                  }
                }
              } else {
                mainConversationId = existingConv.id;
              }
              
              // Store messages
              if (mainConversationId) {
                // Store user message
                await supabase.from('messages').insert([{
                  conversation_id: mainConversationId,
                  role: 'user',
                  text: user_message,
                  created_at: new Date().toISOString()
                }]);
                
                // Store Epsilon AI response
                await supabase.from('messages').insert([{
                  conversation_id: mainConversationId,
                  role: 'epsilon',
                  text: epsilon_response,
                  created_at: new Date().toISOString()
                }]);
                
                _silent('[SUPABASE PROXY] Also stored conversation in messages/conversations tables for training');
              }
            } catch (mainTableError) {
              console.warn('[SUPABASE PROXY] Failed to store in main tables (non-critical):', mainTableError.message);
            }
            
            result = { success: true, conversation_id: conversationId };
            
          } catch (insertError) {
            console.error('[EXTENSIVE LOG] Conversation insert failed:', insertError.message);
            console.error('[EXTENSIVE LOG] Insert error details:', insertError);
            throw new Error(`Failed to store conversation: ${insertError.message}`);
          }
        } catch (error) {
          console.error('[EXTENSIVE LOG] Exception during conversation storage:', error.message);
          console.error('[EXTENSIVE LOG] Error stack:', error.stack);
          throw new Error(`Conversation storage failed: ${error.message}`);
        }
        _silent('[EXTENSIVE LOG] ===== EPSILON AI CONVERSATION STORAGE END =====');
        break;
        
      case 'store-epsilon-feedback':
        try {
          const { conversation_id, user_id, rating, was_helpful, correction_text, improvement_suggestion, feedback_type, feedback_text } = sanitizedData;
          
          _silent('[EXTENSIVE LOG] Received feedback data:', {
            conversation_id: conversation_id,
            user_id: user_id || 'null',
            rating: rating,
            was_helpful: was_helpful,
            correction_text_length: correction_text?.length || 0,
            improvement_suggestion_length: improvement_suggestion?.length || 0,
            feedback_type: feedback_type || 'rating',
            feedback_text_length: feedback_text?.length || 0
          });
          
          if (!conversation_id) {
            throw new Error('Conversation ID is required');
          }
          
          // epsilon_feedback table is required
          const tableExists = await checkTableExists('epsilon_feedback');
          if (!tableExists) {
            throw new Error('epsilon_feedback table is required but not available');
          }
          
          let sentiment_score = null;
          let keywords = {};
          
          // Analyze typed feedback if provided
          if (feedback_text && feedback_text.trim().length > 0) {
            try {
              const { data: analysisData, error: analysisError } = await supabase.rpc(
                'analyze_feedback_text',
                { p_feedback_text: feedback_text }
              );
              
              if (!analysisError && analysisData && analysisData.length > 0) {
                sentiment_score = analysisData[0].sentiment_score;
                keywords = analysisData[0].keywords;
              }
            } catch (analysisErr) {
              console.warn('Feedback analysis failed:', analysisErr.message);
            }
          }
          
          // Call RPC function - function exists in schema
          try {
            const { data: feedbackData, error } = await supabase.rpc(
              'store_epsilon_feedback',
              {
                p_conversation_id: conversation_id,
                p_user_id: user_id || null,
                p_rating: rating || null,
                p_was_helpful: was_helpful || null,
                p_correction_text: correction_text || null,
                p_improvement_suggestion: improvement_suggestion || null,
                p_feedback_type: feedback_type || 'rating',
                p_feedback_text: feedback_text || null,
                p_sentiment_score: sentiment_score,
                p_keywords: keywords
              }
            );
            
            if (error) throw error;
            
            result = { success: true, feedback_id: feedbackData };
          } catch (rpcError) {
            console.error('RPC function store_epsilon_feedback failed:', rpcError.message);
            throw new Error(`RPC function store_epsilon_feedback failed: ${rpcError.message}`);
          }
        } catch (error) {
          console.error('Store Epsilon AI feedback error:', error);
          throw new Error(`Feedback storage failed: ${error.message}`);
        }
        break;
        
      case 'store-epsilon-typed-feedback':
        try {
          const { conversation_id, user_id, feedback_text, feedback_category, sentiment_score } = sanitizedData;
          
          _silent('[EXTENSIVE LOG] Received typed feedback data:', {
            conversation_id: conversation_id,
            user_id: user_id || 'null',
            feedback_text_length: feedback_text?.length || 0,
            feedback_category: feedback_category || 'general',
            sentiment_score: sentiment_score || 'null'
          });
          
          if (!conversation_id || !feedback_text) {
            console.warn('Typed feedback missing required fields:', { conversation_id, feedback_text });
            result = { success: false, error: 'Conversation ID and feedback text are required' };
            break;
          }
          
          // epsilon_typed_feedback table is required
          const tableExists = await checkTableExists('epsilon_typed_feedback');
          if (!tableExists) {
            throw new Error('epsilon_typed_feedback table is required but not available');
          }
          
          // Store typed feedback directly
          const { data: feedbackData, error: insertError } = await supabase
            .from('epsilon_typed_feedback')
            .insert([{
              conversation_id: conversation_id,
              user_id: user_id || null,
              feedback_text: feedback_text,
              feedback_category: feedback_category || 'general',
              sentiment_score: sentiment_score || null
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (insertError) {
            throw new Error(`Failed to store typed feedback: ${insertError.message}`);
          } else {
            result = { success: true, feedback_id: feedbackData.id };
          }
          
          _silent('[EXTENSIVE LOG] ===== EPSILON AI TYPED FEEDBACK STORAGE SUCCESS =====');
        } catch (error) {
          console.error('Store Epsilon AI typed feedback error:', error);
          throw new Error(`Typed feedback storage failed: ${error.message}`);
        }
        break;
        
      case 'get-similar-epsilon-conversations':
        try {
          const { query_text, limit = 5 } = sanitizedData;
          
          if (!query_text) {
            throw new Error('Query text is required');
          }
          
          // Call RPC function - function exists in schema
          try {
            const { data: similarConversations, error } = await supabase.rpc(
              'get_similar_epsilon_conversations',
              {
                p_query_text: query_text,
                p_limit: limit
              }
            );
            
            if (error) throw error;
            
            result = { success: true, conversations: similarConversations || [] };
          } catch (rpcError) {
            console.error('RPC function get_similar_epsilon_conversations failed:', rpcError.message);
            throw new Error(`RPC function get_similar_epsilon_conversations failed: ${rpcError.message}`);
          }
        } catch (error) {
          console.error('Get similar Epsilon AI conversations error:', error);
          throw new Error(`Get similar Epsilon AI conversations failed: ${error.message}`);
        }
        break;
        
      case 'get-feedback-insights':
        try {
          const { query_text, limit = 20 } = sanitizedData;
          
          // Get feedback from Supabase with optional query text filter
          let query = supabase
            .from('epsilon_feedback')
            .select('id, feedback_text, was_helpful, rating, created_at, user_message, epsilon_response')
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (query_text) {
            query = query.or(`feedback_text.ilike.%${query_text}%,user_message.ilike.%${query_text}%`);
          }
          
          const { data: feedback, error } = await query;
          
          if (error) {
            console.warn('[SUPABASE] Failed to get feedback insights:', error.message);
            result = { success: true, feedback: [] };
          } else {
            result = { success: true, feedback: feedback || [] };
          }
        } catch (error) {
          console.error('Get feedback insights error:', error);
          // Return empty array instead of throwing to prevent 500 error
          result = { success: true, feedback: [] };
        }
        break;
        
      case 'get-successful-conversations':
        try {
          const { query_text, limit = 10, min_rating = 4 } = sanitizedData;
          
          // Get conversations with positive feedback/ratings
          let query = supabase
            .from('epsilon_conversations')
            .select('id, user_message, epsilon_response, created_at, session_id')
            .order('created_at', { ascending: false })
            .limit(limit * 2); // Get more to filter by rating
          
          if (query_text) {
            query = query.or(`user_message.ilike.%${query_text}%,epsilon_response.ilike.%${query_text}%`);
          }
          
          const { data: conversations, error: convError } = await query;
          
          if (convError) throw convError;
          
          // Get feedback for these conversations to filter by rating
          const conversationIds = conversations.map(c => c.id);
          const { data: feedback, error: feedbackError } = await supabase
            .from('epsilon_feedback')
            .select('conversation_id, rating, was_helpful')
            .in('conversation_id', conversationIds);
          
          // Create a map of conversation_id to rating
          const ratingMap = {};
          if (feedback) {
            feedback.forEach(f => {
              if (f.conversation_id && (f.rating >= min_rating || f.was_helpful === true)) {
                ratingMap[f.conversation_id] = {
                  rating: f.rating || (f.was_helpful ? 5 : 3),
                  was_helpful: f.was_helpful
                };
              }
            });
          }
          
          // Filter conversations by rating and add rating info
          const successfulConversations = conversations
            .filter(conv => {
              const ratingInfo = ratingMap[conv.id];
              return ratingInfo && (ratingInfo.rating >= min_rating || ratingInfo.was_helpful === true);
            })
            .map(conv => ({
              ...conv,
              rating: ratingMap[conv.id]?.rating || 5,
              feedback_score: ratingMap[conv.id]?.was_helpful ? 1 : 0.5
            }))
            .slice(0, limit);
          
          result = { success: true, conversations: successfulConversations };
        } catch (error) {
          console.error('Get successful conversations error:', error);
          throw error;
        }
        break;
        
      case 'get-epsilon-learning-stats':
        try {
          const { start_date, end_date } = sanitizedData;
          
          // Call RPC function - function exists in schema
            const { data: stats, error } = await supabase.rpc(
              'get_epsilon_conversation_stats',
              {
                start_date: start_date || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
                end_date: end_date || new Date().toISOString()
              }
            );
            
          if (error) {
            console.warn('[SUPABASE] RPC function get_epsilon_conversation_stats failed:', error.message);
            result = { success: true, stats: {} };
          } else {
            result = { success: true, stats: stats[0] || {} };
          }
        } catch (error) {
          console.error('Get Epsilon AI learning stats error:', error);
          // Return empty stats instead of throwing to prevent 500 error
          result = { success: true, stats: {} };
        }
        break;
        
      case 'get-epsilon-learning-insights':
        // Initialize result immediately to prevent undefined errors
        result = { success: true, insights: [] };
        
        try {
          if (!supabase) {
            console.warn('[SUPABASE] Supabase client not available');
            break;
          }
          
          let patternsTableExists = false;
          try {
            patternsTableExists = await checkTableExists('epsilon_learning_patterns');
          } catch (checkError) {
            console.warn('[SUPABASE] Error checking table existence:', checkError?.message || String(checkError));
            patternsTableExists = false;
          }
          
          if (!patternsTableExists) {
            // Table doesn't exist, return empty insights
            break;
          }
          
          // Try direct query first (more reliable than RPC for this use case)
          let insights = [];
          let queryError = null;
          
          try {
            const { data: directInsights, error: directError } = await supabase
              .from('epsilon_learning_patterns')
              .select('pattern_type, pattern_data, confidence_score, created_at, last_used_at, usage_count')
              .order('confidence_score', { ascending: false })
              .order('created_at', { ascending: false })
              .limit(100);
            
            if (directError) {
              queryError = directError;
              console.warn('[SUPABASE] Direct query error:', directError?.message || String(directError));
            } else {
              insights = Array.isArray(directInsights) ? directInsights : [];
              
              // Group by pattern_type and calculate aggregated insights
              const groupedInsights = {};
              try {
                insights.forEach(insight => {
                  if (!insight || typeof insight !== 'object') return;
                  
                  const type = (insight.pattern_type || 'unknown').toString();
                  if (!groupedInsights[type]) {
                    groupedInsights[type] = {
                      pattern_type: type,
                      pattern_count: 0,
                      avg_confidence: 0,
                      total_confidence: 0,
                      last_used: null
                    };
                  }
                  groupedInsights[type].pattern_count++;
                  if (insight.confidence_score != null) {
                    const score = parseFloat(insight.confidence_score);
                    if (!isNaN(score)) {
                      groupedInsights[type].total_confidence += score;
                    }
                  }
                  if (insight.last_used_at) {
                    try {
                      const lastUsed = new Date(insight.last_used_at);
                      if (!isNaN(lastUsed.getTime())) {
                        if (!groupedInsights[type].last_used || lastUsed > new Date(groupedInsights[type].last_used)) {
                          groupedInsights[type].last_used = insight.last_used_at;
                        }
                      }
                    } catch (dateError) {
                      // Ignore date parsing errors
                    }
                  }
                });
                
                // Calculate averages
                Object.keys(groupedInsights).forEach(type => {
                  if (groupedInsights[type].pattern_count > 0) {
                    groupedInsights[type].avg_confidence = groupedInsights[type].total_confidence / groupedInsights[type].pattern_count;
                  }
                });
                
                insights = Object.values(groupedInsights);
              } catch (aggregationError) {
                console.warn('[SUPABASE] Error aggregating insights:', aggregationError?.message || String(aggregationError));
                insights = [];
              }
            }
          } catch (directQueryError) {
            queryError = directQueryError;
            console.warn('[SUPABASE] Direct query threw error:', directQueryError?.message || String(directQueryError));
          }
          
          if (queryError && insights.length === 0) {
            try {
              const rpcResult = await supabase.rpc('get_epsilon_learning_insights');
              if (rpcResult && rpcResult.error) {
                console.warn('[SUPABASE] RPC function error:', rpcResult.error?.message || String(rpcResult.error));
              } else if (rpcResult && rpcResult.data !== undefined && Array.isArray(rpcResult.data)) {
                const rpcInsights = rpcResult.data || [];
                
                // Aggregate RPC results to match expected format
                const groupedInsights = {};
                try {
                  rpcInsights.forEach(insight => {
                    if (!insight || typeof insight !== 'object') return;
                    
                    const type = (insight.pattern_type || 'unknown').toString();
                    if (!groupedInsights[type]) {
                      groupedInsights[type] = {
                        pattern_type: type,
                        pattern_count: 0,
                        avg_confidence: 0,
                        total_confidence: 0,
                        last_used: null
                      };
                    }
                    groupedInsights[type].pattern_count++;
                    if (insight.confidence_score != null) {
                      const score = parseFloat(insight.confidence_score);
                      if (!isNaN(score)) {
                        groupedInsights[type].total_confidence += score;
                      }
                    }
                    if (insight.created_at) {
                      try {
                        const created = new Date(insight.created_at);
                        if (!isNaN(created.getTime())) {
                          if (!groupedInsights[type].last_used || created > new Date(groupedInsights[type].last_used)) {
                            groupedInsights[type].last_used = insight.created_at;
                          }
                        }
                      } catch (dateError) {
                        // Ignore date parsing errors
                      }
                    }
                  });
                  
                  // Calculate averages
                  Object.keys(groupedInsights).forEach(type => {
                    if (groupedInsights[type].pattern_count > 0) {
                      groupedInsights[type].avg_confidence = groupedInsights[type].total_confidence / groupedInsights[type].pattern_count;
                    }
                  });
                  
                  insights = Object.values(groupedInsights);
                } catch (rpcAggregationError) {
                  console.warn('[SUPABASE] Error aggregating RPC insights:', rpcAggregationError?.message || String(rpcAggregationError));
                  insights = [];
                }
              }
            } catch (rpcCallError) {
              console.warn('[SUPABASE] RPC call threw error:', rpcCallError?.message || String(rpcCallError));
            }
          }
          
          // Update result with insights (always ensure it's an array)
          result = { success: true, insights: Array.isArray(insights) ? insights : [] };
        } catch (error) {
          console.error('[SUPABASE] Error in get-epsilon-learning-insights:', error?.message || String(error), error?.stack);
          result = { success: true, insights: [] };
        }
        break;
        
      case 'get-conversations':
        try {
          const { user_id, limit = 50 } = sanitizedData;
          
          if (!user_id) {
            result = { success: true, conversations: [] };
            break;
          }
          
          const conversationsTableExists = await checkTableExists('conversations');
          if (!conversationsTableExists) {
            result = { success: true, conversations: [] };
            break;
          }
          
          // Get conversations - handle is_deleted column if it exists
          let { data: conversations, error } = await supabase
            .from('conversations')
            .select('id, session_id, created_at, conversation_name, folder_id')
            .eq('user_id', user_id)
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error && error.message && error.message.includes('is_deleted')) {
            ({ data: conversations, error } = await supabase
              .from('conversations')
              .select('id, session_id, created_at, conversation_name, folder_id')
              .eq('user_id', user_id)
              .order('created_at', { ascending: false })
              .limit(limit));
          }
          
          // Filter out deleted conversations if is_deleted column exists
          if (!error && conversations && Array.isArray(conversations)) {
            conversations = conversations.filter(conv => conv.is_deleted !== true);
          }
          
          if (error) {
            console.warn('[SUPABASE] Failed to get conversations:', error.message);
            result = { success: true, conversations: [] };
          } else {
            result = { success: true, conversations: conversations || [] };
          }
        } catch (error) {
          console.error('Get conversations error:', error);
          // Return empty array instead of throwing to prevent 500 error
          result = { success: true, conversations: [] };
        }
        break;
        
      case 'get-messages':
        try {
          const { conversation_id } = sanitizedData;
          
          if (!conversation_id) {
            result = { success: true, messages: [] };
            break;
          }
          
          const { data: messages, error } = await supabase
            .from('messages')
            .select('id, role, text, created_at, conversation_id')
            .eq('conversation_id', conversation_id)
            .order('created_at', { ascending: true });
          
          if (error) {
            console.warn('[SUPABASE] Failed to get messages:', error.message);
            result = { success: true, messages: [] };
            break;
          }
          
          let sessionId = null;
          try {
            const { data: convData } = await supabase
              .from('conversations')
              .select('session_id')
              .eq('id', conversation_id)
              .limit(1).maybeSingle();
            
            sessionId = convData?.session_id || null;
          } catch (convError) {
            console.warn('[SUPABASE] Failed to get conversation session ID:', convError.message);
          }
          
          if (sessionId && messages) {
            try {
            const { data: epsilonConvs } = await supabase
              .from('epsilon_conversations')
              .select('id')
              .eq('session_id', sessionId)
              .order('created_at', { ascending: true });
            
            if (epsilonConvs && epsilonConvs.length > 0) {
              let epsilonConvIndex = 0;
              for (let i = 0; i < messages.length; i++) {
                const msg = messages[i];
                if (msg.role === 'epsilon' && epsilonConvIndex < epsilonConvs.length) {
                  msg.conversation_id = epsilonConvs[epsilonConvIndex].id;
                  epsilonConvIndex++;
                } else {
                  msg.conversation_id = null;
                }
              }
              }
            } catch (epsilonError) {
              // Non-critical - continue without epsilon conversation IDs
            }
          }
          
          result = { success: true, messages: messages || [] };
        } catch (error) {
          console.error('Get messages error:', error);
          // Return empty array instead of throwing to prevent 500 error
          result = { success: true, messages: [] };
        }
        break;
        
      case 'get-epsilon-dashboard-data':
        try {
          // epsilon_learning_analytics table - handle missing table gracefully
          const learningAnalyticsExists = await checkTableExists('epsilon_learning_analytics');
          
          if (!learningAnalyticsExists) {
            console.warn('[SUPABASE] epsilon_learning_analytics table not available, returning empty dashboard data');
            result = { success: true, dashboard_data: { conversations: [], feedback: [], analytics: [] } };
            break;
          }
          
          const analyticsPromise = supabase
                .from('epsilon_learning_analytics')
                .select('id, learning_type, created_at')
                .order('created_at', { ascending: false })
            .limit(30);
          
          // Get dashboard data from existing tables
          const [conversationsResult, feedbackResult, analyticsResult] = await Promise.all([
            supabase
              .from('epsilon_conversations')
              .select('id, created_at, response_time_ms')
              .order('created_at', { ascending: false })
              .limit(30),
            supabase
              .from('epsilon_feedback')
              .select('id, rating, created_at')
              .order('created_at', { ascending: false })
              .limit(30),
            analyticsPromise
          ]);
          
          const dashboardData = {
            conversations: conversationsResult.data || [],
            feedback: feedbackResult.data || [],
            analytics: analyticsResult.data || []
          };
          
          result = { success: true, dashboard_data: dashboardData };
        } catch (error) {
          console.error('Get Epsilon AI dashboard data error:', error);
          // Return empty dashboard data instead of throwing to prevent 500 error
          result = { success: true, dashboard_data: { conversations: [], feedback: [], analytics: [] } };
        }
        break;
        
      case 'process-document-for-epsilon':
        try {
          const { document_id, document_type, file_name } = sanitizedData;
          
          if (!document_id) {
            result = { success: false, error: 'Document ID is required' };
            break;
          }
          
          const docIdStr = String(document_id).trim();
          if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
            result = { success: false, error: 'Invalid document ID' };
            break;
          }
          
          const { data: documentData, error: docError } = await supabase
            .from('knowledge_documents')
            .select('content, title, is_chunked, total_chunks')
            .eq('id', docIdStr)
            .limit(1).maybeSingle();
          
          if (docError) {
            const errorStr = docError?.message || docError?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              console.warn('[WARN] [SUPABASE PROXY] Supabase connection issue while fetching document');
            } else {
              console.warn('[SUPABASE] Failed to get document:', docError.message || 'Unknown error');
            }
            result = { success: false, error: 'Failed to retrieve document' };
            break;
          }
          
          if (!documentData) {
            result = { success: false, error: 'Document not found' };
            break;
          }
          
          let content = '';
          if (documentData.is_chunked && docIdStr) {
            try {
              const { data: chunkRows, error: chunkError } = await supabase
                .from('doc_chunks')
                .select('chunk_text, chunk_index')
                .eq('document_id', docIdStr)
                .order('chunk_index', { ascending: true })
                .limit(50);
              
              if (!chunkError && chunkRows && chunkRows.length > 0) {
                content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
                _silent(`[SUPABASE PROXY] Reconstructed chunked document ${docIdStr} from ${chunkRows.length} chunks for processing`);
              } else if (chunkError) {
                // Check if error contains HTML (Supabase downtime)
                const errorStr = chunkError?.message || chunkError?.toString() || '';
                const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                   errorStr.includes('Cloudflare') || 
                                   errorStr.includes('522') || 
                                   errorStr.includes('521');
                if (isHtmlError) {
                  console.warn(`[WARN] [SUPABASE PROXY] Supabase connection issue while fetching chunks for document ${docIdStr}`);
                } else {
                  console.warn(`[WARN] [SUPABASE PROXY] Failed to fetch chunks for document ${docIdStr}: ${chunkError.message || 'Unknown error'}`);
                }
                // Fall back to preview content
                content = documentData.content || '';
              } else {
                console.warn(`[WARN] [SUPABASE PROXY] Document ${docIdStr} marked as chunked but no chunks found`);
                content = documentData.content || '';
              }
            } catch (chunkFetchError) {
              console.warn(`[WARN] [SUPABASE PROXY] Error fetching chunks for document ${docIdStr}: ${chunkFetchError.message || 'Unknown error'}`);
              // Fall back to preview content
              content = documentData.content || '';
            }
          } else {
            // Document is not chunked, use content directly
            content = documentData.content || '';
          }
          
          // Decrypt document content before processing (if encrypted)
          if (content) {
            // Check if content is encrypted (has the format iv:authTag:encrypted)
            const isEncrypted = typeof content === 'string' && content.includes(':') && content.split(':').length === 3;
            
            if (isEncrypted) {
              const decryptedContent = decrypt(content);
              if (decryptedContent) {
                content = decryptedContent;
              } else {
                console.error(`[SUPABASE PROXY] Failed to decrypt content for document ${docIdStr}`);
                console.error(`[SUPABASE PROXY] Encrypted content preview: ${content.substring(0, 100)}...`);
                content = ''; // Use empty string if decryption fails, never use encrypted content
              }
            }
          }
          
          if (!content || content.trim().length === 0) {
            console.warn(`[SUPABASE PROXY] Document ${document_id} has no readable content after decryption`);
          }
          
          // Process the document content for Epsilon AI learning
          const title = documentData.title;
          
          // Extract key information and create training data
          const trainingData = await extractTrainingDataFromDocument(content, document_type, title);
          
          // Store training data - handle missing table gracefully
          const trainingDataTableExists = await checkTableExists('epsilon_training_data');
          if (!trainingDataTableExists) {
            console.warn('[SUPABASE] epsilon_training_data table not available, skipping training data storage');
          } else if (trainingData.length > 0) {
            try {
              const { error: trainingError } = await supabase
                .from('epsilon_training_data')
                .insert(trainingData.map(data => ({
                  input_text: data.input,
                  output_text: data.output,
                  document_id: documentData.id,
                  metadata: { document_type, title }
                })));
              
              if (trainingError) {
                console.warn('[SUPABASE] Failed to store training data:', trainingError.message);
              }
            } catch (trainingStoreError) {
              console.warn('[SUPABASE] Error storing training data:', trainingStoreError.message);
            }
          }
          
          result = { 
            success: true, 
            message: `Document processed successfully. Created ${trainingData.length} training examples.`,
            training_examples: trainingData.length
          };
        } catch (error) {
          console.error('Process document for Epsilon AI error:', error);
          // Return error response instead of throwing to prevent 500 error
          result = { 
            success: false, 
            error: error.message || 'Failed to process document',
            training_examples: 0
          };
        }
        break;
        
      case 'get-document-training-data':
        try {
          const { query, limit = 5 } = sanitizedData;
          
          if (!query) {
            throw new Error('Query is required');
          }
          
          _silent('Searching for training data with query:', query);
          
          // epsilon_training_data table is required
          const tableExists = await checkTableExists('epsilon_training_data');
          if (!tableExists) {
            throw new Error('epsilon_training_data table is required but not available');
          }
          
          // Query training data from table
          const { data: trainingData, error: queryError } = await supabase
            .from('epsilon_training_data')
            .select('input_text, output_text, created_at')
            .or(`input_text.ilike.%${query}%,output_text.ilike.%${query}%`)
            .limit(limit || 5);
          
          if (queryError) {
            throw new Error(`Failed to query training data: ${queryError.message}`);
          }
          
          _silent('Found training data:', trainingData?.length || 0, 'records');
          
          result = { 
            success: true, 
            training_data: trainingData || [],
            query: query
          };
        } catch (error) {
          console.error('Get document training data error:', error);
          throw error;
        }
        break;
        
      case 'store-training-data':
        try {
          const { input_text, expected_output, training_type, quality_score, is_validated, source_document_id } = sanitizedData;
          
          if (!input_text || !expected_output) {
            throw new Error('Input text and expected output are required');
          }
          
          // epsilon_training_data table is required
          const trainingTableExists = await checkTableExists('epsilon_training_data');
          if (!trainingTableExists) {
            throw new Error('epsilon_training_data table is required but not available');
          }
          
          _silent('[TRAINING DATA] Storing training data');
          
          const { data: trainingData, error } = await supabase
            .from('epsilon_training_data')
            .insert([{
              input_text,
              expected_output,
              training_type: training_type || 'conversation',
              quality_score: quality_score || 0.5,
              is_validated: is_validated || false,
              source_document_id: source_document_id || null,
              metadata: {}
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (error) throw error;
          
          result = { success: true, training_data_id: trainingData.id };
        } catch (error) {
          console.error('Store training data error:', error);
          throw error;
        }
        break;
        
      case 'store-learning-pattern':
        try {
          const { pattern_type, pattern_data, confidence_score, source_document_id } = sanitizedData;
          
          if (!pattern_type || !pattern_data) {
            result = { success: false, error: 'Pattern type and pattern data are required' };
            break;
          }
          
          // epsilon_learning_patterns table is required
          let patternsTableExists = false;
          try {
            patternsTableExists = await checkTableExists('epsilon_learning_patterns');
          } catch (checkError) {
            console.warn('[SUPABASE] Error checking epsilon_learning_patterns table:', checkError.message);
            patternsTableExists = false;
          }
          
          if (!patternsTableExists) {
            console.warn('[SUPABASE] epsilon_learning_patterns table not available, skipping pattern storage');
            result = { success: true, pattern_id: null, skipped: true };
            break;
          }
          
          _silent('[LEARNING PATTERNS] Storing learning pattern');
          
          const { data: patternData, error } = await supabase
            .from('epsilon_learning_patterns')
            .insert([{
              pattern_type,
              pattern_data,
              confidence_score: confidence_score || 0.5,
              usage_count: 1,
              metadata: { source_document_id: source_document_id || null }
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (error) {
            console.warn('[SUPABASE] Failed to store learning pattern:', error.message);
            result = { success: true, pattern_id: null, error: error.message };
          } else {
            result = { success: true, pattern_id: patternData?.id || null };
          }
        } catch (error) {
          console.warn('[SUPABASE] Error in store-learning-pattern:', error.message);
          result = { success: true, pattern_id: null, error: error.message };
        }
        break;
        
      case 'get-document-content':
        try {
          const { document_id } = sanitizedData;
          
          if (!document_id) {
            throw new Error('Document ID is required');
          }
          
          const docIdStr = String(document_id).trim();
          if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
            throw new Error('Invalid document ID');
          }
          
          const { data: document, error } = await supabase
            .from('knowledge_documents')
            .select('content, is_chunked, total_chunks')
            .eq('id', docIdStr)
            .limit(1).maybeSingle();
          
          if (error) {
            const errorStr = error?.message || error?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              console.warn('[WARN] [SUPABASE PROXY] Supabase connection issue while fetching document');
              result = { success: false, error: 'Supabase connection timeout - please try again later' };
              break;
            }
            throw error;
          }
          
          if (!document) {
            throw new Error('Document not found');
          }
          
          let content = '';
          if (document.is_chunked && docIdStr) {
            try {
              // Use chunk fetcher utility with timeout handling and batching
              const { fetchChunksInBatches } = require('../utils/chunk-fetcher');
              const chunkRows = await fetchChunksInBatches(supabase, docIdStr, { 
                batchSize: 50,
                silent: true 
              });
              
              if (chunkRows && chunkRows.length > 0) {
                content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
                _silent(`[SUPABASE PROXY] Reconstructed chunked document ${docIdStr} from ${chunkRows.length} chunks`);
              } else {
                console.warn(`[WARN] [SUPABASE PROXY] Document ${docIdStr} marked as chunked but no chunks found`);
                content = document.content || '';
              }
            } catch (chunkFetchError) {
              console.warn(`[WARN] [SUPABASE PROXY] Error fetching chunks for document ${docIdStr}: ${chunkFetchError.message || 'Unknown error'}`);
              // Fall back to preview content
              content = document.content || '';
            }
          } else {
            // Document is not chunked, use content directly
            content = document.content || '';
          }
          
          // Decrypt document content before returning (if encrypted)
          let decryptedContent = '';
          if (content) {
            const isEncrypted = typeof content === 'string' && content.includes(':') && content.split(':').length === 3;
            
            if (isEncrypted) {
              decryptedContent = decrypt(content);
              if (!decryptedContent) {
                console.error(`[SUPABASE PROXY] Failed to decrypt content for document ${docIdStr}`);
                console.error(`[SUPABASE PROXY] Encrypted content preview: ${content.substring(0, 100)}...`);
                decryptedContent = ''; // Return empty string if decryption fails
              } else {
                _silent(`[SUPABASE PROXY] Successfully decrypted content for document ${docIdStr} (${decryptedContent.length} chars)`);
              }
            } else {
              // Content is not encrypted, return as-is
              _silent(`[SUPABASE PROXY] Content for document ${docIdStr} is not encrypted, returning as-is`);
              decryptedContent = content;
            }
          }
          
          result = { success: true, content: decryptedContent };
        } catch (error) {
          // Check if error contains HTML (Supabase downtime)
          const errorStr = error?.message || error?.toString() || '';
          const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                             errorStr.includes('Cloudflare') || 
                             errorStr.includes('522') || 
                             errorStr.includes('521');
          if (isHtmlError) {
            console.warn('[WARN] [SUPABASE PROXY] Supabase connection issue while getting document content');
            result = { success: false, error: 'Supabase connection timeout - please try again later' };
          } else {
            console.error('Get document content error:', error.message || error);
            throw error;
          }
        }
        break;
        
      case 'get-epsilon-response':
        try {
          result = await handleGetEpsilonResponse(body);
        } catch (error) {
          console.error('[ERROR] [PROXY EPSILON] handleGetEpsilonResponse failed:', error.message);
          // Return proper error response - no fallback
          result = {
            success: false,
            error: error.message || 'Failed to generate response',
            session_id: body?.session_id || null,
            meta: {
              source: 'error',
              model_ready: false,
              service_available: false
            }
          };
        }
        break;
        
      // ENHANCED LEARNING SYSTEM ENDPOINTS
      case 'get-all-epsilon-conversations':
        try {
          const { limit = 100, time_range = '7d' } = sanitizedData;
          
          // Calculate date range
          const now = new Date();
          const daysBack = time_range === '7d' ? 7 : time_range === '30d' ? 30 : 1;
          const startDate = new Date(now.getTime() - (daysBack * 24 * 60 * 60 * 1000));
          
          _silent(`[ENHANCED LEARNING] Getting all conversations from last ${daysBack} days`);
          
          const { data: conversations, error } = await supabase
            .from('epsilon_conversations')
            .select('id, user_message, epsilon_response, created_at, session_id')
            .gte('created_at', startDate.toISOString())
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error) throw error;
          
          result = { success: true, data: conversations || [] };
        } catch (error) {
          console.error('Get all Epsilon AI conversations error:', error);
          result = { success: true, data: [] };
        }
        break;
        
      case 'get-all-feedback':
        try {
          const { limit = 200, time_range = '30d' } = sanitizedData;
          
          // Calculate date range
          const now = new Date();
          const daysBack = time_range === '7d' ? 7 : time_range === '30d' ? 30 : 1;
          const startDate = new Date(now.getTime() - (daysBack * 24 * 60 * 60 * 1000));
          
          _silent(`[ENHANCED LEARNING] Getting all feedback from last ${daysBack} days`);
          
          const { data: feedback, error } = await supabase
            .from('epsilon_feedback')
            .select('id, rating, was_helpful, feedback_text, created_at, conversation_id')
            .gte('created_at', startDate.toISOString())
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error) throw error;
          
          result = { success: true, data: feedback || [] };
        } catch (error) {
          console.error('Get all feedback error:', error);
          result = { success: true, data: [] };
        }
        break;
        
      case 'get-all-documents':
        try {
          const { limit = 50 } = sanitizedData;
          
          _silent(`[ENHANCED LEARNING] Getting all documents`);
          
          // Include chunked flags in query
          const { data: documents, error } = await supabase
            .from('knowledge_documents')
            .select('id, title, content, doc_type, document_type, learning_category, learning_status, learning_metadata, file_size, file_hash, created_at, updated_at, is_chunked, total_chunks')
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error) throw error;
          
          // Process documents - handle chunked documents
          const processedDocuments = await Promise.all((documents || []).map(async (doc) => {
            try {
              let content = doc.content || '';
              
              if (doc.is_chunked && doc.id) {
                try {
                  const docIdStr = String(doc.id).trim();
                  if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
                    console.warn(`[WARN] [SUPABASE PROXY] Invalid document_id: ${doc.id}, skipping chunks`);
                  } else {
                    const { data: chunkRows, error: chunkError } = await supabase
                      .from('doc_chunks')
                      .select('chunk_text, chunk_index')
                      .eq('document_id', docIdStr)
                      .order('chunk_index', { ascending: true })
                      .limit(50);
                    
                    if (!chunkError && chunkRows && chunkRows.length > 0) {
                      content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
                      _silent(`[SUPABASE PROXY] Reconstructed chunked document ${docIdStr} from ${chunkRows.length} chunks`);
                    } else if (chunkError) {
                      const errorStr = chunkError?.message || chunkError?.toString() || '';
                      const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                         errorStr.includes('Cloudflare') || 
                                         errorStr.includes('522') || 
                                         errorStr.includes('521');
                      if (isHtmlError) {
                        console.warn(`[WARN] [SUPABASE PROXY] Supabase connection issue while fetching chunks for document ${docIdStr}`);
                      } else {
                        console.warn(`[WARN] [SUPABASE PROXY] Failed to fetch chunks for document ${docIdStr}: ${chunkError.message || 'Unknown error'}`);
                      }
                      // Continue with preview content
                    } else if (!chunkRows || chunkRows.length === 0) {
                      console.warn(`[WARN] [SUPABASE PROXY] Document ${docIdStr} marked as chunked but no chunks found`);
                    }
                  }
                } catch (chunkFetchError) {
                  // Check if error contains HTML (Supabase downtime)
                  const errorStr = chunkFetchError?.message || chunkFetchError?.toString() || '';
                  const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                     errorStr.includes('Cloudflare') || 
                                     errorStr.includes('522') || 
                                     errorStr.includes('521');
                  if (isHtmlError) {
                    console.warn(`[WARN] [SUPABASE PROXY] Supabase connection issue while fetching chunks for document ${doc.id}`);
                  } else {
                    console.warn(`[WARN] [SUPABASE PROXY] Error fetching chunks for document ${doc.id}: ${chunkFetchError.message || 'Unknown error'}`);
                  }
                  // Continue with preview content
                }
              }
              
              // Decrypt document content if encrypted
              if (content) {
                const isEncrypted = typeof content === 'string' && content.includes(':') && content.split(':').length === 3;
                
                if (isEncrypted) {
                  const decryptedContent = decrypt(content);
                  if (decryptedContent) {
                    return {
                      ...doc,
                      content: decryptedContent
                    };
                  } else {
                    console.error(`[SUPABASE PROXY] Failed to decrypt content for document ${doc.id}`);
                    console.error(`[SUPABASE PROXY] Encrypted content preview: ${content.substring(0, 100)}...`);
                    return {
                      ...doc,
                      content: '' // Set empty string if decryption fails
                    };
                  }
                } else {
                  // Content is not encrypted, return as-is
                  return {
                    ...doc,
                    content: content
                  };
                }
              }
              return doc;
            } catch (processError) {
              console.error(`[SUPABASE PROXY] Error processing document ${doc.id}:`, processError);
              return {
                ...doc,
                content: '' // Set empty string if processing fails
              };
            }
          }));
          
          const decryptedDocuments = processedDocuments;
          
          result = { 
            success: true, 
            documents: decryptedDocuments,
            data: decryptedDocuments // Maintain legacy compatibility for clients expecting data property
          };
        } catch (error) {
          console.error('Get all documents error:', error);
          result = { success: true, documents: [], data: [] };
        }
        break;
        
      case 'get-recent-conversations':
        try {
          const { limit = 20, time_range = '1h' } = sanitizedData;
          
          // Calculate date range
          const now = new Date();
          const hoursBack = time_range === '1h' ? 1 : time_range === '24h' ? 24 : 1;
          const startDate = new Date(now.getTime() - (hoursBack * 60 * 60 * 1000));
          
          _silent(`[REAL-TIME LEARNING] Getting recent conversations from last ${hoursBack} hours`);
          
          const { data: conversations, error } = await supabase
            .from('epsilon_conversations')
            .select('id, user_message, epsilon_response, created_at, session_id')
            .gte('created_at', startDate.toISOString())
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error) throw error;
          
          result = { success: true, data: conversations || [] };
        } catch (error) {
          console.error('Get recent conversations error:', error);
          result = { success: true, data: [] };
        }
        break;
        
      case 'get-recent-feedback':
        try {
          const { limit = 50, time_range = '1h' } = sanitizedData;
          
          // Calculate date range
          const now = new Date();
          const hoursBack = time_range === '1h' ? 1 : time_range === '24h' ? 24 : 1;
          const startDate = new Date(now.getTime() - (hoursBack * 60 * 60 * 1000));
          
          _silent(`[REAL-TIME LEARNING] Getting recent feedback from last ${hoursBack} hours`);
          
          const { data: feedback, error } = await supabase
            .from('epsilon_feedback')
            .select('id, rating, was_helpful, feedback_text, created_at, conversation_id')
            .gte('created_at', startDate.toISOString())
            .order('created_at', { ascending: false })
            .limit(limit);
          
          if (error) throw error;
          
          result = { success: true, data: feedback || [] };
        } catch (error) {
          console.error('Get recent feedback error:', error);
          result = { success: true, data: [] };
        }
        break;
      
      case 'verify-supabase-tables':
        try {
          const tablesToCheck = [
            { name: 'knowledge_documents', columns: 'id' },
            { name: 'document_embeddings', columns: 'document_id' },
            { name: 'epsilon_conversations', columns: 'id' },
            { name: 'epsilon_feedback', columns: 'id' },
            { name: 'epsilon_learning_sessions', columns: 'session_id' },
            { name: 'epsilon_learning_patterns', columns: 'pattern_type' },
            { name: 'epsilon_model_weights', columns: 'weight_name' },
            { name: 'epsilon_training_data', columns: 'id' }
          ];
          
          const checks = [];
          for (const tableInfo of tablesToCheck) {
            const resultCheck = await checkTableConnection(tableInfo.name, tableInfo.columns);
            checks.push(resultCheck);
          }
          
          const allConnected = checks.every(check => check.connected);
          
          result = {
            success: true,
            allConnected,
            tables: checks
          };
        } catch (error) {
          console.error('Supabase table verification error:', error);
          result = {
            success: false,
            allConnected: false,
            tables: [],
            error: error.message
          };
        }
        break;
        
      case 'get-performance-metrics':
        try {
          const { time_range = '24h' } = sanitizedData;
          
          // Calculate date range
          const now = new Date();
          const hoursBack = time_range === '1h' ? 1 : time_range === '24h' ? 24 : 1;
          const startDate = new Date(now.getTime() - (hoursBack * 60 * 60 * 1000));
          
          _silent(`[SELF-IMPROVEMENT] Getting performance metrics from last ${hoursBack} hours`);
          
          // Get conversations and feedback for performance analysis
          const [conversationsResult, feedbackResult] = await Promise.all([
            supabase
              .from('epsilon_conversations')
              .select('id, response_time_ms, created_at')
              .gte('created_at', startDate.toISOString()),
            supabase
              .from('epsilon_feedback')
              .select('rating, was_helpful, created_at')
              .gte('created_at', startDate.toISOString())
          ]);
          
          const conversations = conversationsResult.data || [];
          const feedback = feedbackResult.data || [];
          
          // Calculate performance metrics
          const metrics = {
            total_conversations: conversations.length,
            total_feedback: feedback.length,
            avg_response_time: conversations.reduce((sum, c) => sum + (c.response_time_ms || 0), 0) / conversations.length || 0,
            avg_rating: feedback.reduce((sum, f) => sum + (f.rating || 0), 0) / feedback.length || 0,
            helpful_percentage: (feedback.filter(f => f.was_helpful === true).length / feedback.length * 100) || 0,
            quality_trend: 'stable', // Simplified for now
            improvement_areas: [] // Simplified for now
          };
          
          result = { success: true, data: [metrics] };
        } catch (error) {
          console.error('Get performance metrics error:', error);
          // Return default metrics instead of throwing to prevent 500 error
          result = { 
            success: true, 
            data: [{
              total_conversations: 0,
              total_feedback: 0,
              avg_response_time: 0,
              avg_rating: 0,
              helpful_percentage: 0,
              quality_trend: 'stable',
              improvement_areas: []
            }]
          };
        }
        break;

      case 'get-learning-metrics':
        try {
          const metrics = {
            timestamp: new Date().toISOString(),
            documents_total: 0,
            documents_last_7_days: 0,
            chunks_total: 0,
            sales_chunks: 0,
            technical_chunks: 0,
            testimonial_chunks: 0,
            general_chunks: 0,
            avg_feedback_rating: 0,
            feedback_count: 0,
            conversations_24h: 0,
            last_document_at: null,
            tone_distribution: {},
            category_distribution: {}
          };

          const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
          const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

          // Documents count - use knowledge_documents (processed documents table)
          // Handle missing table gracefully
          const documentsCheck = await checkTableExists('knowledge_documents');
          if (!documentsCheck) {
            metrics.documents_total = 0;
            metrics.documents_last_7_days = 0;
            metrics.last_document_at = null;
          } else {
            const [{ count: docTotal, error: docError }, { count: recentDocCount, error: recentDocError }, { data: latestDoc, error: lastDocError }] = await Promise.all([
              supabase.from('knowledge_documents').select('id', { count: 'exact', head: true }),
              supabase.from('knowledge_documents').select('id', { count: 'exact', head: true }).gte('created_at', sevenDaysAgo),
              supabase.from('knowledge_documents').select('created_at').order('created_at', { ascending: false }).limit(1)
            ]);

            if (docError) {
              console.warn('[SUPABASE] Failed to get document count:', docError.message);
              metrics.documents_total = 0;
            } else {
              metrics.documents_total = docTotal || 0;
            }
            
            if (recentDocError) {
              console.warn('[SUPABASE] Failed to get recent document count:', recentDocError.message);
              metrics.documents_last_7_days = 0;
            } else {
              metrics.documents_last_7_days = recentDocCount || 0;
            }
            
            if (lastDocError) {
              console.warn('[SUPABASE] Failed to get latest document:', lastDocError.message);
              metrics.last_document_at = null;
            } else if (latestDoc && latestDoc.length) {
              metrics.last_document_at = latestDoc[0].created_at;
            } else {
              metrics.last_document_at = null;
            }
          }

          // Chunk metrics - use doc_chunks table
          // Handle missing table gracefully
          const chunksCheck = await checkTableExists('doc_chunks');
          
          if (!chunksCheck) {
            metrics.chunks_total = 0;
            metrics.sales_chunks = 0;
            metrics.technical_chunks = 0;
            metrics.testimonial_chunks = 0;
            metrics.general_chunks = 0;
            metrics.tone_distribution = {};
            metrics.category_distribution = {};
          } else {
            const { count: chunkCount, error: chunkError } = await supabase
              .from('doc_chunks')
              .select('id', { count: 'exact', head: true });

            if (chunkError) {
              console.warn('[SUPABASE] Failed to get chunk count:', chunkError.message);
              metrics.chunks_total = 0;
            } else {
              metrics.chunks_total = chunkCount || 0;
            }

            const { data: chunkRows, error: chunkMetaError } = await supabase
              .from('doc_chunks')
              .select('metadata')
              .limit(500);

            if (chunkMetaError) {
              console.warn('[SUPABASE] Failed to get chunk metadata:', chunkMetaError.message);
              metrics.sales_chunks = 0;
              metrics.technical_chunks = 0;
              metrics.testimonial_chunks = 0;
              metrics.general_chunks = 0;
              metrics.tone_distribution = {};
              metrics.category_distribution = {};
            } else if (Array.isArray(chunkRows)) {
              const categoryCounts = {};
              const toneCounts = {};

              chunkRows.forEach(row => {
                const meta = parseMetadata(row.metadata);
                const category = (meta.learning_category || meta.category || meta.document_type || 'general').toLowerCase();
                const tone = (meta.tone || meta.language_tone || 'neutral').toLowerCase();

                categoryCounts[category] = (categoryCounts[category] || 0) + 1;
                toneCounts[tone] = (toneCounts[tone] || 0) + 1;
              });

              metrics.category_distribution = categoryCounts;
              metrics.tone_distribution = toneCounts;
              metrics.sales_chunks = categoryCounts.sales || categoryCounts.sales_training || categoryCounts.communication_guide || 0;
              metrics.technical_chunks = categoryCounts.technical || 0;
              metrics.testimonial_chunks = categoryCounts.testimonial || categoryCounts.case_study || 0;
              metrics.general_chunks = categoryCounts.general || 0;
            } else {
              metrics.sales_chunks = 0;
              metrics.technical_chunks = 0;
              metrics.testimonial_chunks = 0;
              metrics.general_chunks = 0;
              metrics.tone_distribution = {};
              metrics.category_distribution = {};
            }
          }

          // Conversations last 24h - handle missing table gracefully
          const conversationsCheck = await checkTableExists('epsilon_conversations');
          if (!conversationsCheck) {
            metrics.conversations_24h = 0;
          } else {
            const { count: convCount, error: convError } = await supabase
              .from('epsilon_conversations')
              .select('id', { count: 'exact', head: true })
              .gte('created_at', oneDayAgo);

            if (convError) {
              console.warn('[SUPABASE] Failed to get conversation count:', convError.message);
              metrics.conversations_24h = 0;
            } else {
              metrics.conversations_24h = convCount || 0;
            }
          }

          // Feedback metrics - handle missing table gracefully
          const feedbackCheck = await checkTableExists('epsilon_feedback');
          if (!feedbackCheck) {
            metrics.feedback_count = 0;
            metrics.avg_feedback_rating = 0;
          } else {
            const { data: feedbackRows, error: feedbackError } = await supabase
              .from('epsilon_feedback')
              .select('rating')
              .limit(5000);

            if (feedbackError) {
              console.warn('[SUPABASE] Failed to get feedback data:', feedbackError.message);
              metrics.feedback_count = 0;
              metrics.avg_feedback_rating = 0;
            } else if (Array.isArray(feedbackRows) && feedbackRows.length) {
              const validRatings = feedbackRows
                .map(row => Number(row.rating))
                .filter(value => Number.isFinite(value) && value > 0);
              metrics.feedback_count = validRatings.length;
              if (validRatings.length) {
                const total = validRatings.reduce((sum, rating) => sum + rating, 0);
                metrics.avg_feedback_rating = Number((total / validRatings.length).toFixed(2));
              } else {
                metrics.feedback_count = 0;
                metrics.avg_feedback_rating = 0;
              }
            } else {
              metrics.feedback_count = 0;
              metrics.avg_feedback_rating = 0;
            }
          }

          result = { success: true, metrics };
        } catch (error) {
          console.error('Get learning metrics error:', error);
          throw new Error(`Failed to fetch learning metrics: ${error.message || 'Unknown error'}. The metrics service is required.`);
        }
        break;
        
      // ENHANCED LEARNING DATA STORAGE ENDPOINTS
      case 'store-learning-analytics':
        try {
          const { session_id, user_id, learning_type, metric_score, user_message, epsilon_response, metadata } = sanitizedData;
          
          if (!session_id || !learning_type || metric_score === undefined) {
            throw new Error('Session ID, learning type, and metric score are required');
          }
          
          // epsilon_learning_analytics table is required
          const analyticsTableExists = await checkTableExists('epsilon_learning_analytics');
          if (!analyticsTableExists) {
            throw new Error('epsilon_learning_analytics table is required but not available');
          }
          
          _silent(`[LEARNING ANALYTICS] Storing ${learning_type} analytics`);
          
          const { data: analyticsData, error } = await supabase
            .from('epsilon_learning_analytics')
            .insert([{
              session_id,
              user_id: user_id || null,
              learning_type,
              metric_score,
              user_message: user_message?.substring(0, 500) || null,
              epsilon_response: epsilon_response?.substring(0, 500) || null,
              metadata: metadata || {}
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (error) throw error;
          
          result = { success: true, analytics_id: analyticsData.id };
        } catch (error) {
          console.error('Store learning analytics error:', error);
          throw error;
        }
        break;
        
      case 'store-model-weights':
        try {
          const { weight_type, weight_name, weight_value, learning_session_id, metadata } = sanitizedData;
          
          if (!weight_type || !weight_name || weight_value === undefined) {
            throw new Error('Weight type, name, and value are required');
          }
          
          // Models are now stored on Podrun, not Supabase
          // Return success without storing to maintain backward compatibility
          _silent(`[MODEL WEIGHTS] Weight ${weight_type}.${weight_name} = ${weight_value} (models managed on Podrun, not stored in Supabase)`);
          
          result = { success: true, message: 'Models are managed on Podrun, not stored in Supabase' };
        } catch (error) {
          console.error('Store model weights error:', error);
          throw error;
        }
        break;
        
      case 'store-learning-session':
        try {
          const { session_id, session_type, training_data_count, model_version_before, model_version_after, performance_improvement, status, metadata } = sanitizedData;
          
          if (!session_id || !session_type) {
            throw new Error('Session ID and type are required');
          }
          
          // epsilon_learning_sessions table is required
          const sessionsTableExists = await checkTableExists('epsilon_learning_sessions');
          if (!sessionsTableExists) {
            throw new Error('epsilon_learning_sessions table is required but not available');
          }
          
          _silent(`[LEARNING SESSION] Storing ${session_type} session`);
          
          const { data: sessionData, error } = await supabase
            .from('epsilon_learning_sessions')
            .insert([{
              session_id,
              session_type,
              training_data_count: training_data_count || 0,
              model_version_before: model_version_before || '1.0.0',
              model_version_after: model_version_after || '1.0.1',
              performance_improvement: performance_improvement || 0.0,
              status: status || 'active',
              metadata: metadata || {}
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (error) throw error;
          
          result = { success: true, session_id: sessionData.id };
        } catch (error) {
          console.error('Store learning session error:', error);
          throw error;
        }
        break;
        
      case 'store-learning-metric':
        try {
          const { name, value, metadata } = sanitizedData;
          
          if (!name || value === undefined || value === null) {
            throw new Error('Metric name and value are required');
          }
          
          // learning_metrics table is required
          const metricsTableExists = await checkTableExists('learning_metrics');
          if (!metricsTableExists) {
            throw new Error('learning_metrics table is required but not available');
          }
          
          _silent(`[LEARNING METRIC] Storing metric: ${name} = ${value}`);
          
          const { data: metricData, error } = await supabase
            .from('learning_metrics')
            .insert([{
              name,
              value: typeof value === 'number' ? value : parseFloat(value) || 0,
              metadata: metadata || {}
            }])
            .select('id')
            .limit(1).maybeSingle();
          
          if (error) throw error;
          
          result = { success: true, metric_id: metricData.id };
        } catch (error) {
          console.error('Store learning metric error:', error);
          throw error;
        }
        break;
        
      case 'get-trial-data':
        try {
          const { ip_address } = sanitizedData;
          
          if (!ip_address) {
            return res.status(400).json({ error: 'IP address is required' });
          }
          
          // epsilon_trial_tracking table is required
          const trialTableExists = await checkTableExists('epsilon_trial_tracking');
          if (!trialTableExists) {
            throw new Error('epsilon_trial_tracking table is required but not available');
          }
          
          // Check if trial data exists for this IP
          const { data: existingTrial, error: trialError } = await supabase
            .from('epsilon_trial_tracking')
            .select('id, ip_address, messages_used, created_at, expires_at')
            .eq('ip_address', ip_address)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();
          
          if (trialError && trialError.code !== 'PGRST116') {
            console.error('Get trial data error:', trialError);
            throw new Error(`Failed to get trial data: ${trialError.message}`);
          } else if (existingTrial) {
            result = { 
              success: true, 
              trial_data: {
                ip_address: existingTrial.ip_address,
                messages_remaining: existingTrial.messages_remaining,
                created_at: existingTrial.created_at,
                last_used: existingTrial.last_used
              }
            };
          } else {
            // No trial data found for this IP
            result = { 
              success: true, 
              trial_data: null 
            };
          }
        } catch (error) {
          console.error('Get trial data error:', error);
          throw error;
        }
        break;
        
      // RAG ENDPOINTS
      case 'search-rag':
        try {
          const { query, top_k = 6, match_threshold = 0.7 } = sanitizedData;
          
          if (!query) {
            throw new Error('Query is required');
          }
          
          _silent('[RAG] Searching for:', query.substring(0, 50) + '...');
          const normalizedQuery = normalizeEmbeddingText(query);
          const queryEmbedding = generateHashEmbedding(normalizedQuery);
          const intent = inferQueryIntent(normalizedQuery);

          const chunkSelectColumns = 'id, document_id, chunk_index, chunk_text, embedding, metadata, created_at';
          const { data: chunkRows, error: chunkError } = await supabase
            .from('doc_chunks')
            .select(chunkSelectColumns)
            .not('embedding', 'is', null)
            .limit(100);
          
          const embeddingSelectColumns = 'id, document_id, chunk_id, content, embedding, embedding_data, metadata, created_at';
          const { data: embeddingRowsData, error: embeddingError } = await supabase
            .from('document_embeddings')
            .select(embeddingSelectColumns)
            .limit(500);
          
          const allRows = [];
          
          if (!chunkError && chunkRows && chunkRows.length > 0) {
            chunkRows.forEach(row => {
              allRows.push({
                id: row.id,
                document_id: row.document_id,
                chunk_id: row.id,
                content: row.chunk_text || '',
                embedding: row.embedding,
                embedding_data: null,
                metadata: {
                  ...row.metadata,
                  chunk_index: row.chunk_index,
                  source: 'doc_chunks'
                },
                created_at: row.created_at
              });
            });
          } else if (chunkError) {
            const errorStr = chunkError?.message || chunkError?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              console.warn('[WARN] [RAG] Supabase connection issue while fetching doc_chunks');
            } else {
              console.warn('[RAG] Failed to fetch doc_chunks:', chunkError.message || 'Unknown error');
            }
          }
          
          if (embeddingError) {
            const errorStr = embeddingError?.message || embeddingError?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              console.warn('[WARN] [RAG] Supabase connection issue while fetching document_embeddings');
            } else {
              console.warn('[RAG] Failed to fetch document_embeddings:', embeddingError.message || 'Unknown error');
            }
          }
          
          if (!embeddingError && embeddingRowsData && embeddingRowsData.length > 0) {
            embeddingRowsData.forEach(row => {
              // Avoid duplicates if chunk_id matches doc_chunks id
              const isDuplicate = allRows.some(r => r.chunk_id === row.chunk_id);
              if (!isDuplicate) {
                allRows.push({
                  ...row,
                  metadata: {
                    ...row.metadata,
                    source: 'document_embeddings'
                  }
                });
              }
            });
          }
          
          if (allRows.length === 0) {
            console.warn('[RAG] No embeddings available in doc_chunks or document_embeddings tables');
            result = { 
              success: true, 
              results: [],
              query: query,
              message: 'No embeddings available yet'
            };
            break;
          }
          
          // Use allRows as embeddingRows for processing
          const embeddingRows = allRows;

          const scoredResults = [];

          embeddingRows.forEach(row => {
            const vector = parseEmbeddingVector(row);
            if (!vector) {
              return;
            }

            const similarity = cosineSimilarity(queryEmbedding, vector);
            if (!Number.isFinite(similarity)) {
              return;
            }

            const metadata = parseMetadata(row.metadata);
            const category = metadata.learning_category || metadata.category || metadata.document_type || 'general';
            const tone = metadata.tone || metadata.language_tone || 'neutral';
            const categoryWeight = getCategoryWeight(intent, category);
            const toneWeight = getToneWeight(intent, tone);
            const recencyWeight = computeRecencyWeight(row.created_at);
            const learningScore = clampScore(
              1 + (Number(metadata.learning_score) || 0 - 0.6),
              0.75,
              1.25
            );
            const noveltyBoost = clampScore(
              1 + ((metadata?.comparative_snapshot?.improvement_word_percent || 0) / 400),
              0.85,
              1.2
            );
            const stageWeight = getStageWeight(intent, metadata);
            const audienceWeight = getAudienceWeight(intent, metadata);
            const signalWeight = getSignalWeight(metadata);
            const urgencyWeight = intent.urgency === 'high'
              ? clampScore(metadata?.priority || metadata?.signals?.containsTimeline ? 1.12 : 1.06, 0.9, 1.2)
              : 1;
            const score = similarity
              * categoryWeight
              * toneWeight
              * recencyWeight
              * learningScore
              * noveltyBoost
              * stageWeight
              * audienceWeight
              * signalWeight
              * urgencyWeight;

            scoredResults.push({
              document_id: row.document_id,
              chunk_id: row.id,
              content: (row.content || '').replace(/\s+/g, ' ').trim(),
              similarity,
              learningScore,
              metadata,
              created_at: row.created_at || null,
              score,
              categoryWeight,
              toneWeight,
              recencyWeight,
              stageWeight,
              audienceWeight,
              signalWeight,
              urgencyWeight
            });
          });

          const filteredResults = scoredResults
            .filter(item => item.similarity >= Math.max(0, Math.min(1, match_threshold)))
            .sort((a, b) => b.score - a.score)
            .slice(0, Math.max(1, top_k))
            .map(item => {
              const metadata = item.metadata || {};
              const rawTitle = metadata.title || metadata.document_title || metadata.original_filename || '';
              const cleanedTitle = rawTitle
                ? rawTitle.replace(/\.[^/.]+$/, '')
                : '';
              const category = metadata.learning_category || metadata.category || metadata.document_type || 'general';

              return {
                document_id: item.document_id,
                chunk_id: item.chunk_id,
                similarity: Number(item.similarity.toFixed(4)),
                learning_score: Number((item.learningScore || 1).toFixed(3)),
                content: item.content,
                score: Number(item.score.toFixed(4)),
                metadata: {
                  ...metadata,
                  title: cleanedTitle || undefined,
                  learning_category: category,
                  chunk_index: metadata.chunkIndex ?? metadata.chunk_index ?? null,
                  tone: metadata.tone || metadata.language_tone || 'neutral'
                },
                created_at: item.created_at,
                weights: {
                  category: Number(item.categoryWeight.toFixed(3)),
                  tone: Number(item.toneWeight.toFixed(3)),
                  recency: Number(item.recencyWeight.toFixed(3)),
                  stage: Number((item.stageWeight || 1).toFixed(3)),
                  audience: Number((item.audienceWeight || 1).toFixed(3)),
                  signal: Number((item.signalWeight || 1).toFixed(3)),
                  urgency: Number((item.urgencyWeight || 1).toFixed(3)),
                  learning: Number((item.learningScore || 1).toFixed(3))
                }
              };
            });

          _silent(`[RAG] Found ${filteredResults.length} relevant chunks (threshold ${match_threshold})`);

          result = {
            success: true,
            results: filteredResults,
            query,
            intent
          };
        } catch (error) {
          console.error('RAG search error:', error);
          throw error;
        }
        break;
        
      case 'llm-complete':
        try {
          const { prompt, max_tokens = 512, temperature = 0.2, model_name = 'epsilon-rag' } = sanitizedData;
          
          if (!prompt) {
            throw new Error('Prompt is required');
          }
          
          // Use actual language model service if available
          let completion = null;
          let tokensUsed = 0;
          const startTime = Date.now();
          
          try {
            // Try to use global language engine instance if available
            const epsilonLanguageEngine = global.epsilonLanguageEngine || 
              (typeof require !== 'undefined' ? require('../core/epsilon-language-engine') : null);
            
            if (epsilonLanguageEngine && typeof epsilonLanguageEngine.generate === 'function' && 
                (typeof epsilonLanguageEngine.isModelReady === 'function' ? epsilonLanguageEngine.isModelReady() : true)) {
              const generation = await epsilonLanguageEngine.generate({
                userMessage: prompt,
                ragContext: [],
                persona: {}
              });
              
              if (generation && generation.text) {
                completion = generation.text;
                tokensUsed = generation.meta?.tokens_generated || Math.ceil(completion.length / 4);
              }
            }
          } catch (genError) {
            console.warn('[LLM] Language engine generation failed:', genError.message);
          }
          
          // Completion is required - throw error if generation failed
          if (!completion) {
            throw new Error('AI generation failed - no completion generated');
          }
          
          const responseTime = Date.now() - startTime;
          
          // Log the completion attempt - handle missing table gracefully
          const completionLogsExist = await checkTableExists('llm_completion_logs');
          if (completionLogsExist) {
            try {
              await supabase
                .from('llm_completion_logs')
                .insert([{
                  prompt: prompt.substring(0, 1000),
                  completion: completion.substring(0, 2000),
                  model_name: model_name,
                  tokens_used: tokensUsed,
                  response_time_ms: responseTime,
                  user_id: null,
                  session_id: null,
                  rag_context: {},
                  metadata: { source: 'epsilon-language-engine' }
                }]);
            } catch (logError) {
              // Logging is non-critical - continue even if it fails
              console.warn('[SUPABASE] Failed to log LLM completion:', logError.message);
            }
          } else {
            console.warn('[SUPABASE] llm_completion_logs table not available, skipping log');
          }
          
          result = { 
            success: true, 
            completion: completion,
            tokens_used: tokensUsed,
            model_name: model_name,
            response_time_ms: responseTime
          };
        } catch (error) {
          console.error('LLM completion error:', error);
          throw error;
        }
        break;
        
      case 'store-document-embedding':
        try {
          const { document_id, content, embedding, metadata = {} } = sanitizedData;
          
          if (!document_id || !content) {
            throw new Error('Document ID and content are required');
          }
          
          const docIdStr = String(document_id).trim();
          if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
            throw new Error(`Invalid document_id: ${document_id}`);
          }
          
          _silent('[RAG] Storing document embedding for:', docIdStr);
          
          const normalizedMetadata = parseMetadata(metadata);
          if (normalizedMetadata.category && !normalizedMetadata.learning_category) {
            normalizedMetadata.learning_category = normalizedMetadata.category;
          }
          if (normalizedMetadata.tone && !normalizedMetadata.language_tone) {
            normalizedMetadata.language_tone = normalizedMetadata.tone;
          }
          
          let chunkId = normalizedMetadata.chunk_id || normalizedMetadata.chunkId || null;
          // doc_chunks table - handle missing table gracefully
          const docChunksExists = await checkTableExists('doc_chunks');
          
          if (!docChunksExists) {
            console.warn('[SUPABASE] doc_chunks table not available, skipping chunk storage');
            // Continue without storing chunk - embedding will still be stored
            chunkId = null;
          } else {
            const chunkRecord = {
              document_id: docIdStr,
              chunk_index: normalizedMetadata.chunkIndex ?? normalizedMetadata.chunk_index ?? null,
              source_page: normalizedMetadata.sourcePage ?? normalizedMetadata.source_page ?? normalizedMetadata.page_number ?? null,
              chunk_text: content,
              tokens: normalizedMetadata.tokens ?? null,
              category: normalizedMetadata.learning_category || normalizedMetadata.category || null,
              tone: normalizedMetadata.language_tone || normalizedMetadata.tone || null,
              metadata: normalizedMetadata
            };
            
            try {
              if (chunkId) {
                const { error: upsertError } = await supabase
                  .from('doc_chunks')
                  .upsert([{
                    id: chunkId,
                    ...chunkRecord
                  }], { onConflict: 'id' });
                
                if (upsertError) {
                  // Check if error contains HTML (Supabase downtime)
                  const errorStr = upsertError?.message || upsertError?.toString() || '';
                  const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                     errorStr.includes('Cloudflare') || 
                                     errorStr.includes('522') || 
                                     errorStr.includes('521');
                  if (isHtmlError) {
                    console.warn('[WARN] [SUPABASE] Supabase connection issue while upserting doc chunk');
                  } else {
                    console.warn('[SUPABASE] Failed to upsert doc chunk:', upsertError.message || 'Unknown error');
                  }
                }
              } else {
                const { data: chunkData, error: chunkError } = await supabase
                  .from('doc_chunks')
                  .insert([chunkRecord])
                  .select('id')
                  .limit(1).maybeSingle();
                
                if (chunkError) {
                  // Check if error contains HTML (Supabase downtime)
                  const errorStr = chunkError?.message || chunkError?.toString() || '';
                  const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                     errorStr.includes('Cloudflare') || 
                                     errorStr.includes('522') || 
                                     errorStr.includes('521');
                  if (isHtmlError) {
                    console.warn('[WARN] [SUPABASE] Supabase connection issue while inserting doc chunk');
                  } else {
                    console.warn('[SUPABASE] Failed to insert doc chunk:', chunkError.message || 'Unknown error');
                  }
                } else if (chunkData?.id) {
                  chunkId = chunkData.id;
                }
              }
            } catch (chunkStoreError) {
              // Check if error contains HTML (Supabase downtime)
              const errorStr = chunkStoreError?.message || chunkStoreError?.toString() || '';
              const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                 errorStr.includes('Cloudflare') || 
                                 errorStr.includes('522') || 
                                 errorStr.includes('521');
              if (isHtmlError) {
                console.warn('[WARN] [SUPABASE] Supabase connection issue while storing doc chunk');
              } else {
                console.warn('[SUPABASE] Failed to store doc chunk:', chunkStoreError.message || 'Unknown error');
              }
              // Continue without chunk ID - embedding will still be stored
            }
          }
          
          const insertData = {
            document_id: docIdStr,
            content,
            metadata: {
              ...normalizedMetadata,
              chunk_id: chunkId || normalizedMetadata.chunk_id || normalizedMetadata.chunkId || null
            }
          };
          
          if (chunkId) {
            insertData.chunk_id = chunkId;
          }
          
          // Use vector column - pgvector is required
          if (embedding && Array.isArray(embedding)) {
              insertData.embedding = embedding;
          }
          
          const { data: embeddingData, error } = await supabase
            .from('document_embeddings')
            .insert([insertData])
            .select('id, chunk_id')
            .limit(1).maybeSingle();
          
          if (error) {
            const errorStr = error?.message || error?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              console.warn('[WARN] [SUPABASE] Supabase connection issue while storing document embedding');
              result = { success: false, error: 'Supabase connection timeout - please try again later' };
              break;
            }
            throw error;
          }
          
          result = { 
            success: true, 
            embedding_id: embeddingData.id,
            chunk_id: embeddingData.chunk_id || chunkId || null
          };
        } catch (error) {
          // Check if error contains HTML (Supabase downtime)
          const errorStr = error?.message || error?.toString() || '';
          const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                             errorStr.includes('Cloudflare') || 
                             errorStr.includes('522') || 
                             errorStr.includes('521');
          if (isHtmlError) {
            console.warn('[WARN] [SUPABASE] Supabase connection issue while storing document embedding');
            result = { success: false, error: 'Supabase connection timeout - please try again later' };
          } else {
            console.error('Store document embedding error:', error.message || error);
            throw error;
          }
        }
        break;
        
      case 'delete-conversation':
        try {
          const { conversation_id, user_id } = sanitizedData;
          
          if (!conversation_id) {
            result = { success: false, error: 'conversation_id is required' };
            break;
          }
          
          // Verify user owns the conversation
          if (user_id) {
            const { data: conv, error: checkError } = await supabase
              .from('conversations')
              .select('user_id')
              .eq('id', conversation_id)
              .limit(1).maybeSingle();
            
            if (checkError || !conv) {
              result = { success: false, error: 'Conversation not found' };
              break;
            }
            
            if (conv.user_id !== user_id) {
              result = { success: false, error: 'Unauthorized' };
              break;
            }
          }
          
          // Soft delete conversation (mark as deleted, keep data for history)
          const { error } = await supabase
            .from('conversations')
            .update({ 
              is_deleted: true,
              deleted_at: new Date().toISOString()
            })
            .eq('id', conversation_id);
          
          if (error) {
            console.error('Delete conversation error:', error);
            throw new Error(`Failed to delete conversation: ${error.message}`);
          }
          
          {
            // Log the change
            try {
              await supabase.from('conversation_changes').insert([{
                conversation_id: conversation_id,
                user_id: user_id || null,
                change_type: 'delete',
                old_value: null,
                new_value: null,
                metadata: { deleted_at: new Date().toISOString() },
                created_at: new Date().toISOString()
              }]);
            } catch (logError) {
              console.warn('Failed to log conversation delete (non-critical):', logError.message);
            }
            
            result = { success: true };
          }
        } catch (error) {
          console.error('Delete conversation error:', error);
          throw error;
        }
        break;
        
      case 'rename-conversation':
        try {
          const { conversation_id, name, user_id } = sanitizedData;
          
          if (!conversation_id || !name) {
            result = { success: false, error: 'conversation_id and name are required' };
            break;
          }
          
          // Verify user owns the conversation
          if (user_id) {
            const { data: conv, error: checkError } = await supabase
              .from('conversations')
              .select('user_id')
              .eq('id', conversation_id)
              .limit(1).maybeSingle();
            
            if (checkError || !conv) {
              result = { success: false, error: 'Conversation not found' };
              break;
            }
            
            if (conv.user_id !== user_id) {
              result = { success: false, error: 'Unauthorized' };
              break;
            }
          }
          
          // Get old name before updating
          const { data: oldConv } = await supabase
            .from('conversations')
            .select('conversation_name')
            .eq('id', conversation_id)
            .limit(1).maybeSingle();
          
          const oldName = oldConv?.conversation_name || null;
          const newName = name.substring(0, 255);
          
          // Update conversation name
          const { error } = await supabase
            .from('conversations')
            .update({ conversation_name: newName, updated_at: new Date().toISOString() })
            .eq('id', conversation_id);
          
          if (error) {
            console.error('Rename conversation error:', error);
            throw new Error(`Failed to rename conversation: ${error.message}`);
          }
          
          {
            // Log the change
            try {
              await supabase.from('conversation_changes').insert([{
                conversation_id: conversation_id,
                user_id: user_id || null,
                change_type: 'rename',
                old_value: oldName,
                new_value: newName,
                created_at: new Date().toISOString()
              }]);
            } catch (logError) {
              console.warn('Failed to log conversation rename (non-critical):', logError.message);
            }
            
            result = { success: true };
          }
        } catch (error) {
          console.error('Rename conversation error:', error);
          throw error;
        }
        break;
        
      case 'move-conversation-to-folder':
        try {
          const { conversation_id, folder_id, user_id } = sanitizedData;
          
          if (!conversation_id) {
            result = { success: false, error: 'conversation_id is required' };
            break;
          }
          
          // Verify user owns the conversation
          if (user_id) {
            const { data: conv, error: checkError } = await supabase
              .from('conversations')
              .select('user_id, folder_id')
              .eq('id', conversation_id)
              .limit(1).maybeSingle();
            
            if (checkError || !conv) {
              result = { success: false, error: 'Conversation not found' };
              break;
            }
            
            if (conv.user_id !== user_id) {
              result = { success: false, error: 'Unauthorized' };
              break;
            }
            
            const oldFolderId = conv.folder_id || null;
            const newFolderId = folder_id || null;
            
            // Update conversation folder
            const { error } = await supabase
              .from('conversations')
              .update({ folder_id: newFolderId, updated_at: new Date().toISOString() })
              .eq('id', conversation_id);
            
            if (error) {
              console.error('Move conversation error:', error);
              throw new Error(`Failed to move conversation: ${error.message}`);
            }
            
            {
              // Log the change
              try {
                await supabase.from('conversation_changes').insert([{
                  conversation_id: conversation_id,
                  user_id: user_id || null,
                  change_type: 'move_folder',
                  old_value: oldFolderId,
                  new_value: newFolderId,
                  created_at: new Date().toISOString()
                }]);
              } catch (logError) {
                console.warn('Failed to log conversation move (non-critical):', logError.message);
              }
              
              result = { success: true };
            }
          } else {
            result = { success: false, error: 'user_id is required' };
          }
          const { data: oldConv } = await supabase
            .from('conversations')
            .select('folder_id')
            .eq('id', conversation_id)
            .limit(1).maybeSingle();
          
          const oldFolderId = oldConv?.folder_id || null;
          const newFolderId = folder_id || null;
          
          // Update conversation folder
          const { error } = await supabase
            .from('conversations')
            .update({ folder_id: newFolderId, updated_at: new Date().toISOString() })
            .eq('id', conversation_id);
          
          if (error) {
            console.error('Move conversation error:', error);
            throw new Error(`Failed to move conversation: ${error.message}`);
          }
          
          {
            // Log the change
            try {
              await supabase.from('conversation_changes').insert([{
                conversation_id: conversation_id,
                user_id: user_id || null,
                change_type: 'move_folder',
                old_value: oldFolderId,
                new_value: newFolderId,
                created_at: new Date().toISOString()
              }]);
            } catch (logError) {
              console.warn('Failed to log conversation move (non-critical):', logError.message);
            }
            
            result = { success: true };
          }
        } catch (error) {
          console.error('Move conversation error:', error);
          throw error;
        }
        break;
        
      default:
        return res.status(400).json({ error: `Unknown action: ${action}` });
    }
    
    // Return result - ensure it's always valid
    if (!result || typeof result !== 'object') {
      result = { success: false, error: 'Invalid result format' };
    }
    
    // Don't send response if headers already sent
    if (res.headersSent) {
      console.warn('[SUPABASE PROXY] Response already sent, skipping');
      return;
    }
    
    res.json(result);
    
  } catch (error) {
    // Check if error contains HTML (Supabase downtime)
    const errorStr = error?.message || error?.toString() || JSON.stringify(error) || '';
    const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                       errorStr.includes('Cloudflare') || 
                       errorStr.includes('Error code 522') || 
                       errorStr.includes('Error code 521');
    
    if (isHtmlError) {
      console.warn('[WARN] [SUPABASE PROXY] Supabase connection issue detected (likely downtime)');
    } else {
      console.error('Supabase proxy error:', error.message || error);
      if (error?.stack && !error.stack.includes('<!DOCTYPE html>')) {
        console.error('Error stack:', error.stack);
      }
    }
    
    // Don't send response if headers already sent
    if (res.headersSent) {
      console.warn('[SUPABASE PROXY] Response already sent, cannot send error response');
      return;
    }
    
    // Return a more detailed error response
    // Use 200 status to prevent Cloudflare 502 errors - return error in response body instead
    const errorResponse = {
      success: false,
      error: isHtmlError ? 'Supabase connection timeout - please try again later' : (error?.message || String(error) || 'Unknown error occurred'),
      action: req.body?.action || 'unknown',
      timestamp: new Date().toISOString()
    };
    
    // Log additional context (only for non-HTML errors)
    if (!isHtmlError) {
      console.error('Error context:', {
        action: req.body?.action,
        data: req.body?.data,
        userAgent: req.headers['user-agent'],
        ip: req.ip
      });
    }
    
    // Return error response with 200 status to prevent Cloudflare 502 errors
    // The error is in the response body, so the client can handle it properly
    try {
      res.status(200).json(errorResponse);
    } catch (sendError) {
      // If response can't be sent, log it but don't crash
      console.error('[SUPABASE PROXY] Failed to send error response:', sendError.message);
    }
  }
});

// Function to extract training data from documents
async function extractTrainingDataFromDocument(content, documentType, title) {
  const trainingData = [];
  
  try {
    // Split content into sentences/paragraphs
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    // Create training examples based on document type
    switch (documentType) {
      case 'pricing':
        // Extract pricing-related information
        const pricingSentences = sentences.filter(s => 
          s.toLowerCase().includes('price') || 
          s.toLowerCase().includes('cost') || 
          s.toLowerCase().includes('budget') ||
          s.toLowerCase().includes('investment')
        );
        
        pricingSentences.forEach(sentence => {
          trainingData.push({
            input_text: `What does ${title} cost?`,
            expected_output: sentence.trim() + '. For detailed pricing information, I recommend scheduling a consultation to discuss your specific needs and get a customized quote.'
          });
        });
        break;
        
      case 'technical':
        // Extract technical information
        const technicalSentences = sentences.filter(s => 
          s.toLowerCase().includes('api') || 
          s.toLowerCase().includes('integration') || 
          s.toLowerCase().includes('technical') ||
          s.toLowerCase().includes('implementation')
        );
        
        technicalSentences.forEach(sentence => {
          trainingData.push({
            input_text: `How does ${title} work technically?`,
            expected_output: sentence.trim() + '. Our technical team can provide detailed implementation guidance and support throughout the integration process.'
          });
        });
        break;
        
      case 'general':
      default:
        // Extract general knowledge
        const generalSentences = sentences.slice(0, 10); // Take first 10 sentences
        
        generalSentences.forEach(sentence => {
          if (sentence.trim().length > 20) {
            trainingData.push({
              input_text: `Tell me about ${title}`,
              expected_output: sentence.trim() + '. This is part of our comprehensive approach to AI automation solutions.'
            });
          }
        });
        break;
    }
    
    // Add document-specific training examples
    trainingData.push({
      input_text: `What is ${title}?`,
      expected_output: `${title} is part of our AI automation solutions. Based on our documentation: ${content.substring(0, 200)}...`
    });
    
    trainingData.push({
      input_text: `How can ${title} help my business?`,
      expected_output: `${title} can help your business by providing AI automation solutions. Here's what our documentation says: ${content.substring(0, 200)}...`
    });
    
  } catch (error) {
    console.error('Error extracting training data:', error);
  }
  
  return trainingData;
}

module.exports = router;
module.exports.handleGetEpsilonResponse = handleGetEpsilonResponse;
