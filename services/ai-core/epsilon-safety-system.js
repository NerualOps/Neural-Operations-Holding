/**
 * Epsilon AI Safety System
 * ===================
 * Comprehensive safety and guardrails for Epsilon AI
 * NO external AI dependencies
 * 
 * Features:
 * - Output validator
 * - Self-check layer
 * - Toxicity filter
 * - Consistency engine
 * - Hallucination detector
 * - Response validation
 */

class EpsilonSafetySystem {
  constructor() {
    // Toxicity patterns
    this.toxicityPatterns = [
      /\b(kill|murder|suicide|bomb|terrorist|hack|steal|illegal|drugs?|weapon)\w*\b/gi,
      /\b(hate|violence|harm|hurt|attack|destroy)\w*\b/gi,
      /\b(sex|porn|nude|explicit)\w*\b/gi,
      /\b(scam|fraud|cheat|lie|deceive)\w*\b/gi,
      /\b(racist|sexist|discriminat)\w*\b/gi
    ];
    
    // Inconsistency patterns
    this.inconsistencyPatterns = [
      /(yes|no).*but.*(no|yes)/gi,
      /(always|never).*but.*(sometimes|occasionally)/gi,
      /(all|every).*except/gi
    ];
    
    // Hallucination indicators
    this.hallucinationIndicators = [
      /(according to|studies show|research indicates).*but no source/gi,
      /(definitely|absolutely|100%).*without evidence/gi,
      /(proven|confirmed).*but unverified/gi
    ];
    
    // Factual claim patterns (need verification)
    this.factualClaimPatterns = [
      /\b\d{4}\b.*(happened|occurred|discovered)/g, // Years
      /\b\d+%?\b.*(of|in|from)/g, // Statistics
      /\b(always|never|all|none)\b/g // Absolute claims
    ];
  }

  /**
   * Validate output (main safety check)
   */
  validateOutput(text, context = {}) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return {
        passed: false,
        score: 0,
        checks: {},
        safe: false,
        error: 'Invalid text input'
      };
    }
    
    const checks = {
      toxicity: this.checkToxicity(text),
      consistency: this.checkConsistency(text, context),
      hallucination: this.checkHallucination(text, context),
      factual: this.checkFactualClaims(text, context),
      length: this.checkLength(text),
      coherence: this.checkCoherence(text)
    };
    
    const passed = Object.values(checks).every(check => check.passed);
    const score = this._computeSafetyScore(checks);
    
    return {
      passed,
      score,
      checks,
      safe: passed && score > 0.7
    };
  }

  /**
   * Check for toxicity
   */
  checkToxicity(text) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, matches: [], severity: 'high' };
    }
    
    const matches = [];
    
    for (const pattern of this.toxicityPatterns) {
      const found = text.match(pattern);
      if (found) {
        matches.push(...found);
      }
    }
    
    return {
      passed: matches.length === 0,
      matches,
      severity: matches.length > 0 ? 'high' : 'none'
    };
  }

  /**
   * Check consistency
   */
  checkConsistency(text, context) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, inconsistencies: ['invalid_input'] };
    }
    
    const inconsistencies = [];
    
    // Check against context
    if (context.previousResponse) {
      const textLower = text.toLowerCase();
      const prevLower = context.previousResponse.toLowerCase();
      
      // Check for contradictions
      if (this._hasContradiction(textLower, prevLower)) {
        inconsistencies.push('contradicts_previous');
      }
    }
    
    // Check internal consistency
    for (const pattern of this.inconsistencyPatterns) {
      if (pattern.test(text)) {
        inconsistencies.push('internal_inconsistency');
      }
    }
    
    return {
      passed: inconsistencies.length === 0,
      inconsistencies
    };
  }

  /**
   * Check for hallucination
   */
  checkHallucination(text, context) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, indicators: ['invalid_input'], confidence: 0 };
    }
    
    const indicators = [];
    
    // Check for unverified claims
    for (const pattern of this.hallucinationIndicators) {
      if (pattern.test(text)) {
        indicators.push('unverified_claim');
      }
    }
    
    // Check for factual claims without evidence
    let factualClaimCount = 0;
    for (const pattern of this.factualClaimPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        factualClaimCount += matches.length;
      }
    }
    if (factualClaimCount > 2 && !context.hasEvidence) {
      indicators.push('multiple_unverified_facts');
    }
    
    return {
      passed: indicators.length === 0,
      indicators,
      confidence: indicators.length === 0 ? 0.9 : 0.5
    };
  }

  /**
   * Check factual claims
   */
  checkFactualClaims(text, context) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, claims: 0, needsVerification: false };
    }
    
    let claimCount = 0;
    for (const pattern of this.factualClaimPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        claimCount += matches.length;
      }
    }
    const hasEvidence = context.hasEvidence || false;
    
    return {
      passed: claimCount === 0 || hasEvidence,
      claims: claimCount,
      needsVerification: claimCount > 0 && !hasEvidence
    };
  }

  /**
   * Check length
   */
  checkLength(text) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, length: 0, wordCount: 0 };
    }
    
    const length = text.length;
    const wordCount = text.split(/\s+/).length;
    
    // Reasonable limits
    const maxLength = 2000;
    const minLength = 10;
    const maxWords = 400;
    
    return {
      passed: length >= minLength && length <= maxLength && wordCount <= maxWords,
      length,
      wordCount
    };
  }

  /**
   * Check coherence
   */
  checkCoherence(text) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return { passed: false, reason: 'invalid_input', avgLength: 0, sentenceCount: 0 };
    }
    
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    if (sentences.length < 1) {
      return { passed: false, reason: 'no_sentences' };
    }
    
    // Check sentence length
    const avgLength = sentences.reduce((sum, s) => sum + s.length, 0) / sentences.length;
    const tooShort = sentences.filter(s => s.length < 10).length;
    const tooLong = sentences.filter(s => s.length > 200).length;
    
    return {
      passed: avgLength > 20 && avgLength < 150 && tooShort < sentences.length / 2 && tooLong === 0,
      avgLength,
      sentenceCount: sentences.length
    };
  }

  /**
   * Filter toxic content
   */
  filterToxic(text) {
    // Safety check: validate input
    if (typeof text !== 'string') {
      return '';
    }
    
    let filtered = text;
    
    for (const pattern of this.toxicityPatterns) {
      filtered = filtered.replace(pattern, '[FILTERED]');
    }
    
    // If too much filtered, return safe alternative
    if (text.length === 0) {
      return text;
    }
    const filteredRatio = (text.length - filtered.replace(/\[FILTERED\]/g, '').length) / text.length;
    if (filteredRatio > 0.2) {
      return "I can't provide information on that topic. Is there something else about business automation, website development, or AI strategy I can help you with?";
    }
    
    return filtered;
  }

  /**
   * Self-check: Validate before responding
   */
  selfCheck(response, context) {
    // Safety check: validate input
    if (typeof response !== 'string') {
      return {
        safe: true,
        response: "I want to make sure I give you accurate and helpful information. Could you rephrase your question?",
        fixed: false,
        fallback: true
      };
    }
    
    const validation = this.validateOutput(response, context);
    
    if (!validation.safe) {
      // Attempt to fix
      let fixed = this.filterToxic(response);
      
      // Re-validate
      const revalidation = this.validateOutput(fixed, context);
      
      if (revalidation.safe) {
        return { safe: true, response: fixed, original: response, fixed: true };
      } else {
        // Return safe fallback
        return {
          safe: true,
          response: "I want to make sure I give you accurate and helpful information. Could you rephrase your question?",
          original: response,
          fixed: false,
          fallback: true
        };
      }
    }
    
    return { safe: true, response, fixed: false };
  }

  /**
   * Check for contradiction
   */
  _hasContradiction(text1, text2) {
    const contradictions = [
      ['yes', 'no'],
      ['always', 'never'],
      ['all', 'none'],
      ['true', 'false'],
      ['correct', 'incorrect']
    ];
    
    for (const [word1, word2] of contradictions) {
      if ((text1.includes(word1) && text2.includes(word2)) ||
          (text1.includes(word2) && text2.includes(word1))) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Compute overall safety score
   */
  _computeSafetyScore(checks) {
    let score = 1.0;
    
    if (!checks.toxicity.passed) score -= 0.4;
    if (!checks.consistency.passed) score -= 0.2;
    if (!checks.hallucination.passed) score -= 0.2;
    if (!checks.factual.passed) score -= 0.1;
    if (!checks.length.passed) score -= 0.05;
    if (!checks.coherence.passed) score -= 0.05;
    
    return Math.max(0, score);
  }
}

module.exports = EpsilonSafetySystem;


