/**
 * RAG Embedding Service
 * Purpose: Generate embeddings and rich metadata for document chunks.
 * Integration: Works with existing Supabase proxy and Epsilon AI learning engine.
 */

//© 2025 Neural Ops – a division of Neural Operation's & Holding's LLC. All rights reserved.

class RAGEmbeddingService {
    constructor() {
        this.model = null;
        this.isInitialized = false;
        this.embeddingDimension = 384; // fallback dimensionality for hash embeddings
        this.chunkTokenSize = 500;
        this.chunkOverlapTokens = 150;
        this.minChunkTokens = 120;
        
        // Embedding cache for performance
        this.embeddingCache = new Map();
        this.cacheMaxSize = 10000; // Max 10k cached embeddings
        this.cacheHits = 0;
        this.cacheMisses = 0;
        this.classificationConfig = {
            categories: {
                case_study: ['case study', 'customer', 'client', 'results', 'roi', 'success story', 'deployment', 'implementation', 'before-and-after', 'outcome'],
                sales: ['offer', 'pricing', 'package', 'close', 'deal', 'sales', 'conversion', 'call to action', 'purchase', 'pipeline', 'forecast', 'quota'],
                technical: ['api', 'integration', 'architecture', 'workflow', 'endpoint', 'deployment', 'schema', 'database', 'infrastructure', 'roadmap', 'technical'],
                enablement: ['objection', 'script', 'talk track', 'pitch deck', 'battlecard', 'positioning', 'competitive', 'enablement', 'rebuttal'],
                onboarding: ['onboarding', 'kickoff', 'rollout', 'implementation plan', 'training plan', 'adoption', 'launch checklist'],
                retention: ['renewal', 'expansion', 'upsell', 'cross-sell', 'churn', 'retention', 'health score', 'success plan'],
                support: ['support', 'issue', 'bug', 'escalation', 'trouble', 'downtime', 'sla', 'incident', 'ticket'],
                testimonial: ['testimonial', 'quote', 'feedback', 'review', 'endorsement', 'reference call', 'said'],
                general: []
            },
            tones: {
                salesy: ['limited time', 'unlock', 'boost', 'win rate', 'close more', 'increase revenue', 'guarantee'],
                neutral: ['overview', 'summary', 'information', 'analysis', 'insight'],
                formal: ['therefore', 'hereby', 'shall', 'compliance', 'regulation', 'policy'],
                casual: ['hey', 'let\'s', 'quick tip', 'friendly', 'simple', 'easy']
            }
        };
        this.salesPhrases = [
            'conversion rate', 'call to action', 'close rate', 'demo', 'pipeline', 'forecast',
            'renewal', 'upsell', 'cross-sell', 'objection handling', 'deal cycle'
        ];
        this.caseStudySignals = [
            'client saw', 'customer saw', 'resulted in', 'after implementation', 'before and after',
            'reduced by', 'increase of', 'uplift', 'grew', 'improved', 'case study'
        ];
        this.stageKeywords = {
            discovery: ['discovery', 'awareness', 'initial conversation', 'first call', 'top of funnel', 'qualify'],
            evaluation: ['evaluation', 'consideration', 'comparison', 'proof of concept', 'poc', 'pilot', 'scorecard'],
            decision: ['decision', 'negotiation', 'contract', 'signature', 'closing', 'final approval'],
            onboarding: ['onboarding', 'kickoff', 'implementation plan', 'go-live', 'launch', 'enablement plan'],
            renewal: ['renewal', 'expansion', 'upsell', 'cross-sell', 'qbr', 'health review'],
            support: ['support', 'issue', 'incident', 'escalation', 'sla', 'post-mortem']
        };
        this.audienceKeywords = {
            executive: ['executive', 'c-suite', 'ceo', 'cfo', 'coo', 'founder', 'board', 'vp'],
            operations: ['operations', 'ops', 'manager', 'director', 'lead', 'program manager'],
            technical: ['developer', 'engineer', 'architect', 'technical', 'product', 'it', 'cto'],
            sales: ['sales', 'account executive', 'ae', 'sdr', 'bdr', 'bizdev'],
            marketing: ['marketing', 'demand gen', 'growth', 'brand', 'campaign'],
            customer_success: ['customer success', 'cs', 'account manager', 'success manager', 'csm']
        };
        this.urgencyKeywords = ['urgent', 'asap', 'immediately', 'right away', 'deadline', 'critical', 'priority', 'today'];
    }

    async initialize() {
        try {
            this.isInitialized = true;
            return true;
        } catch (error) {
            console.error('[RAG EMBEDDING] Failed to initialize embedding service:', error);
            return false;
        }
    }

    async generateEmbedding(text) {
        if (!text || typeof text !== 'string') {
            throw new Error('text must be a non-empty string');
        }
        if (text.length > 100000) {
            text = text.substring(0, 100000);
        }
        
        if (!this.isInitialized) {
            throw new Error('Embedding service not initialized');
        }

        try {
            const cleanText = this.cleanText(text);
            if (!cleanText.trim()) {
                throw new Error('Empty text provided for embedding');
            }

            // Check cache first
            const cacheKey = cleanText.toLowerCase().trim();
            if (this.embeddingCache.has(cacheKey)) {
                this.cacheHits++;
                return this.embeddingCache.get(cacheKey);
            }

            // Generate embedding
            this.cacheMisses++;
            const embedding = this.generateHashEmbedding(cleanText);
            
            // Cache the embedding (with size limit)
            if (this.embeddingCache.size >= this.cacheMaxSize) {
                // Remove oldest entry (simple FIFO)
                const firstKey = this.embeddingCache.keys().next().value;
                this.embeddingCache.delete(firstKey);
            }
            this.embeddingCache.set(cacheKey, embedding);
            
            return embedding;
        } catch (error) {
            console.error('[RAG EMBEDDING] Error generating embedding:', error);
            throw error;
        }
    }
    
    // Get cache statistics
    getCacheStats() {
        const total = this.cacheHits + this.cacheMisses;
        const hitRate = total > 0 ? (this.cacheHits / total) * 100 : 0;
        return {
            size: this.embeddingCache.size,
            maxSize: this.cacheMaxSize,
            hits: this.cacheHits,
            misses: this.cacheMisses,
            hitRate: hitRate.toFixed(2) + '%'
        };
    }
    
    // Clear cache
    clearCache() {
        this.embeddingCache.clear();
        this.cacheHits = 0;
        this.cacheMisses = 0;
    }

    generateHashEmbedding(text) {
        if (!text || typeof text !== 'string') {
            return new Array(this.embeddingDimension).fill(0);
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        
        const words = text.toLowerCase().split(/\s+/);
        const embedding = new Array(this.embeddingDimension).fill(0);
        
        words.forEach(word => {
            if (!word) return;
            let hash = 0;
            for (let i = 0; i < word.length; i++) {
                hash = ((hash << 5) - hash + word.charCodeAt(i)) & 0xffffffff;
            }
            const index = Math.abs(hash) % this.embeddingDimension;
            embedding[index] += 1;
        });
        
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return embedding.map(val => (magnitude > 0 ? val / magnitude : 0));
    }

    tokenize(text) {
        if (!text || typeof text !== 'string') {
            return [];
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        
        return text
            .replace(/\s+/g, ' ')
            .split(' ')
            .map(token => token.trim())
            .filter(Boolean);
    }

    chunkText(text, options = {}) {
        if (!text || typeof text !== 'string') {
            return [];
        }
        if (text.length > 10000000) {
            text = text.substring(0, 10000000);
        }
        if (!options || typeof options !== 'object' || Array.isArray(options)) {
            options = {};
        }
        
        const maxTokens = options.tokens || this.chunkTokenSize;
        const overlap = options.overlap || this.chunkOverlapTokens;
        
        if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 10000) {
            throw new Error('maxTokens must be an integer between 1 and 10000');
        }
        if (!Number.isInteger(overlap) || overlap < 0 || overlap >= maxTokens) {
            throw new Error('overlap must be an integer between 0 and maxTokens');
        }
        
        const tokens = this.tokenize(text);
        const chunks = [];
        let chunkIndex = 0;

        for (let start = 0; start < tokens.length; start += Math.max(1, maxTokens - overlap)) {
            const end = Math.min(tokens.length, start + maxTokens);
            const tokenSlice = tokens.slice(start, end);
            const chunkText = tokenSlice.join(' ').trim();
            if (!chunkText) continue;

            const tokenCount = tokenSlice.length;
            if (tokenCount < this.minChunkTokens && tokenCount !== tokens.length) {
                continue;
            }

            chunks.push({
                text: chunkText,
                startIndex: start,
                endIndex: end,
                tokenCount,
                metadata: {
                    chunkIndex: chunkIndex++,
                    totalChunks: Math.ceil(tokens.length / Math.max(1, maxTokens - overlap))
                }
            });

            if (end >= tokens.length) break;
        }

        if (!chunks.length && text.trim()) {
                chunks.push({
                text: text.trim(),
                startIndex: 0,
                endIndex: tokens.length,
                tokenCount: tokens.length,
                    metadata: {
                    chunkIndex: 0,
                    totalChunks: 1
                    }
                });
        }
        
        return chunks;
    }

    cleanText(text) {
        if (!text || typeof text !== 'string') {
            return '';
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        
        return text
            .replace(/\s+/g, ' ')
            .replace(/[^\w\s.,!?;:-]/g, '')
            .trim()
            .substring(0, 4000);
    }

    classifyChunk(chunkText) {
        if (!chunkText || typeof chunkText !== 'string') {
            return {
                category: 'general',
                confidence: 0,
                category_scores: {},
                keyword_hits: [],
                signals: {}
            };
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        
        const normalized = chunkText.toLowerCase();
        const scores = {};
        const keywordHits = new Set();

        Object.entries(this.classificationConfig.categories).forEach(([category, keywords]) => {
            scores[category] = 0;
            keywords.forEach(keyword => {
                if (!keyword) return;
                const pattern = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const regex = new RegExp(`\\b${pattern}\\b`, 'gi');
                const matches = normalized.match(regex);
                if (matches) {
                    scores[category] += matches.length;
                    keywordHits.add(keyword.toLowerCase());
                }
            });
        });

        let bestCategory = 'general';
        let bestScore = 0;
        Object.entries(scores).forEach(([category, score]) => {
            if (score > bestScore) {
                bestCategory = category;
                bestScore = score;
            }
        });

        const hasPercent = /\b\d{1,3}%\b/.test(normalized);
        const hasDollar = /\$\d/.test(normalized);
        const mentionsClient = normalized.includes('client') || normalized.includes('customer') || normalized.includes('partner');
        const salesPhraseHit = this.salesPhrases.some(phrase => normalized.includes(phrase));
        const caseStudyPhraseHit = this.caseStudySignals.some(phrase => normalized.includes(phrase));

        if (mentionsClient && (hasPercent || hasDollar || caseStudyPhraseHit)) {
            bestCategory = 'case_study';
            bestScore = Math.max(bestScore, 3);
        } else if (salesPhraseHit) {
            bestCategory = 'sales';
            bestScore = Math.max(bestScore, 2);
        }

        if (bestScore === 0) {
            if (normalized.includes('case study') || normalized.includes('client')) {
                bestCategory = 'case_study';
                bestScore = 1;
            } else if (normalized.includes('api') || normalized.includes('integration')) {
                bestCategory = 'technical';
                bestScore = 1;
            } else if (normalized.includes('price') || normalized.includes('plan')) {
                bestCategory = 'sales';
                bestScore = 1;
            }
        }

        const confidence = Math.min(1, bestScore / 4 || 0.3);
        return {
            category: bestCategory,
            confidence,
            category_scores: scores,
            keyword_hits: Array.from(keywordHits),
            signals: {
                hasPercent,
                hasDollar,
                mentionsClient,
                salesPhraseHit,
                caseStudyPhraseHit
            }
        };
    }

    detectTone(chunkText) {
        if (!chunkText || typeof chunkText !== 'string') {
            return 'neutral';
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        
        const normalized = chunkText.toLowerCase();
        const toneScores = {};

        Object.entries(this.classificationConfig.tones).forEach(([tone, keywords]) => {
            toneScores[tone] = 0;
            keywords.forEach(keyword => {
                if (!keyword) return;
                const regex = new RegExp(`\\b${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
                const matches = normalized.match(regex);
                if (matches) {
                    toneScores[tone] += matches.length;
                }
            });
        });

        let bestTone = 'neutral';
        let bestScore = 0;

        Object.entries(toneScores).forEach(([tone, score]) => {
            if (score > bestScore) {
                bestTone = tone;
                bestScore = score;
            }
        });

        if (bestScore === 0) {
            if (normalized.includes('thank') || normalized.includes('hello')) {
                bestTone = 'casual';
            } else if (normalized.includes('therefore') || normalized.includes('compliance')) {
                bestTone = 'formal';
            } else if (normalized.includes('boost') || normalized.includes('unlock')) {
                bestTone = 'salesy';
            }
        }

        return bestTone;
    }

    extractSignals(chunkText) {
        if (!chunkText || typeof chunkText !== 'string') {
            return {
                percent_values: [],
                currency_values: [],
                timeline_references: [],
                containsOutcomeLanguage: false,
                containsNarrative: false,
                containsCallToAction: false,
                containsUrgencyLanguage: false,
                containsCustomerMention: false,
                containsProductMention: false
            };
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        
        const normalized = chunkText.toLowerCase();
        const percentMatches = chunkText.match(/\b\d{1,3}%\b/g) || [];
        const currencyMatches = chunkText.match(/\$\s?\d+(?:[,.\d+]*)/g) || [];
        const timelineMatches = chunkText.match(/\b\d{1,2}\s?(?:weeks|months|days|quarters|years)\b/gi) || [];

        const containsOutcomeLanguage = /(increase|decrease|boost|reduced|accelerated|improved|grew|uplifted)/i.test(chunkText);
        const containsNarrative = /(before|after|previously|within)/i.test(chunkText);
        const containsCallToAction = /(let's|we can|you can|schedule|connect|reach out)/i.test(chunkText);
        const containsUrgencyLanguage = this.urgencyKeywords.some(keyword => normalized.includes(keyword));

        return {
            percent_values: percentMatches.map(val => val.trim()),
            currency_values: currencyMatches.map(val => val.trim()),
            timeline_references: timelineMatches.map(val => val.trim()),
            containsOutcomeLanguage,
            containsNarrative,
            containsCallToAction,
            containsUrgencyLanguage,
            containsCustomerMention: /(client|customer|partner|brand|team)/i.test(chunkText),
            containsProductMention: /(solution|platform|suite|module|feature)/i.test(chunkText)
        };
    }

    detectStage(chunkText, documentMetadata = {}) {
        if (!chunkText || typeof chunkText !== 'string') {
            return { stage: null, confidence: 0, matches: [], derived_from_document: false };
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        if (!documentMetadata || typeof documentMetadata !== 'object' || Array.isArray(documentMetadata)) {
            documentMetadata = {};
        }
        
        const normalized = chunkText.toLowerCase();
        const docStageRaw =
            documentMetadata.sales_stage ||
            documentMetadata.stage ||
            documentMetadata.funnel_stage ||
            documentMetadata.pipeline_stage ||
            documentMetadata.customer_journey_stage ||
            '';

        const docStage = docStageRaw ? docStageRaw.toString().toLowerCase() : '';
        const stageScores = {};
        const matchMap = {};

        Object.entries(this.stageKeywords).forEach(([stage, keywords]) => {
            let score = 0;
            const matches = [];
            keywords.forEach(keyword => {
                if (!keyword) return;
                if (normalized.includes(keyword)) {
                    score += keyword.split(/\s+/).length > 1 ? 1.2 : 1;
                    matches.push(keyword);
                }
            });
            if (score > 0) {
                stageScores[stage] = score;
                matchMap[stage] = matches;
            }
        });

        if (docStage) {
            stageScores[docStage] = (stageScores[docStage] || 0) + 1.5;
            if (!matchMap[docStage]) {
                matchMap[docStage] = [];
            }
        }

        let bestStage = docStage || null;
        let bestScore = docStage ? stageScores[docStage] || 0 : 0;

        Object.entries(stageScores).forEach(([stage, score]) => {
            if (score > bestScore) {
                bestStage = stage;
                bestScore = score;
            }
        });

        if (!bestStage) {
            return { stage: null, confidence: 0, matches: [], derived_from_document: false };
        }

        const confidence = Math.min(1, (bestScore / 4) + (docStage && bestStage === docStage ? 0.1 : 0));
        return {
            stage: bestStage,
            confidence: Number(confidence.toFixed(3)),
            matches: matchMap[bestStage] || [],
            derived_from_document: Boolean(docStage && (!matchMap[bestStage] || !matchMap[bestStage].length))
        };
    }

    detectAudience(chunkText, documentMetadata = {}) {
        if (!chunkText || typeof chunkText !== 'string') {
            return { audience: null, confidence: 0, matches: [], derived_from_document: false };
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        if (!documentMetadata || typeof documentMetadata !== 'object' || Array.isArray(documentMetadata)) {
            documentMetadata = {};
        }
        
        const normalized = chunkText.toLowerCase();
        const docAudienceRaw =
            documentMetadata.target_persona ||
            documentMetadata.audience ||
            documentMetadata.persona ||
            documentMetadata.role ||
            documentMetadata.stakeholder ||
            '';

        const docAudience = docAudienceRaw ? docAudienceRaw.toString().toLowerCase() : '';
        const audienceScores = {};
        const matchMap = {};

        Object.entries(this.audienceKeywords).forEach(([audience, keywords]) => {
            let score = 0;
            const matches = [];
            keywords.forEach(keyword => {
                if (!keyword) return;
                if (normalized.includes(keyword)) {
                    score += 1;
                    matches.push(keyword);
                }
            });
            if (score > 0) {
                audienceScores[audience] = score;
                matchMap[audience] = matches;
            }
        });

        if (docAudience) {
            audienceScores[docAudience] = (audienceScores[docAudience] || 0) + 1.2;
            if (!matchMap[docAudience]) {
                matchMap[docAudience] = [];
            }
        }

        let bestAudience = docAudience || null;
        let bestScore = docAudience ? audienceScores[docAudience] || 0 : 0;

        Object.entries(audienceScores).forEach(([audience, score]) => {
            if (score > bestScore) {
                bestAudience = audience;
                bestScore = score;
            }
        });

        if (!bestAudience) {
            return { audience: null, confidence: 0, matches: [], derived_from_document: false };
        }

        const confidence = Math.min(1, (bestScore / 4) + (docAudience && bestAudience === docAudience ? 0.08 : 0));
        return {
            audience: bestAudience,
            confidence: Number(confidence.toFixed(3)),
            matches: matchMap[bestAudience] || [],
            derived_from_document: Boolean(docAudience && (!matchMap[bestAudience] || !matchMap[bestAudience].length))
        };
    }

    detectUrgency(chunkText, signals = {}) {
        if (!chunkText || typeof chunkText !== 'string') {
            return { urgency: 'normal', confidence: 0.4, triggers: [] };
        }
        if (chunkText.length > 100000) { // Prevent DoS
            chunkText = chunkText.substring(0, 100000);
        }
        if (!signals || typeof signals !== 'object' || Array.isArray(signals)) {
            signals = {};
        }
        
        const normalized = chunkText.toLowerCase();
        const triggers = new Set();

        this.urgencyKeywords.forEach(keyword => {
            if (normalized.includes(keyword)) {
                triggers.add(keyword);
            }
        });

        if (signals.timeline_references && signals.timeline_references.length) {
            triggers.add('timeline_reference');
        }
        if (signals.containsUrgencyLanguage) {
            triggers.add('urgency_phrase');
        }

        const triggerCount = triggers.size;
        const urgency = triggerCount > 0 ? 'high' : 'normal';
        const baseConfidence = triggerCount > 0 ? 0.7 + Math.min(0.2, triggerCount * 0.05) : 0.4;

        return {
            urgency,
            confidence: Number(Math.min(1, baseConfidence).toFixed(3)),
            triggers: Array.from(triggers)
        };
    }

    async processDocument(documentId, content, metadata = {}) {
        if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
            throw new Error('documentId must be a non-empty string or number');
        }
        if (!content || typeof content !== 'string') {
            throw new Error('content must be a non-empty string');
        }
        if (content.length > 10000000) { // Prevent DoS (10MB max)
            content = content.substring(0, 10000000);
        }
        if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
            metadata = {};
        }
        
        try {
            const chunks = this.chunkText(content, {
                tokens: this.chunkTokenSize,
                overlap: this.chunkOverlapTokens
            });
            const embeddings = [];
            
            for (const chunk of chunks) {
                const embedding = await this.generateEmbedding(chunk.text);
                const classification = this.classifyChunk(chunk.text);
                const tone = this.detectTone(chunk.text);
                const signals = this.extractSignals(chunk.text);
                const stageInfo = this.detectStage(chunk.text, metadata);
                const audienceInfo = this.detectAudience(chunk.text, metadata);
                const urgencyInfo = this.detectUrgency(chunk.text, signals);

                const combinedSignals = {
                    ...(metadata.signals || {}),
                    ...signals,
                    stage_matches: stageInfo.matches,
                    audience_matches: audienceInfo.matches,
                    urgency_triggers: urgencyInfo.triggers
                };

                const keywordAccumulator = new Set(
                    [
                        ...(metadata.keyword_hits || []),
                        ...(classification.keyword_hits || []),
                        ...stageInfo.matches,
                        ...audienceInfo.matches,
                        ...urgencyInfo.triggers
                    ]
                        .filter(Boolean)
                        .map(entry => entry.toString().toLowerCase())
                );

                const chunkMetadata = {
                    ...metadata,
                    ...chunk.metadata,
                    tokens: chunk.tokenCount,
                    category: classification.category,
                    category_confidence: classification.confidence,
                    category_scores: classification.category_scores,
                    signals: combinedSignals,
                    tone,
                    processedAt: new Date().toISOString(),
                    keyword_hits: Array.from(keywordAccumulator)
                };

                if (stageInfo.stage) {
                    chunkMetadata.sales_stage = stageInfo.stage;
                    chunkMetadata.sales_stage_confidence = stageInfo.confidence;
                }

                if (audienceInfo.audience) {
                    chunkMetadata.target_persona = audienceInfo.audience;
                    chunkMetadata.target_persona_confidence = audienceInfo.confidence;
                }

                chunkMetadata.urgency = urgencyInfo.urgency;
                chunkMetadata.urgency_confidence = urgencyInfo.confidence;
                chunkMetadata.intent_summary = {
                    stage: stageInfo.stage,
                    stage_matches: stageInfo.matches,
                    audience: audienceInfo.audience,
                    audience_matches: audienceInfo.matches,
                    urgency: urgencyInfo.urgency,
                    urgency_triggers: urgencyInfo.triggers
                };

                embeddings.push({
                    documentId,
                    content: chunk.text,
                    embedding,
                    metadata: chunkMetadata
                });
            }
            
            return embeddings;
        } catch (error) {
            console.error('[RAG EMBEDDING] Error processing document:', error);
            throw error;
        }
    }

    async searchSimilar(query, embeddings, topK = 6, threshold = 0.7) {
        if (!query || typeof query !== 'string') {
            throw new Error('query must be a non-empty string');
        }
        if (query.length > 10000) { // Prevent DoS
            query = query.substring(0, 10000);
        }
        if (!embeddings || !Array.isArray(embeddings)) {
            throw new Error('embeddings must be a non-empty array');
        }
        if (embeddings.length > 10000) { // Prevent DoS
            embeddings = embeddings.slice(0, 10000);
        }
        if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
            topK = 6; // Default
        }
        if (typeof threshold !== 'number' || threshold < 0 || threshold > 1 || !isFinite(threshold)) {
            threshold = 0.7; // Default
        }
        
        if (!this.isInitialized) {
            throw new Error('Embedding service not initialized');
        }

        try {
            const queryEmbedding = await this.generateEmbedding(query);
            const similarities = embeddings.map(embedding => {
                const similarity = this.cosineSimilarity(queryEmbedding, embedding.embedding);
                return {
                    ...embedding,
                    similarity
                };
            });
            
            const results = similarities
                .filter(item => item.similarity >= threshold)
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, topK);
            
            return results;
        } catch (error) {
            console.error('[RAG EMBEDDING] Error searching similar documents:', error);
            throw error;
        }
    }

    cosineSimilarity(a, b) {
        if (!Array.isArray(a) || !Array.isArray(b)) {
            throw new Error('Both arguments must be arrays');
        }
        if (a.length !== b.length) {
            throw new Error('Vectors must have the same length');
        }
        if (a.length === 0 || a.length > 10000) { // Prevent DoS
            throw new Error('Vector length must be between 1 and 10000');
        }
        
        // Safety check: validate array elements are numbers
        for (let i = 0; i < a.length; i++) {
            if (typeof a[i] !== 'number' || !isFinite(a[i])) {
                throw new Error('All vector elements must be finite numbers');
            }
            if (typeof b[i] !== 'number' || !isFinite(b[i])) {
                throw new Error('All vector elements must be finite numbers');
            }
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
}

if (typeof window !== 'undefined') {
    window.RAGEmbeddingService = RAGEmbeddingService;
} else if (typeof module !== 'undefined') {
    module.exports = RAGEmbeddingService;
}
