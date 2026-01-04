/**
 * RAG LLM Service
 * Purpose: Local LLM integration for generating responses
 * Integration: Works with Ollama or fallback to heuristic responses
 * Dependencies: Fetch API for Ollama communication
 */

//© 2025 Neural Ops – a division of Neural Operation's & Holding's LLC. All rights reserved.

// Note: epsilonLanguageEngine is optional and may not be available in browser context
// It will be checked dynamically when needed
let epsilonLanguageEngine = null;
if (typeof window !== 'undefined' && window.epsilonLanguageEngine) {
    epsilonLanguageEngine = window.epsilonLanguageEngine;
} else if (typeof require !== 'undefined') {
    try {
        epsilonLanguageEngine = require('../core/epsilon-language-engine');
    } catch (e) {
        // Not available in this context
    }
}

class RAGLLMService {
    constructor() {
        this.modelName = 'fallback-heuristic';
        this.maxTokens = 512;
        this.temperature = 0.2;
        this.isOllamaAvailable = false;
        this.fallbackResponses = this.initializeFallbackResponses();
        this.lastPersona = null;
        this.mathExpressionPattern = /^[-+/*()\d.\s]+$/;
    }

    handleGeneralIntent(userMessage = '', persona = null) {
        if (!userMessage || typeof userMessage !== 'string') {
            return null;
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        if (persona !== null && (typeof persona !== 'object' || Array.isArray(persona))) {
            persona = null;
        }
        
        const trimmed = userMessage.trim();
        if (!trimmed) {
            return null;
        }

        // Check for quick greetings or thanks
        if (/^(hi|hey|hello|good (morning|afternoon|evening)|what's up)\b/i.test(trimmed)) {
            const responses = [
                "Hey there—ready when you are.",
                "Hi! What can I help you move forward today?",
                "Hello! What's the next thing you'd like us to tackle?"
            ];
            return {
                completion: this.refineGeneratedText(responses[Math.floor(Math.random() * responses.length)], persona),
                tokensUsed: 0,
                model: 'conversation-heuristic',
                source: 'general'
            };
        }

        // Quick gratitude acknowledgement
        if (/thank(s| you)/i.test(trimmed)) {
            return {
                completion: this.refineGeneratedText("Happy to help. Just say the word if there's anything else.", persona),
                tokensUsed: 0,
                model: 'conversation-heuristic',
                source: 'general'
            };
        }

        // Simple math evaluation
        if (this.mathExpressionPattern.test(trimmed) && /[\d)(]/.test(trimmed)) {
            try {
                const result = this.evaluateMathExpression(trimmed);
                if (result !== null) {
                    return {
                        completion: this.refineGeneratedText(`That comes out to ${result}.`, persona),
                        tokensUsed: 0,
                        model: 'conversation-heuristic',
                        source: 'general-math'
                    };
                }
            } catch (error) {
                console.warn('[RAG LLM] Math evaluation failed:', error.message);
            }
        }

        // Date / time cues
        if (/^what(?:'s| is) the date\b/i.test(trimmed)) {
            const today = new Date();
            const formatted = today.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
            return {
                completion: this.refineGeneratedText(`Today is ${formatted}.`, persona),
                tokensUsed: 0,
                model: 'conversation-heuristic',
                source: 'general-date'
            };
        }

        if (/^what time is it\b/i.test(trimmed)) {
            const now = new Date();
            const formatted = now.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
            return {
                completion: this.refineGeneratedText(`It's ${formatted} where I'm operating.`, persona),
                tokensUsed: 0,
                model: 'conversation-heuristic',
                source: 'general-time'
            };
        }

        // Casual request fallback
        if (/tell me a joke/i.test(trimmed)) {
            return {
                completion: this.refineGeneratedText("Here's one from the ops playbook: Why did the CRM cross the road? To get to the best-performing pipeline on the other side.", persona),
                tokensUsed: 0,
                model: 'conversation-heuristic',
                source: 'small-talk'
            };
        }

        return null;
    }

    evaluateMathExpression(expression) {
        if (!expression || typeof expression !== 'string') {
            return null;
        }
        if (expression.length > 1000) {
            return null;
        }
        
        const sanitized = expression.replace(/[^-+/*()\d.\s]/g, '');
        if (!sanitized || !this.mathExpressionPattern.test(sanitized)) {
            return null;
        }

        try {
            // eslint-disable-next-line no-new-func
            const result = Function(`"use strict"; return (${sanitized});`)();
            if (Number.isFinite(result)) {
                return Number(result.toFixed(6)).toString();
            }
        } catch (error) {
            return null;
        }
        return null;
    }
    async initialize() {
        try {
            
            // Skip Ollama check - use fallback responses only
            this.isOllamaAvailable = false;
            
            return true;
        } catch (error) {
            console.error('[RAG LLM] Failed to initialize LLM service:', error);
            return false;
        }
    }


    async generateCompletion(prompt, options = {}) {
        if (!prompt || typeof prompt !== 'string') {
            throw new Error('prompt must be a non-empty string');
        }
        if (prompt.length > 100000) {
            prompt = prompt.substring(0, 100000);
        }
        if (!options || typeof options !== 'object' || Array.isArray(options)) {
            options = {};
        }
        
        try {
            // Always use fallback response (no external LLM)
            return await this.generateFallbackResponse(prompt, options);
        } catch (error) {
            console.error('[RAG LLM] Error generating completion:', error);
            return await this.generateFallbackResponse(prompt, options);
        }
    }


    async generateFallbackResponse(prompt, options = {}) {
        if (!prompt || typeof prompt !== 'string') {
            prompt = '';
        }
        if (prompt.length > 100000) {
            prompt = prompt.substring(0, 100000);
        }
        if (!options || typeof options !== 'object' || Array.isArray(options)) {
            options = {};
        }
        
        try {
            // Extract key topics from prompt
            const topics = this.extractTopics(prompt);
            const context = this.detectContext(prompt);
            const persona = options.persona || this.lastPersona || this.determinePersona(prompt, []);
            
            // Generate contextual response
            let response = this.generateContextualResponse(topics, context, prompt, persona);
            
            
            return {
                completion: this.refineGeneratedText(response, persona),
                tokensUsed: 0,
                model: 'fallback-heuristic',
                source: 'fallback'
            };
        } catch (error) {
            console.error('[RAG LLM] Fallback generation failed:', error);
            return {
                completion: "I apologize, but I'm having trouble generating a response right now. Please try again.",
                tokensUsed: 0,
                model: 'error-fallback',
                source: 'error'
            };
        }
    }

    // REMOVED: composeRagAnswer - No longer used. We now use trained model exclusively.

    groupContextByDocument(ragContext) {
        if (!ragContext || !Array.isArray(ragContext)) {
            return [];
        }
        if (ragContext.length > 1000) { // Prevent DoS
            ragContext = ragContext.slice(0, 1000);
        }
        
        const docMap = new Map();

        ragContext.forEach(item => {
            if (!item || typeof item !== 'object' || Array.isArray(item)) {
                return;
            }
            const docId = item.document_id || item.documentId || item.id || `doc_${docMap.size + 1}`;
            const metadata = item.metadata || {};
            const snippet = this.normalizeSnippet(item.content || '');

            if (!snippet) {
                return;
            }

            const existing = docMap.get(docId) || {
                id: docId,
                title: metadata.title || metadata.document_title || metadata.original_filename || '',
                category: metadata.learning_category || metadata.document_type || 'general',
                similarity: item.similarity || 0,
                snippets: [],
                metadata: {}
            };

            existing.similarity = Math.max(existing.similarity, item.similarity || 0);
            if (!existing.snippets.includes(snippet)) {
                existing.snippets.push(snippet);
            }
            existing.metadata = {
                ...existing.metadata,
                ...metadata
            };

            docMap.set(docId, existing);
        });

        return Array.from(docMap.values())
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 4);
    }

    normalizeSnippet(snippet) {
        if (!snippet || typeof snippet !== 'string') {
            return '';
        }
        if (snippet.length > 100000) { // Prevent DoS
            snippet = snippet.substring(0, 100000);
        }

        let cleaned = snippet
            .replace(/[\r\n]+/g, ' ')
            .replace(/\s+/g, ' ')
            .replace(/^\d+[\.\)]\s*/, '')
            .replace(/^[\-\*]\s*/, '')
            .replace(/^according to (our |the )?knowledge base[:,-]?\s*/i, '')
            .replace(/^according to (the |our )?knowledge base[:,-]?\s*/i, '')
            .replace(/^based on (our |the )?knowledge base[:,-]?\s*/i, '')
            .replace(/^according to (the |our )?document[:,-]?\s*/i, '')
            .replace(/^from (the |our )?document[:,-]?\s*/i, '')
            .replace(/^(the art of mastering|the psychology of|a study on)[:,\s]*/i, '')
            .replace(/\b(by|author|published|document|pdf|study|paper)\b[:\s]*[A-Z][^.]*\./gi, '')
            .replace(/\b©\s*\d{4}[^.]*\./gi, '')
            .replace(/[""]/g, '')
            .trim();

        cleaned = this.stripDocumentHeaders(cleaned);
        return cleaned;
    }

    stripDocumentHeaders(text) {
        if (!text || typeof text !== 'string') {
            return '';
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }

        let cleaned = text.trim();
        const patterns = [
            /^(case study|case overview|case summary|client story|client spotlight|project summary|project overview|executive summary|sales play|sales script|sales recap|talk track|persona spotlight|introduction|background|overview)[:\-–]\s*/i,
            /^\(?\d{4}\)?\s*[\-–]\s*/,
            /^[A-Z][A-Z\s]{3,30}[:\-–]\s*/,
            /^page\s*\d+[:\-–]?\s*/i,
            /^slide\s*\d+[:\-–]?\s*/i
        ];

        let replaced = true;
        let guard = 0;
        while (replaced && guard < 6) {
            replaced = false;
            guard += 1;
            for (const pattern of patterns) {
                if (pattern.test(cleaned)) {
                    cleaned = cleaned.replace(pattern, '').trim();
                    replaced = true;
                }
            }
        }

        cleaned = cleaned.replace(/^(["'«»])(.*)\1$/, '$2').trim();
        cleaned = cleaned.replace(/^[\-•]+\s*/, '').trim();
        return cleaned;
    }

    createSummaryFromSnippets(snippets, maxSentences = 2) {
        if (!snippets || !Array.isArray(snippets)) {
            return '';
        }
        if (snippets.length > 1000) { // Prevent DoS
            snippets = snippets.slice(0, 1000);
        }
        if (!Number.isInteger(maxSentences) || maxSentences < 1 || maxSentences > 100) {
            maxSentences = 2;
        }
        
        const combined = snippets.join(' ').replace(/\s+/g, ' ').trim();
        if (!combined) return '';

        const normalized = this.stripDocumentHeaders(combined);

        // Remove all document references and metadata
        let cleaned = normalized
            .replace(/according to (our |the )?knowledge base[:,-]?\s*/gi, '')
            .replace(/based on (our |the )?knowledge base[:,-]?\s*/gi, '')
            .replace(/according to (the |our )?document[:,-]?\s*/gi, '')
            .replace(/from (the |our )?document[:,-]?\s*/gi, '')
            .replace(/^(the art of mastering|the psychology of|a study on)[:,\s]*/gi, '')
            .replace(/\b(by|author|published|document|pdf|study|paper)\b[:\s]*[A-Z][^.]*\./gi, '')
            .replace(/\b©\s*\d{4}[^.]*\./gi, '');

        const sentences = cleaned
            .split(/(?<=[.!?])\s+/)
            .map(sentence => this.stripDocumentHeaders(sentence.trim()))
            .filter(sentence => sentence.length > 20 && !sentence.match(/^(the art of|the psychology|a study)/i));

        if (!sentences.length) {
            return this.stripDocumentHeaders(cleaned);
        }

        const selected = sentences.slice(0, maxSentences).join(' ');
        return this.ensureSentenceCase(selected);
    }

    ensureSentenceCase(text) {
        if (!text || typeof text !== 'string') {
            return '';
        }
        if (text.length > 10000) { // Prevent DoS
            text = text.substring(0, 10000);
        }
        
        const trimmed = text.trim();
        if (!trimmed) return '';
        const capitalized = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
        return capitalized.endsWith('.') ? capitalized : `${capitalized}.`;
    }

    getCategoryLabel(category) {
        if (!category || typeof category !== 'string') {
            return 'internal resources';
        }
        if (category.length > 100) { // Prevent DoS
            category = category.substring(0, 100);
        }
        
        const catalog = {
            case_study: 'case study playbooks',
            sales: 'sales enablement scripts',
            sales_training: 'sales language guidance',
            pricing: 'pricing frameworks',
            technical: 'technical implementation notes',
            testimonial: 'testimonial library',
            knowledge: 'knowledge base',
            learning: 'training material',
            general: 'reference material'
        };
        return catalog[category] || 'internal resources';
    }

    buildClosingLine(docs, userMessage) {
        if (!docs || !Array.isArray(docs)) {
            docs = [];
        }
        if (docs.length > 100) { // Prevent DoS
            docs = docs.slice(0, 100);
        }
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        
        const hasSales = docs.some(doc => (doc.category || '').includes('sales'));
        const hasTechnical = docs.some(doc => (doc.category || '').includes('technical'));

        if (hasSales) {
            return 'If you want, I can translate this guidance into outreach language or role-play a client conversation with you.';
        }

        if (hasTechnical) {
            return 'Let me know if you need implementation checklists, architecture guidance, or client-ready explanations based on this material.';
        }

        return 'I can adapt these insights further—just point me to the scenario or audience you want to focus on.';
    }

    determinePersona(userMessage, docs) {
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        if (!docs || !Array.isArray(docs)) {
            docs = [];
        }
        if (docs.length > 100) { // Prevent DoS
            docs = docs.slice(0, 100);
        }
        
        const text = userMessage.toLowerCase();
        const categories = docs.map(doc => String(doc.category || 'general').toLowerCase());
        const hasSalesDoc = categories.some(cat => cat.includes('sales'));
        const hasCaseDoc = categories.some(cat => cat.includes('case'));
        const hasTechnicalDoc = categories.some(cat => cat.includes('technical'));

        const persona = {
            mode: 'advisor',
            tone: 'neutral',
            energy: 'steady',
            callToAction: 'Let me know where you want to take this next.',
            prefersBullets: false
        };

        if (text.includes('pitch') || text.includes('close') || text.includes('sell') || hasSalesDoc) {
            persona.mode = 'sales';
            persona.tone = 'confident';
            persona.energy = 'upbeat';
            persona.callToAction = 'If you want, I can draft the outreach copy or prep objection-handling lines.';
        } else if (text.includes('integration') || text.includes('api') || text.includes('architecture') || hasTechnicalDoc) {
            persona.mode = 'technical';
            persona.tone = 'precise';
            persona.energy = 'structured';
            persona.callToAction = 'Happy to turn this into checklists, diagrams, or sprint-ready tasks.';
            persona.prefersBullets = true;
        } else if (hasCaseDoc || text.includes('results') || text.includes('proof')) {
            persona.mode = 'credibility';
            persona.tone = 'assured';
            persona.energy = 'steady';
            persona.callToAction = 'Let me know if you need more proof points or a client-ready narrative.';
        }

        return persona;
    }

    buildPersonaIntro(userMessage, persona, docs) {
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        if (!docs || !Array.isArray(docs)) {
            docs = [];
        }
        if (docs.length > 100) { // Prevent DoS
            docs = docs.slice(0, 100);
        }
        
        // Don't reference documents directly - use learned patterns instead
        const topic = this.detectContext(userMessage)?.replace(/_/g, ' ') || 'this';

        if (persona.mode === 'sales') {
            return `Here's how I approach this:`;
        }

        if (persona.mode === 'technical') {
            return `Here's what I know about ${topic}:`;
        }

        if (persona.mode === 'credibility') {
            return `Based on what I've learned, here's what works:`;
        }

        return `Let me help you with that:`;
    }

    buildConversationBridge(userMessage, persona) {
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        
        const trimmed = userMessage.trim();
        if (!trimmed) {
            return persona.mode === 'sales'
                ? 'Let me frame this the way I’d set up a buyer conversation.'
                : 'Let me walk through what stands out.';
        }

        const lower = trimmed.toLowerCase();
        if (lower.startsWith('how')) {
            return 'Let’s walk through how this plays out step by step.';
        }
        if (lower.startsWith('what')) {
            return 'Here’s what matters most based on what we’ve learned.';
        }
        if (lower.startsWith('why')) {
            return 'Here’s why this approach keeps working for us.';
        }
        if (lower.includes('can you') || lower.includes('could you') || lower.includes('would you')) {
            return 'Absolutely—here’s how I’m thinking about it in real terms.';
        }
        if (lower.includes('tell me')) {
            return 'Sure, here’s the throughline as I see it.';
        }

        return persona.mode === 'sales'
            ? 'I’ll keep it conversational so you can use it live.'
            : 'Let me share the pieces that matter most here.';
    }

    buildPersonaParagraph(doc, persona) {
        if (!doc || typeof doc !== 'object' || Array.isArray(doc)) {
            return '';
        }
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        
        const summary = this.rewriteSummary(doc, persona);
        if (!summary) return '';

        const signals = doc.metadata?.signals || {};
        const comparative = doc.metadata?.comparative_snapshot || {};
        const toneHint = doc.metadata?.tone || 'neutral';
        const stage = doc.metadata?.sales_stage || doc.metadata?.stage || doc.metadata?.funnel_stage || null;
        const audience = doc.metadata?.target_persona || doc.metadata?.audience || null;

        // Use learned patterns, not document references
        let sentence = summary;

        if (signals.containsOutcomeLanguage && comparative.improvement_word_percent !== undefined) {
            const swing = Math.abs(comparative.improvement_word_percent || 0).toFixed(1);
            if (swing > 0) {
                sentence += ` Compared with the previous language set, this version shifts emphasis by about ${swing}% in terms of detail and proof.`;
            }
        }

        if (signals.containsCallToAction && persona.mode === 'sales') {
            sentence += ' It naturally flows into a confident ask, so we can use it as the backbone of an outreach sequence.';
        }

        if (toneHint && persona.mode !== 'technical') {
            sentence += ` I keep the tone ${toneHint.toLowerCase()} so it feels natural in conversation.`;
        }

        const percentExample = Array.isArray(signals.percent_values) ? signals.percent_values.find(Boolean) : null;
        if (percentExample) {
            sentence += ` Teams in similar situations are seeing around ${percentExample.replace(/[^0-9.%+-]/g, '')} movement once this goes live.`;
        }

        const currencyExample = Array.isArray(signals.currency_values) ? signals.currency_values.find(Boolean) : null;
        if (currencyExample && persona.mode === 'sales') {
            sentence += ` That translates to ${currencyExample.replace(/[^0-9.$€£,+-]/g, '')} in captured value we can reference when the buyer asks.`;
        }

        if (stage) {
            sentence += ` This sits in our ${stage.toLowerCase()} stage guidance, so the sequence stays on track.`;
        }

        if (audience && persona.mode !== 'technical') {
            sentence += ` It’s written for a ${audience.toLowerCase()} audience, which keeps the positioning sharp.`;
        }

        return sentence.replace(/\s+/g, ' ').trim();
    }

    rewriteSummary(doc, persona) {
        if (!doc || typeof doc !== 'object' || Array.isArray(doc)) {
            return '';
        }
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        
        // Extract key concepts from snippets, but rewrite in Epsilon AI's voice
        const base = this.createSummaryFromSnippets(doc.snippets || [], persona.prefersBullets ? 1 : 2);
        if (!base) return '';

        let summary = base
            .replace(/["""]/g, '')
            .replace(/\s+/g, ' ')
            .trim();

        summary = this.stripDocumentHeaders(summary);

        // Remove all document references and rewrite in conversational tone
        summary = summary
            .replace(/according to (our |the )?knowledge base[:,]?\s*/gi, '')
            .replace(/based on (our |the )?knowledge base[:,]?\s*/gi, '')
            .replace(/according to (the |our )?document[:,]?\s*/gi, '')
            .replace(/from (the |our )?document[:,]?\s*/gi, '')
            .replace(/^(the art of mastering|the psychology of|a study on)[:,\s]*/gi, '')
            .replace(/\b(by|author|published|document|pdf|study|paper)\b[:\s]*[A-Z][^.]*\./gi, '')
            .replace(/\b©\s*\d{4}[^.]*\./gi, '');

        // Rewrite in Epsilon AI's voice based on persona
        if (persona.mode === 'sales') {
            summary = summary
                .replace(/\bwe\b/gi, 'I')
                .replace(/\bour\b/gi, 'my')
                .replace(/\bclients?\b/gi, 'customers')
                .replace(/\bthe\s+art\s+of\b/gi, 'the approach to')
                .replace(/\btechniques?\b/gi, 'approaches');
        } else if (persona.mode === 'technical') {
            summary = summary
                .replace(/\boffers\b/gi, 'provides')
                .replace(/\bhelps\b/gi, 'supports')
                .replace(/\bwe\b/gi, 'the system');
        } else if (persona.mode === 'credibility') {
            summary = summary.replace(/\bwe\b/gi, 'I');
        }

        // Make it sound like Epsilon AI learned this, not quoting
        if (!summary.toLowerCase().startsWith('i ') && !summary.toLowerCase().startsWith('we ')) {
            summary = summary.charAt(0).toLowerCase() + summary.slice(1);
        }
 
        return this.ensureSentenceCase(summary);
    }

    buildPersonaClosing(persona, docs, userMessage) {
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        if (!docs || !Array.isArray(docs)) {
            docs = [];
        }
        if (docs.length > 100) { // Prevent DoS
            docs = docs.slice(0, 100);
        }
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        
        if (persona.mode === 'sales') {
            return 'Want me to stitch this into outreach wording or a follow-up script?';
        }

        if (persona.mode === 'technical') {
            return 'I can turn this into a step-by-step rollout or a handoff checklist if that helps.';
        }

        if (persona.mode === 'credibility') {
            return 'If you need more proof points, I can surface adjacent case stories or tailor the tone for a specific buyer.';
        }

        return this.buildClosingLine(docs, userMessage);
    }

    sanitizeResponse(text) {
        if (!text || typeof text !== 'string') {
            return '';
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        
        let cleaned = text
            // Remove document title references
            .replace(/according to (our |the )?knowledge base[:,]?\s*/gi, '')
            .replace(/based on (our |the )?knowledge base[:,]?\s*/gi, '')
            .replace(/according to (the |our )?document[:,]?\s*/gi, '')
            .replace(/from (the |our )?document[:,]?\s*/gi, '')
            .replace(/^(the art of mastering|the psychology of|a study on)[:,\s]*/gi, '')
            .replace(/\b(by|author|published|document|pdf|study|paper)\b[:\s]*[A-Z][^.]*\./gi, '')
            .replace(/\b©\s*\d{4}[^.]*\./gi, '')
            // Remove document titles that appear as standalone phrases
            .replace(/\b(The Art of Mastering Sales Management|The Psychology of Selling|A Study on Persuasive Language)[:,\s]*/gi, '')
            // Fix grammar errors
            .replace(/would love to I/gi, 'can I')
            .replace(/would love to we/gi, 'can we')
            .replace(/How would love to I/gi, 'How can I')
            .replace(/\s+/g, ' ')
            .trim();
            
        return cleaned;
    }

    refineGeneratedText(text = '', persona = null) {
        if (!text || typeof text !== 'string') {
            return '';
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        if (persona !== null && (typeof persona !== 'object' || Array.isArray(persona))) {
            persona = null;
        }

        let cleaned = text
            .replace(/\r/g, ' ')
            .replace(/\u00A0/g, ' ')
            .replace(/\s+/g, ' ')
            .replace(/\s+([,;:])/g, '$1')
            .replace(/\s+\./g, '.')
            .trim();

        if (!cleaned) {
            return '';
        }

        const sentences = cleaned
            .split(/(?<=[.!?])\s+/)
            .map(sentence => sentence.trim())
            .filter(Boolean)
            .map(sentence => this.ensureSentenceCase(sentence.replace(/\s+,/g, ',').replace(/\s+\./g, '.')));

        const normalized = sentences.join(' ').trim();
        let finalText = normalized.replace(/\s+(!|\?)/g, '$1');

        if (persona?.mode === 'technical') {
            finalText = finalText
                .replace(/\bwe\b/gi, 'the team')
                .replace(/\bclient\b/gi, 'implementation team');
        } else if (persona?.mode === 'sales') {
            finalText = finalText.replace(/\bwe\b/gi, 'we');
        }

        return finalText;
    }

    estimateTokenUsage(text) {
        if (!text || typeof text !== 'string') {
            return 0;
        }
        if (text.length > 100000) { // Prevent DoS
            text = text.substring(0, 100000);
        }
        
        const words = text.trim().split(/\s+/).length;
        return Math.ceil(words * 1.3);
    }

    extractTopics(text) {
        if (!text || typeof text !== 'string') {
            return [];
        }
        if (text.length > 10000) { // Prevent DoS
            text = text.substring(0, 10000);
        }
        
        const businessKeywords = ['automation', 'ai', 'business', 'strategy', 'operations', 'efficiency', 'productivity'];
        const techKeywords = ['technology', 'software', 'development', 'system', 'integration', 'api', 'database'];
        const generalKeywords = ['help', 'assist', 'support', 'question', 'problem', 'solution'];
        
        const lowerText = text.toLowerCase();
        const topics = [];
        
        [...businessKeywords, ...techKeywords, ...generalKeywords].forEach(keyword => {
            if (lowerText.includes(keyword)) {
                topics.push(keyword);
            }
        });
        
        return topics;
    }

    detectContext(text) {
        if (!text || typeof text !== 'string') {
            return 'general';
        }
        if (text.length > 10000) { // Prevent DoS
            text = text.substring(0, 10000);
        }
        
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('automation') || lowerText.includes('process')) {
            return 'automation';
        } else if (lowerText.includes('ai') || lowerText.includes('artificial intelligence')) {
            return 'ai';
        } else if (lowerText.includes('business') || lowerText.includes('strategy')) {
            return 'business';
        } else if (lowerText.includes('technical') || lowerText.includes('code')) {
            return 'technical';
        } else {
            return 'general';
        }
    }

    getPersonaVoice(persona = {}, context = 'general') {
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        if (!context || typeof context !== 'string') {
            context = 'general';
        }
        if (context.length > 100) { // Prevent DoS
            context = context.substring(0, 100);
        }
        
        const voices = {
            sales: {
                opener: 'From a revenue perspective,',
                closer: 'Want me to turn that into language you can drop into your next buyer touchpoint?'
            },
            technical: {
                opener: 'Operationally speaking,',
                closer: 'I can sketch out the implementation steps or prep a quick checklist.'
            },
            credibility: {
                opener: 'Here’s what we’ve proven in the field:',
                closer: 'Let me know if you want that framed as a short case story or proof point.'
            },
            advisor: {
                opener: 'Here’s how I’m looking at it:',
                closer: 'Tell me where you want to take this next and I’ll keep building.'
            }
        };

        const normalizedMode = persona?.mode || 'advisor';
        return voices[normalizedMode] || voices.advisor;
    }

    generateContextualResponse(topics, context, originalPrompt, persona = {}) {
        if (!topics || !Array.isArray(topics)) {
            topics = [];
        }
        if (topics.length > 100) { // Prevent DoS
            topics = topics.slice(0, 100);
        }
        if (!context || typeof context !== 'string') {
            context = 'general';
        }
        if (context.length > 100) { // Prevent DoS
            context = context.substring(0, 100);
        }
        if (!originalPrompt || typeof originalPrompt !== 'string') {
            originalPrompt = '';
        }
        if (originalPrompt.length > 100000) { // Prevent DoS
            originalPrompt = originalPrompt.substring(0, 100000);
        }
        if (!persona || typeof persona !== 'object' || Array.isArray(persona)) {
            persona = {};
        }
        
        const voice = this.getPersonaVoice(persona, context);

        const baseInsights = {
            automation: 'we can hand off repetitive tasks so every handoff is clean, signals escalate automatically, and your team stays focused on revenue work.',
            ai: 'we use AI to surface the right context at the right moment, tighten feedback loops, and keep outcomes measurable.',
            business: 'we focus on removing friction across your go-to-market systems so every team moves in sync toward the same outcome.',
            technical: 'we line up the integration points, keep the data contracts tidy, and make sure adoption doesn’t stall after launch.',
            general: 'I’m here to simplify the work, surface the priorities, and keep momentum on the things that matter.'
        };

        let body = baseInsights[context] || baseInsights.general;

        if (topics.includes('pricing')) {
            body += ' Pricing questions usually need a clean ROI story, so we can anchor the conversation in the lift we’ve already proven.';
        } else if (topics.includes('integration')) {
            body += ' We’ll lock in the integration checkpoints early so engineering never gets surprised downstream.';
        } else if (topics.includes('automation')) {
            body += ' Automation simply means you get reliable outcomes without babysitting every step.';
        }

        const response = `${voice.opener} ${body} ${voice.closer}`;
        return response.replace(/\s+/g, ' ').trim();
    }

    buildGeneralFallback(userMessage) {
        if (!userMessage || typeof userMessage !== 'string') {
            userMessage = '';
        }
        if (userMessage.length > 10000) {
            userMessage = userMessage.substring(0, 10000);
        }
        
        const context = this.detectContext(userMessage);
        const topics = this.extractTopics(userMessage);
        const persona = this.determinePersona(userMessage, []);
        this.lastPersona = persona;
        const response = this.generateContextualResponse(topics, context, userMessage, persona);
        return this.refineGeneratedText(response, persona);
    }

    initializeFallbackResponses() {
        return {
            greeting: "Hello! I'm Epsilon AI, your AI operations assistant. I specialize in business automation, AI strategy, and operational efficiency. How can I help you today?",
            
            automation: "I can help you with business automation strategies. Let me know what processes you'd like to automate or optimize.",
            
            ai: "As an AI operations assistant, I can help you understand AI implementation and choose the right AI tools for your business.",
            
            business: "I specialize in business operations and strategy. I can help you optimize processes and improve efficiency.",
            
            technical: "I can assist with technical implementation and system integration. What technical challenge are you facing?",
            
            error: "I apologize, but I'm having trouble generating a response right now. Please try again or rephrase your question."
        };
    }

    async generateRAGResponse(userMessage, ragContext = []) {
        if (!userMessage || typeof userMessage !== 'string') {
            throw new Error('userMessage must be a non-empty string');
        }
        if (userMessage.length > 100000) { // Prevent DoS (100KB max)
            userMessage = userMessage.substring(0, 100000);
        }
        if (!ragContext || !Array.isArray(ragContext)) {
            ragContext = [];
        }
        if (ragContext.length > 1000) { // Prevent DoS
            ragContext = ragContext.slice(0, 1000);
        }
        
        try {
            
            if (!this.isOllamaAvailable) {
                let personaHint = null;

                // ALWAYS try trained model first - it has learned from documents
                // IMPORTANT: Epsilon AI uses learned patterns, NOT document search
                // Sales documents teach her HOW to talk (tone/style), not WHAT to say
                // Knowledge documents are digested during training, not retrieved during conversation
                try {
                    if (epsilonLanguageEngine && typeof epsilonLanguageEngine.isModelReady === 'function' && epsilonLanguageEngine.isModelReady()) {
                        if (typeof epsilonLanguageEngine.buildPersonaHint === 'function') {
                            // Build persona from user message only - no document context needed
                            // Model has already learned tone/style from sales training documents
                            personaHint = epsilonLanguageEngine.buildPersonaHint(userMessage, []);
                        }
                        if (typeof epsilonLanguageEngine.generate === 'function') {
                            // NEVER pass ragContext - model has learned and digested all information
                            // Epsilon AI uses learned patterns naturally, like transformer-based language models
                            const generation = await epsilonLanguageEngine.generate({
                                userMessage,
                                ragContext: [], // Always empty - model uses learned patterns
                                persona: personaHint
                            });

                            if (generation && generation.text) {
                                // Sanitize response to remove any document references
                                const sanitized = this.sanitizeResponse(generation.text);
                                return {
                                    completion: sanitized,
                                    tokensUsed: generation.meta?.tokens_generated || this.estimateTokenUsage(sanitized),
                                    model: 'epsilon-mini-llm',
                                    source: 'epsilon-language-model'
                                };
                            }
                        }
                    }
                } catch (err) {
                    console.warn('[RAG LLM] Epsilon AI language engine generation failed, falling back', err.message);
                }

                if (!personaHint) {
                    // Build persona from user message only - no document context needed
                    // Model has already learned tone/style from sales training documents
                    const syntheticPersona = this.determinePersona(userMessage, []);
                    personaHint = { ...syntheticPersona, source: 'synthetic' };
                }

                const generalIntent = this.handleGeneralIntent(userMessage, personaHint);
                if (generalIntent) {
                    return generalIntent;
                }

                // Use fallback that doesn't quote documents
                // Epsilon AI uses learned patterns, not document search
                const fallback = await this.generateFallbackResponse(userMessage, { persona: personaHint });
                if (fallback) {
                    return fallback;
                }
                
                // Last resort: minimal response without document references
                return {
                    completion: this.refineGeneratedText(`I can help you with that. What specific aspect would you like to explore?`, personaHint),
                    tokensUsed: 0,
                    model: 'minimal-fallback',
                    source: 'minimal'
                };
            }
            
            // Build context from retrieved documents
            const contextText = ragContext.map((doc, i) => `[${i + 1}] ${doc.content}`).join('\n\n');
            
            // Create RAG prompt
            const prompt = `System: You are Epsilon AI, an advanced AI operations assistant. Use ONLY the facts provided in the CONTEXT below to answer the user's question. If the context doesn't contain relevant information, say so clearly.

CONTEXT:
${contextText}

User Question: ${userMessage}

Instructions:
- Answer based ONLY on the provided context
- Be accurate and helpful
- If context is insufficient, explain what information you need
- Maintain Epsilon AI's professional and enthusiastic personality
- Keep responses concise but informative

Answer:`;

            // Generate response
            const result = await this.generateCompletion(prompt, {
                maxTokens: 512,
                temperature: 0.2
            });
            
            return result;
        } catch (error) {
            console.error('[RAG LLM] Error generating RAG response:', error);
            return {
                completion: this.fallbackResponses.error,
                tokensUsed: 0,
                model: 'error-fallback',
                source: 'error'
            };
        }
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.RAGLLMService = RAGLLMService;
} else if (typeof module !== 'undefined') {
    module.exports = RAGLLMService;
}
