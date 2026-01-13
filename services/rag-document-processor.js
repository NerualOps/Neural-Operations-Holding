/**
 * RAG Document Processor
 * Purpose: Process and manage documents for RAG system
 * Integration: Works with Supabase proxy and embedding service
 * Features: Document chunking, metadata extraction, batch processing
 */

//© 2025 Neural Ops – a division of Neural Operation's & Holding's LLC. All rights reserved.

class RAGDocumentProcessor {
    constructor(supabaseProxy, embeddingService) {
        this.supabaseProxy = supabaseProxy;
        this.embeddingService = embeddingService;
        this.processedDocuments = new Set();
        this.batchSize = 5; // Process 5 documents at a time
        this.maxRetries = 3;
    }

    async processDocument(documentId, content, metadata = {}) {
        if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
            return { success: false, error: 'documentId must be a non-empty string or number' };
        }
        if (!content || typeof content !== 'string') {
            return { success: false, error: 'content must be a non-empty string' };
        }
        if (content.length > 10000000) { // Prevent DoS (10MB max)
            content = content.substring(0, 10000000);
        }
        if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
            metadata = {};
        }
        
        try {
            
            // Check if already processed
            if (this.processedDocuments.has(documentId)) {
                return { success: true, message: 'Already processed' };
            }
            
            let normalizedMetadata = {};
            try {
                if (typeof metadata === 'string') {
                    normalizedMetadata = JSON.parse(metadata);
                } else if (metadata && typeof metadata === 'object') {
                    normalizedMetadata = { ...metadata };
                }
            } catch (parseError) {
                console.warn(`[RAG PROCESSOR] Unable to parse metadata for document ${documentId}:`, parseError.message);
            }
            
            // Generate embeddings for document chunks
            const embeddings = await this.embeddingService.processDocument(documentId, content, normalizedMetadata);
            
            // Store embeddings in Supabase
            const storedEmbeddings = [];
            for (const embedding of embeddings) {
                const result = await this.storeDocumentEmbedding(embedding);
                if (result.success) {
                    storedEmbeddings.push(result.embeddingId);
                }
            }
            
            // Mark as processed
            this.processedDocuments.add(documentId);
            
            
            return {
                success: true,
                documentId,
                embeddingsStored: storedEmbeddings.length,
                embeddingIds: storedEmbeddings
            };
        } catch (error) {
            console.error(`[RAG PROCESSOR] Error processing document ${documentId}:`, error);
            return {
                success: false,
                documentId,
                error: error.message
            };
        }
    }

    async storeDocumentEmbedding(embeddingData) {
        if (!embeddingData || typeof embeddingData !== 'object' || Array.isArray(embeddingData)) {
            return { success: false, error: 'embeddingData must be a non-empty object' };
        }
        if (!embeddingData.documentId || (typeof embeddingData.documentId !== 'string' && typeof embeddingData.documentId !== 'number')) {
            return { success: false, error: 'embeddingData.documentId must be a non-empty string or number' };
        }
        
        const docIdStr = String(embeddingData.documentId).trim();
        if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
            return { success: false, error: `Invalid document_id: ${embeddingData.documentId}` };
        }
        
        if (!embeddingData.content || typeof embeddingData.content !== 'string') {
            return { success: false, error: 'embeddingData.content must be a non-empty string' };
        }
        if (embeddingData.content.length > 100000) { // Prevent DoS (100KB max)
            embeddingData.content = embeddingData.content.substring(0, 100000);
        }
        if (!embeddingData.embedding || !Array.isArray(embeddingData.embedding)) {
            return { success: false, error: 'embeddingData.embedding must be a non-empty array' };
        }
        if (embeddingData.embedding.length > 10000) { // Prevent DoS
            embeddingData.embedding = embeddingData.embedding.slice(0, 10000);
        }
        if (embeddingData.metadata && (typeof embeddingData.metadata !== 'object' || Array.isArray(embeddingData.metadata))) {
            embeddingData.metadata = {};
        }
        
        try {
            const response = await this.supabaseProxy('store-document-embedding', {
                document_id: docIdStr,
                content: embeddingData.content,
                embedding: embeddingData.embedding,
                metadata: embeddingData.metadata
            });
            
            if (response.success) {
                return {
                    success: true,
                    embeddingId: response.embedding_id
                };
            } else {
                throw new Error(response.error || 'Failed to store embedding');
            }
        } catch (error) {
            console.error('[RAG PROCESSOR] Error storing document embedding:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async processBatch(documents) {
        if (!documents || !Array.isArray(documents)) {
            return { success: false, error: 'documents must be a non-empty array' };
        }
        if (documents.length > 1000) { // Prevent DoS
            documents = documents.slice(0, 1000);
        }
        
        try {
            
            const results = [];
            const batches = this.chunkArray(documents, this.batchSize);
            
            for (const batch of batches) {
                const batchPromises = batch.map(doc => 
                    this.processDocument(
                        doc.id,
                        doc.content || '',
                        doc.metadata || doc.learning_metadata || {}
                    )
                );
                
                const batchResults = await Promise.all(batchPromises);
                results.push(...batchResults);
                
                // Small delay between batches to avoid overwhelming the system
                await this.delay(1000);
            }
            
            const successCount = results.filter(r => r.success).length;
            const failureCount = results.length - successCount;
            
            
            return {
                success: true,
                totalProcessed: results.length,
                successCount,
                failureCount,
                results
            };
        } catch (error) {
            console.error('[RAG PROCESSOR] Error processing batch:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async processKnowledgeDocuments() {
        try {
            
            // Get all knowledge documents from Supabase
            const response = await this.supabaseProxy('get-all-documents', {});
            
            const documents = response.documents || response.data || [];
            
            if (!response.success || documents.length === 0) {
                return { success: true, message: 'No documents to process' };
            }
            
            
            // Process documents
            const result = await this.processBatch(documents);
            
            return result;
        } catch (error) {
            console.error('[RAG PROCESSOR] Error processing knowledge documents:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async searchDocuments(query, topK = 6, threshold = 0.7) {
        if (!query || typeof query !== 'string') {
            return { success: false, error: 'query must be a non-empty string', results: [] };
        }
        if (query.length > 10000) { // Prevent DoS
            query = query.substring(0, 10000);
        }
        if (!Number.isInteger(topK) || topK < 1 || topK > 100) {
            topK = 6; // Default
        }
        if (typeof threshold !== 'number' || threshold < 0 || threshold > 1 || !isFinite(threshold)) {
            threshold = 0.7; // Default
        }
        
        try {
            
            // Search using Supabase vector search
            const response = await this.supabaseProxy('search-rag', {
                query: query,
                top_k: topK,
                match_threshold: threshold
            });
            
            if (response.success && response.results) {
                return {
                    success: true,
                    results: response.results,
                    query: query
                };
            } else {
                return {
                    success: true,
                    results: [],
                    query: query
                };
            }
        } catch (error) {
            console.error('[RAG PROCESSOR] Error searching documents:', error);
            return {
                success: false,
                error: error.message,
                results: []
            };
        }
    }

    async addNewDocument(documentId, content, metadata = {}) {
 (delegates to processDocument, but add explicit checks here too)
        if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
            return { success: false, error: 'documentId must be a non-empty string or number' };
        }
        if (!content || typeof content !== 'string') {
            return { success: false, error: 'content must be a non-empty string' };
        }
        if (content.length > 10000000) { // Prevent DoS (10MB max)
            content = content.substring(0, 10000000);
        }
        if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
            metadata = {};
        }
        
        try {
            
            // Process the new document
            const result = await this.processDocument(documentId, content, metadata);
            
            if (result.success) {
            } else {
                console.error(`[RAG PROCESSOR] Failed to add document ${documentId}:`, result.error);
            }
            
            return result;
        } catch (error) {
            console.error('[RAG PROCESSOR] Error adding new document:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async updateDocument(documentId, content, metadata = {}) {
 (delegates to processDocument, but add explicit checks here too)
        if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
            return { success: false, error: 'documentId must be a non-empty string or number' };
        }
        if (!content || typeof content !== 'string') {
            return { success: false, error: 'content must be a non-empty string' };
        }
        if (content.length > 10000000) { // Prevent DoS (10MB max)
            content = content.substring(0, 10000000);
        }
        if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
            metadata = {};
        }
        
        try {
            
            // Remove old embeddings
            await this.removeDocumentEmbeddings(documentId);
            
            // Process updated document
            const result = await this.processDocument(documentId, content, metadata);
            
            if (result.success) {
            } else {
                console.error(`[RAG PROCESSOR] Failed to update document ${documentId}:`, result.error);
            }
            
            return result;
        } catch (error) {
            console.error('[RAG PROCESSOR] Error updating document:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async removeDocumentEmbeddings(documentId) {
        if (!documentId || (typeof documentId !== 'string' && typeof documentId !== 'number')) {
            return;
        }
        
        try {
            // This would require a new API endpoint in supabase-proxy.js
            // For now, we'll mark it as processed to avoid reprocessing
            this.processedDocuments.delete(documentId);
        } catch (error) {
            console.error('[RAG PROCESSOR] Error removing document embeddings:', error);
        }
    }

    chunkArray(array, chunkSize) {
        if (!array || !Array.isArray(array)) {
            return [];
        }
        if (!Number.isInteger(chunkSize) || chunkSize < 1 || chunkSize > 1000) {
            chunkSize = 5; // Default
        }
        
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    getProcessedDocuments() {
        return Array.from(this.processedDocuments);
    }

    clearProcessedDocuments() {
        this.processedDocuments.clear();
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.RAGDocumentProcessor = RAGDocumentProcessor;
} else if (typeof module !== 'undefined') {
    module.exports = RAGDocumentProcessor;
}
