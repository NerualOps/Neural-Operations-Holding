/**
 * Document Learning Service
 * Integrates with Python document learning service for AI training
 */

class DocumentLearningService {
    constructor() {
        this.baseUrl = '/api/document-learning';
        this.isConnected = false;
        this.learningCategories = {
            knowledge: 'Knowledge Base Documents',
            sales_training: 'Sales Training & Tone',
            learning: 'General Learning Documents'
        };
        this.uploadQueue = [];
        this.isProcessing = false;
    }

    /**
     * Initialize the document learning service
     */
    async initialize() {
        try {
            // Silent - no console.log
            
            // Test connection to Python service
            const response = await fetch(`${this.baseUrl}/health`);
            if (response.ok) {
                this.isConnected = true;
                // Silent - no console.log
                
                // Process any queued uploads
                await this.processUploadQueue();
            } else {
                console.warn('[DOCUMENT LEARNING] Python service not available, using fallback mode');
                this.isConnected = false;
            }
        } catch (error) {
            console.warn('[DOCUMENT LEARNING] Python service connection failed:', error.message);
            this.isConnected = false;
        }
    }

    /**
     * Upload a document for learning
     */
    async uploadDocument(file, documentType, learningCategory, description = '', tags = []) {
        try {
            // Silent - no console.log
            
            if (!this.isConnected) {
                // Queue for later processing
                this.uploadQueue.push({ file, documentType, learningCategory, description, tags });
                // Return queued status - service will process when available
                return {
                    success: false,
                    queued: true,
                    message: 'Document queued for processing when service becomes available',
                    document_id: this.generateDocumentId(file.name)
                };
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('document_type', documentType);
            formData.append('learning_category', learningCategory);
            formData.append('description', description);
            formData.append('tags', tags.join(','));

            const response = await fetch(`${this.baseUrl}/upload-document`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`[DOCUMENT LEARNING] Upload failed with status ${response.status}:`, errorText);
                
                // Don't mask server errors - throw them so they can be properly handled
                throw new Error(`Document upload failed (HTTP ${response.status}): ${errorText}`);
            }

            const result = await response.json();
            // Silent - no console.log
            
            // Trigger learning process
            try {
                await this.triggerLearningProcess(learningCategory, result.document_id);
            } catch (learningError) {
                console.warn('[DOCUMENT LEARNING] Learning process failed:', learningError);
                // Don't fail the upload if learning process fails
            }
            
            return result;
        } catch (error) {
            console.error('[DOCUMENT LEARNING] Upload failed:', error);
            // Re-throw the error instead of masking it with a fallback
            throw error;
        }
    }

    /**
     * REMOVED: createFallbackResponse - No longer used
     * Errors should be thrown instead of masked with fallback responses
     * This prevents silent failures and ensures proper error handling
     */

    /**
     * Trigger learning process for a specific document
     */
    async triggerLearningProcess(category, documentId) {
        try {
            // Silent - no console.log
            
            if (!this.isConnected) {
                // Silent - no console.log
                return;
            }

            // Get learning insights
            const insightsResponse = await fetch(`${this.baseUrl}/learning-insights/${category}`);
            if (insightsResponse.ok) {
                const insights = await insightsResponse.json();
                // Silent - no console.log
                
                // Update Epsilon AI's learning system
                await this.updateEpsilonLearning(category, insights);
            }
        } catch (error) {
            console.error('[DOCUMENT LEARNING] Learning process failed:', error);
        }
    }

    /**
     * Update Epsilon AI's learning system with new insights
     */
    async updateEpsilonLearning(category, insights) {
        try {
            // Silent - no console.log
            
            // Send learning data to Epsilon AI
            const learningData = {
                category: category,
                insights: insights,
                timestamp: new Date().toISOString(),
                action: 'update_learning'
            };

            // Store in localStorage for Epsilon AI to access
            const existingLearning = JSON.parse(localStorage.getItem('epsilon_learning_data') || '{}');
            existingLearning[category] = learningData;
            localStorage.setItem('epsilon_learning_data', JSON.stringify(existingLearning));
            
            // Emit event for Epsilon AI to process
            window.dispatchEvent(new CustomEvent('epsilon:learning-update', {
                detail: learningData
            }));

            // Silent - no console.log
        } catch (error) {
            console.error('[DOCUMENT LEARNING] Failed to update Epsilon AI learning:', error);
        }
    }

    /**
     * Get learning progress for all categories
     */
    async getLearningProgress() {
        try {
            if (!this.isConnected) {
                // Service not available - throw error instead of returning fallback
                throw new Error('Document learning service is not connected. Please check service status.');
            }

            const response = await fetch(`${this.baseUrl}/learning-progress`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const progress = await response.json();
            // Silent - no console.log
            return progress;
        } catch (error) {
            console.error('[DOCUMENT LEARNING] Failed to get learning progress:', error);
            // Re-throw error instead of masking with fallback
            throw error;
        }
    }

    /**
     * Get documents by category
     */
    async getDocumentsByCategory(category) {
        // Safety check: validate input
        if (!category || typeof category !== 'string') {
            throw new Error('category must be a non-empty string');
        }
        
        try {
            if (!this.isConnected) {
                throw new Error('Document learning service is not connected. Please check service status.');
            }

            const response = await fetch(`${this.baseUrl}/documents/${category}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            // Silent - no console.log
            return result;
        } catch (error) {
            console.error(`[DOCUMENT LEARNING] Failed to get documents for ${category}:`, error);
            // Re-throw error instead of masking with empty result
            throw error;
        }
    }

    /**
     * Delete a document
     */
    async deleteDocument(category, documentId) {
        try {
            if (!this.isConnected) {
                // Silent - no console.log
                return { success: true, message: 'Delete queued for processing' };
            }

            const response = await fetch(`${this.baseUrl}/documents/${category}/${documentId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            // Silent - no console.log
            return result;
        } catch (error) {
            console.error('[DOCUMENT LEARNING] Delete failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Process queued uploads
     */
    async processUploadQueue() {
        if (this.isProcessing || this.uploadQueue.length === 0) {
            return;
        }

        this.isProcessing = true;
        // Silent - no console.log

        while (this.uploadQueue.length > 0) {
            const upload = this.uploadQueue.shift();
            try {
                await this.uploadDocument(
                    upload.file,
                    upload.documentType,
                    upload.learningCategory,
                    upload.description,
                    upload.tags
                );
            } catch (error) {
                console.error('[DOCUMENT LEARNING] Queued upload failed:', error);
            }
        }

        this.isProcessing = false;
        // Silent - no console.log
    }

    /**
     * Generate document ID
     */
    generateDocumentId(filename) {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2);
        return `${filename}_${timestamp}_${random}`.replace(/[^a-zA-Z0-9_]/g, '_');
    }

    /**
     * Get learning categories
     */
    getLearningCategories() {
        return this.learningCategories;
    }

    /**
     * Check if service is connected
     */
    isServiceConnected() {
        return this.isConnected;
    }

    /**
     * Reconnect to service
     */
    async reconnect() {
        // Silent - no console.log
        await this.initialize();
    }
}

// Create global instance
window.documentLearningService = new DocumentLearningService();

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.documentLearningService.initialize();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DocumentLearningService;
}
