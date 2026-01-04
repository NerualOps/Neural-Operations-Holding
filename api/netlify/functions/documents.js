const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client
// Use SERVICE_KEY for admin operations (document management requires owner access)
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
    console.error('Missing Supabase environment variables');
    throw new Error('Missing Supabase configuration');
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Import decryption function
// Note: This assumes the encryption module is available in the Netlify function environment
// If not available, you may need to copy the encryption logic here or use a shared module
let decrypt;
try {
    // Try to require from runtime directory (adjust path as needed for Netlify)
    const encryption = require('../../../runtime/encryption');
    decrypt = encryption.decrypt;
} catch (error) {
    console.warn('Encryption module not available in Netlify function, decryption will be skipped');
    decrypt = (text) => {
        console.warn('Decryption not available');
        return text; // Return as-is if decryption unavailable
    };
}

exports.handler = async (event, context) => {
    // Set CORS headers
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Content-Type': 'application/json'
    };

    // Handle preflight requests
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers,
            body: ''
        };
    }

    try {
        // Route based on HTTP method (Netlify functions use httpMethod)
        const method = event.httpMethod;

        if (method === 'GET') {
            return await handleGetDocuments(event, headers);
        } else if (method === 'POST') {
            return await handleUploadDocument(event, headers);
        } else if (method === 'DELETE') {
            return await handleDeleteDocument(event, headers);
        }

        return {
            statusCode: 405,
            headers,
            body: JSON.stringify({ error: 'Method not allowed' })
        };

    } catch (error) {
        console.error('Documents function error:', error);
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({ 
                error: 'Internal server error',
                details: error.message 
            })
        };
    }
};

// Get all documents
async function handleGetDocuments(event, headers) {
    try {
        // Check if Supabase is properly initialized
        if (!supabase) {
            console.error('Supabase client not initialized');
            return {
                statusCode: 500,
                headers,
                body: JSON.stringify({ error: 'Database connection failed' })
            };
        }

        // Fetch all documents from knowledge_documents table - include chunked flags
        const { data: documents, error } = await supabase
            .from('knowledge_documents')
            .select('*, is_chunked, total_chunks')
            .order('created_at', { ascending: false });

        if (error) {
            console.error('Error fetching documents:', error);
            return {
                statusCode: 500,
                headers,
                body: JSON.stringify({ error: 'Failed to fetch documents' })
            };
        }

        // Process documents - handle chunked documents and decrypt content
        const processedDocuments = await Promise.all((documents || []).map(async (doc) => {
            try {
                let content = doc.content || '';
                
                if (doc.is_chunked && doc.id) {
                    try {
                        const docIdStr = String(doc.id).trim();
                        if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
                            console.warn(`[NETLIFY] Invalid document_id: ${doc.id}, skipping chunks`);
                        } else {
                            const { data: chunkRows, error: chunkError } = await supabase
                                .from('doc_chunks')
                                .select('chunk_text, chunk_index')
                                .eq('document_id', docIdStr)
                                .order('chunk_index', { ascending: true })
                                .limit(1000);
                            
                            if (!chunkError && chunkRows && chunkRows.length > 0) {
                                content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
                            } else if (chunkError) {
                                const errorStr = chunkError?.message || chunkError?.toString() || '';
                                const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                                   errorStr.includes('Cloudflare') || 
                                                   errorStr.includes('522') || 
                                                   errorStr.includes('521');
                                if (isHtmlError) {
                                    console.warn(`[WARN] [NETLIFY] Supabase connection issue while fetching chunks for document ${docIdStr}`);
                                } else {
                                    console.warn(`[NETLIFY] Failed to fetch chunks for document ${docIdStr}: ${chunkError.message || 'Unknown error'}`);
                                }
                                // Continue with preview content
                            } else if (!chunkRows || chunkRows.length === 0) {
                                console.warn(`[WARN] [NETLIFY] Document ${docIdStr} marked as chunked but no chunks found`);
                            }
                        }
                    } catch (chunkFetchError) {
                        const errorStr = chunkFetchError?.message || chunkFetchError?.toString() || '';
                        const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                                           errorStr.includes('Cloudflare') || 
                                           errorStr.includes('522') || 
                                           errorStr.includes('521');
                        if (isHtmlError) {
                            console.warn(`[WARN] [NETLIFY] Supabase connection issue while fetching chunks for document ${doc.id}`);
                        } else {
                            console.warn(`[NETLIFY] Error fetching chunks for document ${doc.id}: ${chunkFetchError.message || 'Unknown error'}`);
                        }
                        // Continue with preview content
                    }
                }
                
                // Decrypt document content if encrypted
                if (content && decrypt) {
                    const isEncrypted = typeof content === 'string' && content.includes(':') && content.split(':').length === 3;
                    
                    if (isEncrypted) {
                        const decryptedContent = decrypt(content);
                        if (decryptedContent) {
                            return {
                                ...doc,
                                content: decryptedContent
                            };
                        } else {
                            console.error(`[NETLIFY] Failed to decrypt content for document ${doc.id}`);
                            return {
                                ...doc,
                                content: '' // Return empty string if decryption fails, never return encrypted content
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
                console.error(`[NETLIFY] Error processing document ${doc.id}:`, processError);
                return {
                    ...doc,
                    content: '' // Return empty string if processing fails, never return encrypted content
                };
            }
        }));
        
        const decryptedDocuments = processedDocuments;

        return {
            statusCode: 200,
            headers,
            body: JSON.stringify({ documents: decryptedDocuments || [] })
        };

    } catch (error) {
        console.error('Get documents error:', error);
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({ error: 'Failed to fetch documents' })
        };
    }
}

// Upload document
async function handleUploadDocument(event, headers) {
    try {
        const body = JSON.parse(event.body);
        const { title, content, document_type = 'general', metadata = {} } = body;

        if (!title || !content) {
            return {
                statusCode: 400,
                headers,
                body: JSON.stringify({ error: 'Title and content are required' })
            };
        }

        const { data: document, error } = await supabase
            .from('knowledge_documents')
            .insert([
                {
                    title,
                    content,
                    document_type,
                    file_size: content.length,
                    learning_metadata: metadata
                }
            ])
            .select()
            .limit(1).maybeSingle();

        if (error) {
            console.error('Error uploading document:', error);
            return {
                statusCode: 500,
                headers,
                body: JSON.stringify({ error: 'Failed to upload document' })
            };
        }

        return {
            statusCode: 201,
            headers,
            body: JSON.stringify({ document })
        };

    } catch (error) {
        console.error('Upload document error:', error);
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({ error: 'Failed to upload document' })
        };
    }
}

// Delete document
async function handleDeleteDocument(event, headers) {
    try {
        const { id } = JSON.parse(event.body);

        if (!id) {
            return {
                statusCode: 400,
                headers,
                body: JSON.stringify({ error: 'Document ID is required' })
            };
        }

        const { error } = await supabase
            .from('knowledge_documents')
            .delete()
            .eq('id', id);

        if (error) {
            console.error('Error deleting document:', error);
            return {
                statusCode: 500,
                headers,
                body: JSON.stringify({ error: 'Failed to delete document' })
            };
        }

        return {
            statusCode: 200,
            headers,
            body: JSON.stringify({ success: true })
        };

    } catch (error) {
        console.error('Delete document error:', error);
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({ error: 'Failed to delete document' })
        };
    }
}
