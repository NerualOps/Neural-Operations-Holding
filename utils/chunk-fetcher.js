/**
 * Unified Chunk Fetching Utility
 * Fetches doc_chunks in small, reliable batches to prevent Supabase statement timeouts
 * Consistent with upload batching strategy - if we batch uploads, we batch downloads
 * 
 * FEATURES:
 * - Automatic timeout detection and retry with smaller batches
 * - Progressive batch size reduction on errors
 * - Handles Cloudflare errors and connection issues
 * - Sorts chunks by index for correct order
 */

/**
 * Detect if error is a statement timeout
 * @param {object|string} error - Error object or string
 * @returns {boolean} True if timeout error
 */
function isTimeoutError(error) {
  const errorStr = error?.message || error?.toString() || '';
  return errorStr.includes('statement timeout') || 
         errorStr.includes('57014') ||
         errorStr.includes('canceling statement');
}

/**
 * Fetch all chunks for a document in small batches
 * @param {object} supabase - Supabase client instance
 * @param {string} documentId - UUID of the document
 * @param {object} options - Configuration
 * @returns {Promise<Array>} Array of chunk objects {chunk_text, chunk_index}
 */
async function fetchChunksInBatches(supabase, documentId, options = {}) {
  const {
    batchSize = 20,
    silent = false,
    maxRetries = 10,
    maxChunks = null
  } = options;
  
  const docIdStr = String(documentId).trim();
  if (!docIdStr || docIdStr === 'undefined' || docIdStr === 'null') {
    if (!silent) console.warn(`[CHUNK FETCHER] Invalid document_id: ${documentId}`);
    return [];
  }
  
  const allChunks = [];
  let offset = 0;
  let hasMore = true;
  let currentBatchSize = batchSize;
  let consecutiveErrors = 0;
  let lastSuccessfulOffset = -1;
  
  while (hasMore) {
    if (maxChunks !== null && allChunks.length >= maxChunks) {
      if (!silent) console.log(`[CHUNK FETCHER] Reached maxChunks limit (${maxChunks}), stopping fetch`);
      break;
    }
    try {
      const queryPromise = supabase
        .from('doc_chunks')
        .select('chunk_text, chunk_index')
        .eq('document_id', docIdStr)
        .order('chunk_index', { ascending: true })
        .range(offset, offset + currentBatchSize - 1);
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Query timeout after 15s')), 15000)
      );
      
      const { data: batchChunks, error: fetchError } = await Promise.race([
        queryPromise,
        timeoutPromise
      ]).catch(e => {
        // Handle timeout promise rejection
        if (e.message === 'Query timeout after 15s') {
          return { data: null, error: { message: 'canceling statement due to statement timeout', code: '57014' } };
        }
        throw e;
      });
      
      if (fetchError) {
        const errorStr = fetchError?.message || fetchError?.toString() || '';
        const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                           errorStr.includes('Cloudflare') || 
                           errorStr.includes('522') || 
                           errorStr.includes('521');
        const isTimeout = isTimeoutError(fetchError);
        
        if (isHtmlError) {
          if (!silent) console.warn(`[CHUNK FETCHER] Supabase connection issue (Cloudflare error) for document ${docIdStr}`);
          break;
        }
        
        if (isTimeout && currentBatchSize > 5 && consecutiveErrors < maxRetries) {
          const newBatchSize = Math.max(5, Math.floor(currentBatchSize / 2));
          consecutiveErrors++;
          
          // If we've had multiple failures at this offset, try smaller batch
          if (consecutiveErrors >= 2 && offset === lastSuccessfulOffset + currentBatchSize) {
            currentBatchSize = newBatchSize;
            // Don't advance offset - retry same range with smaller batch
          } else {
            currentBatchSize = newBatchSize;
            // Reset to last successful position if we've failed too many times
            if (consecutiveErrors >= 3 && lastSuccessfulOffset >= 0) {
              offset = lastSuccessfulOffset + 1;
              currentBatchSize = 5; // Minimum batch size
            }
          }
          
          if (!silent) console.warn(`[CHUNK FETCHER] Timeout at offset ${offset}, reducing batch size to ${currentBatchSize} (attempt ${consecutiveErrors}/${maxRetries})`);
          
          // Add delay before retry
          await new Promise(resolve => setTimeout(resolve, 500));
          continue; // Retry with smaller batch
        }
        
        if (!silent) {
          console.warn(`[CHUNK FETCHER] Error for document ${docIdStr} at offset ${offset}: ${fetchError.message || 'Unknown error'}`);
        }
        break;
      }
      
      if (!batchChunks || batchChunks.length === 0) {
        break;
      }
      
      consecutiveErrors = 0;
      lastSuccessfulOffset = offset;
      
      if (maxChunks !== null && allChunks.length + batchChunks.length > maxChunks) {
        const remaining = maxChunks - allChunks.length;
        allChunks.push(...batchChunks.slice(0, remaining));
        hasMore = false;
        if (!silent) console.log(`[CHUNK FETCHER] Limited to ${maxChunks} chunks (stopped at ${allChunks.length})`);
      } else {
      allChunks.push(...batchChunks);
      }
      
      offset += batchChunks.length;
      
      // Reset batch size to original if we've had success
      if (currentBatchSize < batchSize && consecutiveErrors === 0) {
        currentBatchSize = Math.min(batchSize, currentBatchSize * 2); // Gradually increase back
      }
      
      if (batchChunks.length < currentBatchSize || (maxChunks !== null && allChunks.length >= maxChunks)) {
        hasMore = false;
      }
      
      // Add small delay to prevent rate limiting
      if (hasMore) {
        await new Promise(resolve => setTimeout(resolve, 200)); // Increased delay for large documents
      }
    } catch (e) {
      const isTimeout = isTimeoutError(e);
      
      if (isTimeout && currentBatchSize > 5 && consecutiveErrors < maxRetries) {
        const newBatchSize = Math.max(5, Math.floor(currentBatchSize / 2));
        currentBatchSize = newBatchSize;
        consecutiveErrors++;
        
        // Reset to last successful position if we've failed too many times
        if (consecutiveErrors >= 3 && lastSuccessfulOffset >= 0) {
          offset = lastSuccessfulOffset + 1;
          currentBatchSize = 5;
        }
        
        if (!silent) console.warn(`[CHUNK FETCHER] Exception timeout at offset ${offset}, reducing batch size to ${currentBatchSize} (attempt ${consecutiveErrors}/${maxRetries})`);
        
        // Add delay before retry
        await new Promise(resolve => setTimeout(resolve, 500));
        continue;
      }
      
      if (!silent) console.warn(`[CHUNK FETCHER] Exception for document ${docIdStr} at offset ${offset}: ${e.message}`);
      break;
    }
  }
  
  if (allChunks.length > 0) {
    allChunks.sort((a, b) => (a.chunk_index || 0) - (b.chunk_index || 0));
  }
  
  if (!silent && allChunks.length > 0) {
    console.log(`[CHUNK FETCHER] Successfully fetched ${allChunks.length} chunks for document ${docIdStr}`);
  }
  
  return allChunks;
}

/**
 * Fetch chunks and join into text string
 * @param {object} supabase - Supabase client instance
 * @param {string} documentId - UUID of the document
 * @param {string} separator - Join separator (default: '\n\n')
 * @returns {Promise<string>} Joined chunk text
 */
async function fetchChunksAsText(supabase, documentId, separator = '\n\n') {
  const chunks = await fetchChunksInBatches(supabase, documentId, { silent: true });
  return chunks.map(c => c.chunk_text).filter(t => t && t.trim()).join(separator);
}

module.exports = {
  fetchChunksInBatches,
  fetchChunksAsText
};
