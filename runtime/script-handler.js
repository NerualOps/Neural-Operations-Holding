// script-handler.js - UPDATED
const path = require('path');
const fs = require('fs');
const { fork } = require('child_process');
const JavaScriptObfuscator = require('javascript-obfuscator');
const { sanitizeFilename } = require('./sanitize');
const { logSecurityEvent } = require('./logging');

// Enhanced phantom script endpoint
const handleScriptRequest = (req, res) => {
  // Silent - no console.log
  
  // Prevent double-send errors
  let responded = false;
  const safeSend = (status, body, headers = {}) => {
    if (responded || res.headersSent) return;
    responded = true;
    if (headers) {
      for (const [key, value] of Object.entries(headers)) {
        res.setHeader(key, value);
      }
    }
    res.status(status).send(body);
  };
  
  const filename = req.params.filename;
  
  // Sanitize filename to prevent path traversal
  const sanitizedFilename = sanitizeFilename(filename);
  if (sanitizedFilename !== filename) {
    logSecurityEvent('PATH_TRAVERSAL_ATTEMPT', {
      path: req.path,
      ip: req.ip,
      filename: filename
    }, 'warn');
    return safeSend(400, 'Invalid filename');
  }

  // Only allow specific file extensions
  if (!filename.match(/\.(js|mjs)$/i)) {
    return safeSend(400, 'Only JavaScript files can be accessed');
  }

  // Whitelist of allowed files
  const allowedFiles = [
    'epsilon-chunk-1.js',
    'epsilon-chunk-2.js',
    'epsilon-chunk-3.js',
    'epsilon-learning-engine.js'
  ];

  if (!allowedFiles.includes(filename)) {
    logSecurityEvent('UNAUTHORIZED_SCRIPT_ACCESS', {
      path: req.path,
      ip: req.ip,
      filename: filename
    }, 'warn');
    return safeSend(403, '// Access denied');
  }

  const filePath = path.join(__dirname, filename);

  if (!fs.existsSync(filePath)) {
    console.error(`Phantom Script: File not found at ${filePath}`);
    return safeSend(404, '// Script not found.');
  }

  try {
    const code = fs.readFileSync(filePath, 'utf8');
    // Silent - no console.log
    
    // Always obfuscate files regardless of size
    // Silent - no console.log
    
    // FIXED: For large files, use lighter obfuscation with corrected options
    const options = code.length > 500000 ? {
      compact: true,
      controlFlowFlattening: false,
      deadCodeInjection: false,
      stringArray: true,
      stringArrayEncoding: ['base64'],
      stringArrayThreshold: 0.5,
      // Remove problematic options
      debugProtection: false,
      debugProtectionInterval: 0 // Set to 0 instead of true
    } : {
      // Full obfuscation options for smaller files - FIXED
      compact: true,
      controlFlowFlattening: true,
      controlFlowFlatteningThreshold: 0.5,
      deadCodeInjection: true,
      deadCodeInjectionThreshold: 0.3,
      debugProtection: false, // Changed from true
      debugProtectionInterval: 0, // Changed from true to 0
      disableConsoleOutput: false,
      identifierNamesGenerator: 'hexadecimal',
      log: false,
      renameGlobals: false,
      reservedNames: [
        'navigateToEpsilon',
        'showComingSoon',
        'closeModal',
        'toggleProfileDropdown',
        'toggleOwnerDropdown',
        'toggleMobileMenu',
        'applySuggestion',
        'approveDeployment',
        'rejectDeployment',
        'getCsrfToken',
        'showMessage',
        'resetButton',
        'addEventListener',
        'removeEventListener',
        'getElementById',
        'querySelector',
        'querySelectorAll'
      ],
      reservedStrings: [
        'addEventListener',
        'removeEventListener',
        'getElementById',
        'querySelector',
        'querySelectorAll',
        'document',
        'window',
        'localStorage',
        'sessionStorage',
        'fetch',
        'XMLHttpRequest',
        'navigateToEpsilon',
        'showComingSoon',
        'closeModal',
        'toggleProfileDropdown',
        'toggleOwnerDropdown',
        'toggleMobileMenu',
        'applySuggestion',
        'approveDeployment',
        'rejectDeployment'
      ],
      rotateStringArray: true,
      selfDefending: true,
      shuffleStringArray: true,
      splitStrings: true,
      splitStringsChunkLength: 10,
      stringArray: true,
      stringArrayEncoding: ['base64'],
      stringArrayThreshold: 0.5,
      transformObjectKeys: true,
      unicodeEscapeSequence: false
    };
    
    // Generate a cache key based on file content and options
    const cacheKey = require('crypto').createHash('md5').update(code + JSON.stringify(options)).digest('hex');
    
    // Check if client has a valid cached version
    const ifNoneMatch = req.headers['if-none-match'];
    if (ifNoneMatch && ifNoneMatch.includes(cacheKey)) {
      return safeSend(304, ''); // Not Modified
    }
    
    // Try direct obfuscation first for smaller files
    if (code.length < 100000) {
      try {
        // Silent - no console.log
        const obfuscatedCode = JavaScriptObfuscator.obfuscate(code, options).getObfuscatedCode();
        
        // Set headers and send response with strong caching
        return safeSend(200, obfuscatedCode, {
          'Content-Type': 'application/javascript',
          'Cache-Control': 'public, max-age=86400',
          'ETag': `"${cacheKey}"`,
          'X-Content-Type-Options': 'nosniff'
        });
      } catch (directError) {
        console.error(`Phantom Script: Direct obfuscation failed: ${directError.message}`);
        // Fall through to worker approach
      }
    }
    
    // Use worker process for obfuscation to avoid memory issues
    const worker = fork(path.join(__dirname, 'obfuscate-worker.js'));
    
    // Set a timeout in case the worker takes too long
    const timeout = setTimeout(() => {
      worker.kill();
      // Silent - no console.log
      
      // Serve original as fallback
      safeSend(200, code, {
        'Content-Type': 'application/javascript',
        'Cache-Control': 'public, max-age=3600',
        'ETag': `"${require('crypto').createHash('md5').update(code).digest('hex')}"`
      });
    }, 10000); // 10 second timeout
    
    worker.on('message', (result) => {
      clearTimeout(timeout);
      
      if (result.success) {
        // Silent - no console.log
        
        // Set headers and send response with strong caching
        safeSend(200, result.obfuscatedCode, {
          'Content-Type': 'application/javascript',
          'Cache-Control': 'public, max-age=86400',
          'ETag': `"${cacheKey}"`,
          'X-Content-Type-Options': 'nosniff'
        });
      } else {
        console.error(`Phantom Script: Worker error: ${result.error}`);
        
        // Fallback to original
        safeSend(200, code, {
          'Content-Type': 'application/javascript',
          'Cache-Control': 'public, max-age=3600',
          'ETag': `"${require('crypto').createHash('md5').update(code).digest('hex')}"`
        });
      }
      
      worker.kill();
    });
    
    worker.on('error', (error) => {
      clearTimeout(timeout);
      console.error(`Phantom Script: Worker error: ${error.message}`);
      worker.kill();
      
      // Fallback to original
      safeSend(200, code, {
        'Content-Type': 'application/javascript',
        'Cache-Control': 'public, max-age=3600',
        'ETag': `"${require('crypto').createHash('md5').update(code).digest('hex')}"`
      });
    });
    
    // Send the code to the worker
    worker.send({ code, options });
    
  } catch (error) {
    console.error(`Phantom Script: Error: ${error.message}`);
    return safeSend(500, '// Error processing script');
  }
};

module.exports = { handleScriptRequest };
