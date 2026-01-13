// Production Obfuscation Helper
// Serves obfuscated files in production, creates them on-demand if needed
// Never obfuscates locally - only in production deployment

const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const JavaScriptObfuscator = require('javascript-obfuscator');

const isProduction = process.env.NODE_ENV === 'production';

// Optimized obfuscation for production - balance security and performance
const obfuscationOptions = {
  compact: true,
  controlFlowFlattening: true,
  controlFlowFlatteningThreshold: 0.75,
  deadCodeInjection: true,
  deadCodeInjectionThreshold: 0.4,
  debugProtection: false,
  debugProtectionInterval: 0,
  disableConsoleOutput: false, // Allow console for debugging
  identifierNamesGenerator: 'hexadecimal',
  log: false,
  numbersToExpressions: true,
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
  selfDefending: false, // Disable to prevent breaking navigation
  simplify: false,
  splitStrings: true,
  splitStringsChunkLength: 10,
  stringArray: true,
  stringArrayCallsTransform: true,
  stringArrayCallsTransformThreshold: 0.75,
  stringArrayEncoding: ['base64'],
  stringArrayIndexShift: true,
  stringArrayRotate: true,
  stringArrayShuffle: true,
  stringArrayWrappersCount: 2,
  stringArrayWrappersChainedCalls: true,
  stringArrayWrappersParametersMaxCount: 2,
  stringArrayWrappersType: 'function',
  stringArrayThreshold: 0.75,
  transformObjectKeys: true,
  unicodeEscapeSequence: true
};

function obfuscateFileOnDemand(sourcePath, obfuscatedPath) {
  try {
    // Security: Normalize paths to prevent path traversal attacks
    const normalizedSourcePath = path.normalize(sourcePath).replace(/^(\.\.(\/|\\|$))+/, '');
    const normalizedObfuscatedPath = path.normalize(obfuscatedPath).replace(/^(\.\.(\/|\\|$))+/, '');
    
    // Additional security: Ensure paths don't contain directory traversal
    if (normalizedSourcePath.includes('..') || normalizedObfuscatedPath.includes('..')) {
      console.error(`[SECURITY] Path traversal attempt detected: ${sourcePath} or ${obfuscatedPath}`);
      return false;
    }
    
    const fullSourcePath = path.join(__dirname, '..', normalizedSourcePath);
    const fullObfuscatedPath = path.join(__dirname, '..', normalizedObfuscatedPath);
    
    // Security: Ensure resolved paths are within the project directory
    const projectRoot = path.resolve(__dirname, '..');
    const resolvedSourcePath = path.resolve(fullSourcePath);
    const resolvedObfuscatedPath = path.resolve(fullObfuscatedPath);
    
    if (!resolvedSourcePath.startsWith(projectRoot) || !resolvedObfuscatedPath.startsWith(projectRoot)) {
      console.error(`[SECURITY] Path outside project root detected: ${sourcePath} or ${obfuscatedPath}`);
      return false;
    }
    
    if (!fs.existsSync(fullSourcePath)) {
      return false;
    }
    
    // Ensure obfuscated directory exists
    const obfuscatedDir = path.dirname(fullObfuscatedPath);
    if (!fs.existsSync(obfuscatedDir)) {
      fs.mkdirSync(obfuscatedDir, { recursive: true });
    }
    
    // Read file as buffer first to detect BOM
    const fileBuffer = fs.readFileSync(fullSourcePath);
    
    // Remove UTF-8 BOM (0xEF 0xBB 0xBF) if present
    let code;
    if (fileBuffer[0] === 0xEF && fileBuffer[1] === 0xBB && fileBuffer[2] === 0xBF) {
      code = fileBuffer.slice(3).toString('utf8');
    } else if (fileBuffer[0] === 0xFE && fileBuffer[1] === 0xFF) {
      // UTF-16 BE BOM
      code = fileBuffer.slice(2).toString('utf16le');
    } else if (fileBuffer[0] === 0xFF && fileBuffer[1] === 0xFE) {
      // UTF-16 LE BOM
      code = fileBuffer.slice(2).toString('utf16le');
    } else {
      code = fileBuffer.toString('utf8');
    }
    
    // Remove any other non-printable characters at the start
    code = code.replace(/^[\u200B-\u200D\uFEFF]+/, '');
    
    // Clean up any problematic characters that might break obfuscation
    code = code.replace(/\u0000/g, ''); // Remove null bytes
    
    // Skip obfuscation for very large files
    if (code.length > 5000000) {
      console.warn(`[OBFUSCATION] File too large, copying without obfuscation: ${sourcePath}`);
      fs.copyFileSync(fullSourcePath, fullObfuscatedPath);
      return true;
    }
    
    try {
      const obfuscatedCode = JavaScriptObfuscator.obfuscate(code, obfuscationOptions).getObfuscatedCode();
      fs.writeFileSync(fullObfuscatedPath, obfuscatedCode, 'utf8');
      console.log(`[OBFUSCATION] Successfully obfuscated: ${obfuscatedPath}`);
      return true;
    } catch (obfuscateError) {
      console.error(`[OBFUSCATION] Obfuscation failed for ${sourcePath}:`, obfuscateError.message);
      if (isProduction) {
        return false;
      }
      // Copy source as fallback in dev
      fs.copyFileSync(fullSourcePath, fullObfuscatedPath);
      return true;
    }
  } catch (error) {
    console.error(`[OBFUSCATION] Error obfuscating ${sourcePath}:`, error.message);
    return false;
  }
}

function serveObfuscatedFile(req, res, sourcePath, obfuscatedPath) {
  try {
    // Security: Normalize paths to prevent path traversal attacks
    const normalizedSourcePath = path.normalize(sourcePath).replace(/^(\.\.(\/|\\|$))+/, '');
    const normalizedObfuscatedPath = path.normalize(obfuscatedPath).replace(/^(\.\.(\/|\\|$))+/, '');
    
    // Additional security: Ensure paths don't contain directory traversal
    if (normalizedSourcePath.includes('..') || normalizedObfuscatedPath.includes('..')) {
      console.error(`[SECURITY] Path traversal attempt detected: ${sourcePath} or ${obfuscatedPath}`);
      return res.status(403).send('// Invalid path');
    }
    
    const projectRoot = path.resolve(__dirname, '..');
    let filePath;
    
    if (isProduction) {
      // In production, use cached obfuscated files for performance
      const obfuscatedFilePath = path.join(__dirname, '..', normalizedObfuscatedPath);
      const sourceFilePath = path.join(__dirname, '..', normalizedSourcePath);
      const resolvedSourcePath = path.resolve(sourceFilePath);
      
      if (!resolvedSourcePath.startsWith(projectRoot)) {
        console.error(`[SECURITY] Path outside project root detected: ${sourcePath}`);
        return res.status(403).send('// Invalid path');
      }
      
      if (!fs.existsSync(sourceFilePath)) {
        console.error(`[OBFUSCATION] Source file not found: ${normalizedSourcePath}`);
        // In development, try to serve from core directory directly
        if (!isProduction) {
          const altPath = path.join(__dirname, '..', 'core', path.basename(normalizedSourcePath));
          if (fs.existsSync(altPath)) {
            const altContent = fs.readFileSync(altPath, 'utf8');
            res.set('Content-Type', 'application/javascript');
            return res.send(altContent);
          }
        }
        return res.status(404).send('// File not found');
      }
      
      // Check if cached obfuscated file exists and is newer than source
      let content;
      let useCache = false;
      
      if (fs.existsSync(obfuscatedFilePath)) {
        const sourceStats = fs.statSync(sourceFilePath);
        const obfuscatedStats = fs.statSync(obfuscatedFilePath);
        // Use cache if obfuscated file is newer or same age as source
        if (obfuscatedStats.mtime >= sourceStats.mtime) {
          content = fs.readFileSync(obfuscatedFilePath, 'utf8');
          useCache = true;
        }
      }
      
      // If no cache or cache is stale, obfuscate on-the-fly
      if (!useCache) {
        const sourceContent = fs.readFileSync(sourceFilePath, 'utf8');
        try {
          content = JavaScriptObfuscator.obfuscate(sourceContent, obfuscationOptions).getObfuscatedCode();
          // Cache the obfuscated file for future requests
          try {
            const obfuscatedDir = path.dirname(obfuscatedFilePath);
            if (!fs.existsSync(obfuscatedDir)) {
              fs.mkdirSync(obfuscatedDir, { recursive: true });
            }
            fs.writeFileSync(obfuscatedFilePath, content, 'utf8');
          } catch (cacheError) {
            // If caching fails, continue without cache
          }
        } catch (obfError) {
          console.error(`[OBFUSCATION] Failed to obfuscate: ${normalizedSourcePath}`, obfError.message);
          // In production, try to serve source file directly if obfuscation fails
          try {
            const sourceContent = fs.readFileSync(sourceFilePath, 'utf8');
            res.set('Content-Type', 'application/javascript');
            return res.send(sourceContent);
          } catch (fallbackError) {
            return res.status(500).send('// Error loading file');
          }
        }
      }
      
      const etag = `"${crypto.createHash('md5').update(content).digest('hex')}"`;
      const ifNoneMatch = req.headers['if-none-match'];
      if (ifNoneMatch === etag) {
        return res.status(304).end();
      }
      
      res.set({
        'Content-Type': 'application/javascript',
        'ETag': etag,
        'Cache-Control': 'public, max-age=3600',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY'
      });
      
      return res.send(content);
    } else {
      // In development, ALWAYS serve source files (never obfuscate locally)
      filePath = path.join(__dirname, '..', normalizedSourcePath);
      
      // Security: Ensure resolved path is within project root
      const resolvedPath = path.resolve(filePath);
      if (!resolvedPath.startsWith(projectRoot)) {
        console.error(`[SECURITY] Path outside project root detected: ${sourcePath}`);
        return res.status(403).send('// Invalid path');
      }
      
      if (!fs.existsSync(filePath)) {
        console.error(`[OBFUSCATION] Source file not found: ${normalizedSourcePath}`);
        // Try core directory as fallback
        const coreFallbackPath = path.join(__dirname, '..', 'core', path.basename(normalizedSourcePath));
        if (fs.existsSync(coreFallbackPath)) {
          filePath = coreFallbackPath;
        } else {
          return res.status(404).send('File not found');
        }
      }
    }
    
    // Read file as buffer to handle BOM
    const fileBuffer = fs.readFileSync(filePath);
    let content;
    // Remove UTF-8 BOM (0xEF 0xBB 0xBF) if present
    if (fileBuffer[0] === 0xEF && fileBuffer[1] === 0xBB && fileBuffer[2] === 0xBF) {
      content = fileBuffer.slice(3).toString('utf8');
    } else {
      content = fileBuffer.toString('utf8');
    }
    // Remove any other non-printable characters at the start
    content = content.replace(/^[\u200B-\u200D\uFEFF]+/, '');
    const etag = `"${crypto.createHash('md5').update(content).digest('hex')}"`;
    const ifNoneMatch = req.headers['if-none-match'];
    
    if (ifNoneMatch === etag) {
      return res.status(304).end();
    }
    
    res.set({
      'Content-Type': 'application/javascript',
      'ETag': etag,
      'Cache-Control': 'public, max-age=3600',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY'
    });
    
    res.send(content);
  } catch (error) {
    console.error(`[OBFUSCATION] Error serving file ${sourcePath}:`, error.message);
    console.error(`[OBFUSCATION] Stack:`, error.stack);
    // Try to serve source file directly as fallback (both dev and production)
    try {
      const fallbackPath = path.join(__dirname, '..', normalizedSourcePath);
      if (fs.existsSync(fallbackPath)) {
        const fallbackContent = fs.readFileSync(fallbackPath, 'utf8');
        res.set('Content-Type', 'application/javascript');
        return res.send(fallbackContent);
      }
      // Also try core directory as fallback
      const coreFallbackPath = path.join(__dirname, '..', 'core', path.basename(normalizedSourcePath));
      if (fs.existsSync(coreFallbackPath)) {
        const fallbackContent = fs.readFileSync(coreFallbackPath, 'utf8');
        res.set('Content-Type', 'application/javascript');
        return res.send(fallbackContent);
      }
    } catch (fallbackError) {
      // Ignore fallback errors
    }
    return res.status(500).send('// Error loading file');
  }
}

module.exports = { serveObfuscatedFile, isProduction };

