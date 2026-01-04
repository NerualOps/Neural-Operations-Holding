// Secure Static File Handler
// Obfuscates all JavaScript files in production
// Blocks all source code access
// © 2025 Neural Operation's & Holding's LLC. All rights reserved.

const fs = require('fs');
const path = require('path');
const { serveObfuscatedFile } = require('./production-obfuscation');
const { secureHTML } = require('./html-security');
const JavaScriptObfuscator = require('javascript-obfuscator');
const isProduction = process.env.NODE_ENV === 'production';

// Optimized obfuscation for all files - balance security and performance
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

// Files that should NEVER be served (blocked in production)
const blockedFiles = [
  /\.env$/i,
  /\.config\.js$/i,
  /\.config\.json$/i,
  /package(-lock)?\.json$/i,
  /\.md$/i,
  /\.ts$/i,
  /\.tsx$/i,
  /\.jsx$/i,
  /\.vue$/i,
  /\.svelte$/i,
  /\.map$/i,
  /\.log$/i,
  /\.git/i,
  /\.gitignore$/i,
  /\.eslintrc/i,
  /\.prettierrc/i,
  /tsconfig\.json$/i,
  /webpack\.config/i,
  /\.test\.js$/i,
  /\.spec\.js$/i
];

// Files that are allowed (CDN, libraries, etc.) - but still obfuscated if JS
const allowedPaths = [
  '/node_modules/',
  '/cdn/',
  '/assets/',
  '/Imagies/',
  '/images/',
  '/img/',
  '/fonts/',
  '/favicon.ico',
  '/robots.txt'
];

// Files that should NEVER be obfuscated (external libraries)
const neverObfuscate = [
  '/node_modules/',
  '/cdn/',
  '/assets/libs/',
  '/services/libs/supabase.js'  // External library
];

// Files that should be obfuscated
const obfuscateExtensions = ['.js'];

function isBlocked(filePath) {
  if (!isProduction) return false;
  
  const normalizedPath = path.normalize(filePath).replace(/\\/g, '/');
  
  // Check allowed paths first
  if (allowedPaths.some(allowed => normalizedPath.includes(allowed))) {
    return false;
  }
  
  // Check blocked patterns
  return blockedFiles.some(pattern => pattern.test(normalizedPath));
}

function shouldObfuscate(filePath) {
  if (!isProduction) return false;
  
  const normalizedPath = path.normalize(filePath).replace(/\\/g, '/');
  
  // Don't obfuscate external libraries
  if (neverObfuscate.some(path => normalizedPath.includes(path))) {
    return false;
  }
  
  // Obfuscate ALL other JS files
  return obfuscateExtensions.some(ext => filePath.endsWith(ext));
}

function obfuscateCode(code) {
  if (!isProduction) return code;
  
  try {
    return JavaScriptObfuscator.obfuscate(code, obfuscationOptions).getObfuscatedCode();
  } catch (error) {
    return code;
  }
}

function secureStaticFile(req, res, filePath) {
  try {
    // Security: Normalize and validate path
    const normalizedPath = path.normalize(filePath).replace(/^(\.\.(\/|\\|$))+/, '');
    const fullPath = path.resolve(normalizedPath);
    const projectRoot = path.resolve(__dirname, '..');
    
    // Ensure path is within project root
    if (!fullPath.startsWith(projectRoot)) {
      return res.status(403).send('Access denied');
    }
    
    // Check if file is blocked
    if (isBlocked(normalizedPath)) {
      return res.status(403).send('Access denied - Source files are protected');
    }
    
    if (!fs.existsSync(fullPath)) {
      return res.status(404).send('File not found');
    }
    
    const stats = fs.statSync(fullPath);
    if (!stats.isFile()) {
      return res.status(403).send('Access denied');
    }
    
    let content = fs.readFileSync(fullPath, 'utf8');
    const ext = path.extname(fullPath).toLowerCase();
    
    // Handle HTML files
    if (ext === '.html') {
      if (isProduction) {
        // secureHTML already includes 3-pass minification
        content = secureHTML(content);
      }
      res.set({
        'Content-Type': 'text/html',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY'
      });
      return res.send(content);
    }
    
    // Handle JavaScript files - ALL JS files must be obfuscated or blocked in production
    if (ext === '.js') {
      if (isProduction) {
        // In production, obfuscate ALL JS files (except external libraries)
        if (shouldObfuscate(normalizedPath)) {
          content = obfuscateCode(content);
          res.set({
            'Content-Type': 'application/javascript',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY'
          });
          return res.send(content);
        } else {
          // External library - serve as-is but with security headers
          res.set({
            'Content-Type': 'application/javascript',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY'
          });
          return res.send(content);
        }
      } else {
        // Development: serve as-is
        res.set('Content-Type', 'application/javascript');
        return res.send(content);
      }
    }
    
    // Handle CSS files
    if (ext === '.css') {
      res.set('Content-Type', 'text/css');
      return res.send(content);
    }
    
    // Handle other files (images, fonts, etc.) - binary files
    const mimeTypes = {
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml',
      '.ico': 'image/x-icon',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2',
      '.ttf': 'font/ttf',
      '.eot': 'application/vnd.ms-fontobject'
    };
    
    if (mimeTypes[ext]) {
      res.set('Content-Type', mimeTypes[ext]);
      // For binary files, read as buffer and send (no sendFile to avoid bypass)
      const buffer = fs.readFileSync(fullPath);
      return res.send(buffer);
    }
    
    // In production, block any other file types that aren't explicitly handled
    if (isProduction) {
      return res.status(403).send('Access denied - File type not allowed');
    }
    
    // Development: allow other files (read and send, not sendFile)
    const buffer = fs.readFileSync(fullPath);
    res.send(buffer);
  } catch (error) {
    res.status(500).send('Error loading file');
  }
}

module.exports = { secureStaticFile, isBlocked, shouldObfuscate };

