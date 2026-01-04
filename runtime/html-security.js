// HTML Security and Obfuscation Middleware
// Obfuscates inline JavaScript in HTML files for production
// © 2025 Neural Operation's & Holding's LLC. All rights reserved.

const JavaScriptObfuscator = require('javascript-obfuscator');
const isProduction = process.env.NODE_ENV === 'production';

// Maximum obfuscation for production (optimized for performance)
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

// Minimal security script - only prevents code inspection, allows console and navigation
const securityScript = `
(function() {
  'use strict';
  if (typeof window === 'undefined') return;
  
  // Only block eval/Function for security, but allow normal operation
  try {
    Object.defineProperty(window, 'eval', {
      value: function() { 
        console.warn('eval() is disabled for security');
        return null; 
      },
      writable: false,
      configurable: false
    });
  } catch(e) {}
  
  try {
    Object.defineProperty(window, 'Function', {
      value: function() { 
        console.warn('Function() constructor is disabled for security');
        return function() {}; 
      },
      writable: false,
      configurable: false
    });
  } catch(e) {}
})();
`;

function obfuscateJavaScript(code) {
  if (!isProduction) return code;
  
  try {
    // For inline scripts, use lighter obfuscation to prevent wrapping issues
    // and ensure window assignments remain accessible
    const options = {
      ...obfuscationOptions,
      // Reduce intensity for inline scripts to prevent wrapping
      controlFlowFlatteningThreshold: 0.5,
      deadCodeInjectionThreshold: 0.2,
      stringArrayThreshold: 0.5,
      stringArrayWrappersCount: 1
    };
    return JavaScriptObfuscator.obfuscate(code, options).getObfuscatedCode();
  } catch (error) {
    return code;
  }
}

function obfuscateInlineScripts(html) {
  if (!isProduction) return html;
  
  // Extract and obfuscate <script> tags
  const scriptRegex = /<script(?:\s+[^>]*)?>([\s\S]*?)<\/script>/gi;
  
  return html.replace(scriptRegex, (match, scriptContent) => {
    // Skip if it's an external script (has src attribute)
    if (match.includes('src=')) {
      return match;
    }
    
    // Skip if it's a type that shouldn't be obfuscated
    if (match.includes('type="application/json"') || match.includes("type='application/json'")) {
      return match;
    }
    
    try {
      const obfuscated = obfuscateJavaScript(scriptContent);
      return match.replace(scriptContent, obfuscated);
    } catch (error) {
      return match;
    }
  });
}

function injectSecurityScript(html) {
  return html;
}

function minifyHTML(html) {
  if (!isProduction) return html;
  
  // Preserve onclick attribute strings - don't minify function calls in onclick
  const onclickPattern = /onclick\s*=\s*["']([^"']+)["']/gi;
  const onclickPreserves = [];
  let preserveIndex = 0;
  
  // Extract and preserve onclick handlers
  html = html.replace(onclickPattern, (match, handler) => {
    const placeholder = `__ONCLICK_PRESERVE_${preserveIndex}__`;
    onclickPreserves[preserveIndex] = handler;
    preserveIndex++;
    return match.replace(handler, placeholder);
  });
  
  // Run minification 3 times for maximum compression
  for (let pass = 0; pass < 3; pass++) {
    html = html
      .replace(/<!--[\s\S]*?-->/g, '') // Remove HTML comments
      .replace(/\s+/g, ' ') // Replace multiple spaces/tabs/newlines with single space
      .replace(/>\s+</g, '><') // Remove spaces between tags
      .replace(/\s+/g, ' ') // Clean up remaining whitespace again
      .replace(/\s*([{}:;,=])\s*/g, '$1') // Remove spaces around CSS/JS operators
      .replace(/;\s*}/g, '}') // Remove semicolons before closing braces
      .replace(/\s*>\s*/g, '>') // Remove spaces around > 
      .replace(/\s*<\s*/g, '<') // Remove spaces around <
      .trim();
  }
  
  // Restore preserved onclick handlers
  onclickPreserves.forEach((handler, index) => {
    html = html.replace(`__ONCLICK_PRESERVE_${index}__`, handler);
  });
  
  return html;
}

function secureHTML(html) {
  return html;
}

module.exports = { secureHTML, obfuscateJavaScript, minifyHTML };

