// Build script - Obfuscates JavaScript files for production
// © 2025 Neural Operation's & Holding's LLC. All rights reserved.

const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

// Parse command line arguments
// NEVER obfuscate locally - only when explicitly requested (e.g., on Render)
const args = process.argv.slice(2);
const isProduction = process.env.NODE_ENV === 'production';
// Only obfuscate if explicitly requested via flag (Render postbuild script)
const obfuscate = args.includes('--obfuscate=true') || args.includes('--obfuscate');
// antiDebug flag parsed but not currently used (debugProtection is disabled)
// const antiDebug = args.includes('--anti-debug=true') || args.includes('--anti-debug');

// Obfuscation options - production-grade protection
const obfuscationOptions = {
  compact: true,
  controlFlowFlattening: true,
  controlFlowFlatteningThreshold: 0.8,
  deadCodeInjection: true,
  deadCodeInjectionThreshold: 0.5,
  debugProtection: false, // Disabled - causes debugger triggers in console
  debugProtectionInterval: 0,
  disableConsoleOutput: isProduction,
  identifierNamesGenerator: 'hexadecimal',
  log: false,
  numbersToExpressions: true,
  renameGlobals: false,
  selfDefending: true,
  simplify: true,
  splitStrings: true,
  splitStringsChunkLength: 8,
  stringArray: true,
  stringArrayCallsTransform: true,
  stringArrayCallsTransformThreshold: 0.8,
  stringArrayEncoding: ['base64', 'rc4'],
  stringArrayIndexShift: true,
  stringArrayRotate: true,
  stringArrayShuffle: true,
  stringArrayWrappersCount: 3,
  stringArrayWrappersChainedCalls: true,
  stringArrayWrappersParametersMaxCount: 5,
  stringArrayWrappersType: 'function',
  stringArrayThreshold: 0.8,
  transformObjectKeys: true,
  unicodeEscapeSequence: false
};

// Files to obfuscate (source -> destination)
// ALL client-side JavaScript files must be obfuscated in production
const filesToObfuscate = [
  // Core AI engine files (loaded by HTML)
  { src: 'core/epsilon-learning-engine.js', dest: 'obfuscated/epsilon-learning-engine.js' },
  { src: 'core/epsilon-language-engine.js', dest: 'obfuscated/epsilon-language-engine.js' },
  { src: 'core/epsilon-self-learning.js', dest: 'obfuscated/epsilon-self-learning.js' },
  
  // RAG service files (loaded by epsilon-learning-engine.js)
  { src: 'services/rag-embedding-service.js', dest: 'obfuscated/rag-embedding-service.js' },
  { src: 'services/rag-llm-service.js', dest: 'obfuscated/rag-llm-service.js' },
  { src: 'services/rag-document-processor.js', dest: 'obfuscated/rag-document-processor.js' },
  
  // Library files (loaded by HTML)
  { src: 'services/libs/supabase.js', dest: 'obfuscated/supabase.js' },
  
  // API proxy (if served to clients)
  { src: 'api/supabase-proxy.js', dest: 'obfuscated/supabase-proxy.js' },
  
  // Document learning service (if served to clients)
  { src: 'services/document-learning-service.js', dest: 'obfuscated/document-learning-service.js' },
  
  // AI Core files (if any are client-side)
  { src: 'services/ai-core/epsilon-embeddings.js', dest: 'obfuscated/epsilon-embeddings.js' },
  { src: 'services/ai-core/epsilon-tokenizer.js', dest: 'obfuscated/epsilon-tokenizer.js' }
];

// Ensure obfuscated directory exists
const obfuscatedDir = path.join(__dirname, 'obfuscated');
if (!fs.existsSync(obfuscatedDir)) {
  fs.mkdirSync(obfuscatedDir, { recursive: true });
}

function obfuscateFile(srcPath, destPath) {
  try {
    const fullSrcPath = path.join(__dirname, srcPath);
    const fullDestPath = path.join(__dirname, destPath);
    
    if (!fs.existsSync(fullSrcPath)) {
      console.warn(`[BUILD] Source file not found: ${srcPath}`);
      return false;
    }
    
    const code = fs.readFileSync(fullSrcPath, 'utf8');
    
    if (obfuscate || isProduction) {
      // Always obfuscate in production, or if --obfuscate flag is set
      const obfuscatedCode = JavaScriptObfuscator.obfuscate(code, obfuscationOptions).getObfuscatedCode();
      fs.writeFileSync(fullDestPath, obfuscatedCode, 'utf8');
    } else {
      // Just copy the file if not obfuscating (development)
      fs.copyFileSync(fullSrcPath, fullDestPath);
    }
    
    return true;
  } catch (error) {
    console.error(`[BUILD] Error processing ${srcPath}:`, error.message);
    return false;
  }
}

// Copy Supabase library (no obfuscation needed, already minified)
function copySupabase() {
  try {
    const supabaseSrc = path.join(__dirname, 'node_modules/@supabase/supabase-js/dist/umd/supabase.js');
    const supabaseDest = path.join(__dirname, 'obfuscated/supabase.js');
    const supabaseLibDest = path.join(__dirname, 'services/libs/supabase.js');
    
    if (fs.existsSync(supabaseSrc)) {
      fs.copyFileSync(supabaseSrc, supabaseDest);
      fs.copyFileSync(supabaseSrc, supabaseLibDest);
      // Silent - no console.log
    } else {
      console.warn('[BUILD] Supabase library not found in node_modules');
    }
  } catch (error) {
    console.error('[BUILD] Error copying Supabase library:', error.message);
  }
}

// Main build function
function build() {
  // Silent - no console.log
  // Silent - no console.log
  // Silent - no console.log
  // Silent - no console.log
  
  let successCount = 0;
  let failCount = 0;
  
  // Process all files
  for (const file of filesToObfuscate) {
    if (obfuscateFile(file.src, file.dest)) {
      successCount++;
    } else {
      failCount++;
    }
  }
  
  // Copy Supabase library
  copySupabase();
  
  // Silent - no console.log
  // Silent - no console.log
}

// Run build
build();

