/**
 * Obfuscation Worker Process
 * Handles obfuscation in a separate process to avoid blocking the main server
 */

const JavaScriptObfuscator = require('javascript-obfuscator');

process.on('message', ({ code, options }) => {
  try {
    const obfuscatedCode = JavaScriptObfuscator.obfuscate(code, options).getObfuscatedCode();
    process.send({ success: true, obfuscatedCode });
  } catch (error) {
    process.send({ success: false, error: error.message });
  }
});

// Handle uncaught exceptions in the worker process
process.on('uncaughtException', (error) => {
  try {
    process.send({ success: false, error: error.message });
  } catch (sendError) {
    // If we can't send, just exit
    process.exit(1);
  }
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason) => {
  try {
    const errorMessage = reason instanceof Error ? reason.message : String(reason);
    process.send({ success: false, error: errorMessage });
  } catch (sendError) {
    // If we can't send, just exit
    process.exit(1);
  }
});

