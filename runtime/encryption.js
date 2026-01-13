// encryption.js
const crypto = require('crypto');
const { logSecurityEvent } = require('./logging');

// Algorithm to use for encryption
const ALGORITHM = 'aes-256-gcm';

// Encrypt data
const encrypt = (text) => {
  if (!text) return null;
  
  try {
    const key = process.env.ENCRYPTION_KEY;
    if (!key) {
      console.error('ENCRYPTION_KEY not configured');
      logSecurityEvent('ENCRYPTION_KEY_MISSING', {
        error: 'Encryption key not configured'
      }, 'error');
      // SECURITY: Fail secure - return null instead of plaintext
      return null;
    }
    
    // Use a 256-bit key (32 bytes)
    const keyBuffer = crypto.createHash('sha256').update(key).digest();
    
    // Generate a random initialization vector
    const iv = crypto.randomBytes(16);
    
    // Create cipher
    const cipher = crypto.createCipheriv(ALGORITHM, keyBuffer, iv);
    
    // Encrypt the text
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    // Get the authentication tag
    const authTag = cipher.getAuthTag().toString('hex');
    
    // Return IV + Auth Tag + Encrypted data
    return iv.toString('hex') + ':' + authTag + ':' + encrypted;
  } catch (error) {
    console.error('Encryption error:', error);
    
    logSecurityEvent('ENCRYPTION_ERROR', {
      error: error.message
    }, 'error');
    
    // SECURITY: Fail secure - return null instead of plaintext to prevent data leakage
    return null;
  }
};

// Decrypt data
const decrypt = (encryptedText) => {
  if (!encryptedText) return null;

  if (typeof encryptedText !== 'string') {
    return encryptedText;
  }

  const encryptedPattern = /^[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+$/;
  if (!encryptedPattern.test(encryptedText.trim())) {
    return encryptedText;
  }
  
  try {
    const key = process.env.ENCRYPTION_KEY;
    if (!key) {
      console.error('ENCRYPTION_KEY not configured');
      return null;
    }
    
    // Use a 256-bit key (32 bytes)
    const keyBuffer = crypto.createHash('sha256').update(key).digest();
    
    // Split the encrypted text into IV, Auth Tag, and encrypted data
    const parts = encryptedText.split(':');
    if (parts.length !== 3) {
      return encryptedText;
    }
    
    const iv = Buffer.from(parts[0], 'hex');
    const authTag = Buffer.from(parts[1], 'hex');
    const encryptedData = parts[2];
    
    // Create decipher
    const decipher = crypto.createDecipheriv(ALGORITHM, keyBuffer, iv);
    decipher.setAuthTag(authTag);
    
    // Decrypt the data
    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  } catch (error) {
    console.error('Decryption error:', error);
    
    logSecurityEvent('DECRYPTION_ERROR', {
      error: error.message
    }, 'error');
    
    return null;
  }
};

module.exports = { encrypt, decrypt };
