// env-validator.js
const fs = require('fs');
const path = require('path');

// Validate environment variables
const validateEnv = () => {
  // Required environment variables
  const requiredVars = [
    'SUPABASE_URL',
    'SUPABASE_SERVICE_KEY',
    'JWT_SECRET',
    'FRONTEND_URL'
  ];
  
  // Optional environment variables with defaults
  const optionalVars = {
    'PORT': '10000',
    'ENCRYPTION_KEY': null,
    'LOG_LEVEL': 'info'
  };
  
  // Check for .env file and load it if not in production
  if (process.env.NODE_ENV !== 'production') {
    const projectEnvPath = path.join(process.cwd(), '.env');
    const localEnvPath = path.join(__dirname, '.env');

    if (fs.existsSync(projectEnvPath)) {
      // Silent - no console.log
      require('dotenv').config({ path: projectEnvPath });
    } else if (fs.existsSync(localEnvPath)) {
      // Silent - no console.log
      require('dotenv').config({ path: localEnvPath });
    } else {
      // Silent - no console.log
    }
  }
  
  // Set defaults for optional variables
  Object.entries(optionalVars).forEach(([key, defaultValue]) => {
    if (!process.env[key] && defaultValue !== null) {
      // Silent - no console.log
      process.env[key] = defaultValue;
    }
  });
  
  // Check required variables
  const missingVars = requiredVars.filter(varName => !process.env[varName]);
  
  if (missingVars.length > 0) {
    console.error('Missing required environment variables:');
    missingVars.forEach(varName => {
      console.error(`   - ${varName}`);
    });
    
    if (process.env.NODE_ENV === 'production') {
      console.error('Environment validation failed. Exiting.');
      process.exit(1);
    } else {
      console.warn('Environment validation failed but continuing in development mode.');
    }
    
    return false;
  }
  
  // All required variables are present
  return true;
};

module.exports = { validateEnv };
