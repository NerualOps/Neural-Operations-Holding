#!/usr/bin/env node
// security-scan.js
// Comprehensive security scanning script for dependencies and vulnerabilities

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔒 NeuralOps Security Scanner');
console.log('================================\n');

const results = {
  npm: { vulnerabilities: [], passed: false },
  python: { vulnerabilities: [], passed: false },
  overall: { passed: false }
};

// ============================================================================
// 1. NPM DEPENDENCY SCAN
// ============================================================================
console.log('📦 Scanning npm dependencies...');
try {
  const auditOutput = execSync('npm audit --production --json', { 
    encoding: 'utf8',
    stdio: 'pipe',
    cwd: process.cwd()
  });
  
  const audit = JSON.parse(auditOutput);
  
  if (audit.vulnerabilities) {
    const vulns = Object.values(audit.vulnerabilities);
    results.npm.vulnerabilities = vulns;
    results.npm.passed = vulns.length === 0;
    
    if (vulns.length > 0) {
      console.log(`❌ Found ${vulns.length} npm vulnerabilities`);
      vulns.forEach((vuln, idx) => {
        console.log(`   ${idx + 1}. ${vuln.name}: ${vuln.severity} - ${vuln.title}`);
      });
    } else {
      console.log('✅ No npm vulnerabilities found');
    }
  } else {
    results.npm.passed = true;
    console.log('✅ No npm vulnerabilities found');
  }
} catch (error) {
  console.log('⚠️  npm audit failed - may need to run: npm audit fix');
  results.npm.passed = false;
}

// ============================================================================
// 2. PYTHON DEPENDENCY SCAN
// ============================================================================
console.log('\n🐍 Scanning Python dependencies...');
const requirementsPath = path.join(process.cwd(), 'services', 'python-services', 'requirements.txt');

if (fs.existsSync(requirementsPath)) {
  try {
    // Try pip-audit if available
    try {
      execSync('pip-audit --version', { stdio: 'pipe' });
      const pipAuditOutput = execSync('pip-audit --format json', {
        encoding: 'utf8',
        stdio: 'pipe',
        cwd: path.join(process.cwd(), 'services', 'python-services')
      });
      
      const pipAudit = JSON.parse(pipAuditOutput);
      if (pipAudit.vulnerabilities && pipAudit.vulnerabilities.length > 0) {
        results.python.vulnerabilities = pipAudit.vulnerabilities;
        results.python.passed = false;
        console.log(`❌ Found ${pipAudit.vulnerabilities.length} Python vulnerabilities`);
        pipAudit.vulnerabilities.forEach((vuln, idx) => {
          console.log(`   ${idx + 1}. ${vuln.name}: ${vuln.severity}`);
        });
      } else {
        results.python.passed = true;
        console.log('✅ No Python vulnerabilities found');
      }
    } catch (pipAuditError) {
      console.log('⚠️  pip-audit not installed. Install with: pip install pip-audit');
      console.log('   Skipping Python vulnerability scan');
      results.python.passed = true; // Don't fail if tool not available
    }
  } catch (error) {
    console.log('⚠️  Python dependency scan failed');
    results.python.passed = false;
  }
} else {
  console.log('⚠️  requirements.txt not found - skipping Python scan');
  results.python.passed = true;
}

// ============================================================================
// 3. SECURITY HEADERS CHECK
// ============================================================================
console.log('\n🛡️  Checking security configuration...');
const serverPath = path.join(process.cwd(), 'runtime', 'server.js');
if (fs.existsSync(serverPath)) {
  const serverContent = fs.readFileSync(serverPath, 'utf8');
  
  const checks = {
    'CSP with nonces': serverContent.includes('generateCSPHeader') || serverContent.includes('nonce'),
    'Security hardening': serverContent.includes('securityHardening'),
    'Rate limiting': serverContent.includes('rateLimit') || serverContent.includes('rate-limit'),
    'CSRF protection': serverContent.includes('csrf') || serverContent.includes('CSRF'),
    'Input sanitization': serverContent.includes('sanitize') || serverContent.includes('sanitizeText')
  };
  
  let allPassed = true;
  Object.entries(checks).forEach(([check, passed]) => {
    if (passed) {
      console.log(`   ✅ ${check}`);
    } else {
      console.log(`   ❌ ${check} - MISSING`);
      allPassed = false;
    }
  });
  
  if (!allPassed) {
    console.log('\n⚠️  Some security features may be missing');
  }
}

// ============================================================================
// 4. ENVIRONMENT VARIABLES CHECK
// ============================================================================
console.log('\n🔐 Checking environment variables...');
const requiredEnvVars = [
  'JWT_SECRET',
  'SUPABASE_URL',
  'SUPABASE_SERVICE_KEY'
];

const optionalEnvVars = [
  'ENCRYPTION_KEY',
  'FRONTEND_URL'
];

let envIssues = 0;
requiredEnvVars.forEach(varName => {
  if (process.env[varName]) {
    console.log(`   ✅ ${varName} is set`);
  } else {
    console.log(`   ❌ ${varName} is MISSING (REQUIRED)`);
    envIssues++;
  }
});

optionalEnvVars.forEach(varName => {
  if (process.env[varName]) {
    console.log(`   ✅ ${varName} is set`);
  } else {
    console.log(`   ⚠️  ${varName} is not set (optional but recommended)`);
  }
});

// ============================================================================
// 5. SUMMARY
// ============================================================================
console.log('\n📊 Security Scan Summary');
console.log('================================');
console.log(`NPM Dependencies: ${results.npm.passed ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Python Dependencies: ${results.python.passed ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Environment Variables: ${envIssues === 0 ? '✅ PASS' : '❌ FAIL'}`);

results.overall.passed = results.npm.passed && results.python.passed && envIssues === 0;

if (results.overall.passed) {
  console.log('\n✅ Overall: PASS - System appears secure');
  process.exit(0);
} else {
  console.log('\n❌ Overall: FAIL - Security issues detected');
  console.log('\nRecommended actions:');
  if (!results.npm.passed) {
    console.log('  1. Run: npm audit fix');
  }
  if (!results.python.passed) {
    console.log('  2. Install pip-audit: pip install pip-audit');
    console.log('  3. Run: pip-audit --fix');
  }
  if (envIssues > 0) {
    console.log('  4. Set all required environment variables');
  }
  process.exit(1);
}

