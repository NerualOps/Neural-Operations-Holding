#!/usr/bin/env node
/**
 * Deploy Epsilon AI Model to Render
 * ---------------------------
 * This script:
 * 1. Exports the trained model from local Python service
 * 2. Uploads it to Supabase
 * 3. Render will auto-load it on next restart
 * 
 * Usage: node scripts/deploy-epsilon-model.js
 */

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const archiver = require('archiver');
require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

async function deployModel() {
  // Updated for local-only training workflow
  // Step 1: Load exported model from services/python-services/models/latest/
  const exportDir = path.join(__dirname, '..', 'services', 'python-services', 'models', 'latest');
  const modelPath = path.join(exportDir, 'model.pt');
  const configPath = path.join(exportDir, 'config.json');
  const metadataPath = path.join(exportDir, 'run_meta.json');
  
  if (!fs.existsSync(exportDir)) {
    throw new Error(`Export directory not found: ${exportDir}\nPlease export a model first using: ml_local/train/export.py`);
  }
  
  if (!fs.existsSync(modelPath)) {
    throw new Error(`Model file not found: ${modelPath}\nPlease export a model first using: ml_local/train/export.py`);
  }
  
  // Load model metadata
  let stats = {};
  let version = '1.0.0';
  
  if (fs.existsSync(metadataPath)) {
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    stats = metadata;
    version = metadata.version || metadata.git_commit || '1.0.0';
  } else if (fs.existsSync(configPath)) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    stats = { config };
  } else {
    throw new Error(`Metadata not found. Please ensure run_meta.json exists in ${exportDir}`);
  }
  
  try {
    // Silent - no console.log
    // Silent - no console.log
    // Silent - no console.log
    // Silent - no console.log

    // Step 2: Upload to Supabase
    // Silent - no console.log
    
    if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
      throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file');
    }

    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
      auth: { persistSession: false }
    });

    // Step 2: Create zip artifact containing all model files
    const modelId = stats.model_id || `run_${Date.now()}`;
    const zipFileName = `epsilon_artifact_${modelId}.zip`;
    const zipPath = path.join(exportDir, '..', zipFileName);
    
    // Create zip file
    const output = fs.createWriteStream(zipPath);
    const archive = archiver('zip', { zlib: { level: 9 } });
    
    return new Promise((resolve, reject) => {
      archive.on('error', reject);
      output.on('close', async () => {
        try {
          // Upload zip to Supabase Storage
          const zipBuffer = fs.readFileSync(zipPath);
          
          const { data: uploadData, error: uploadError } = await supabase.storage
            .from('epsilon-models')
            .upload(zipFileName, zipBuffer, {
              contentType: 'application/zip',
              upsert: false
            });
          
          if (uploadError) {
            throw new Error(`Failed to upload model artifact to storage: ${uploadError.message}`);
          }
          
          const storagePath = uploadData.path;
          
          // Clean up local zip
          fs.unlinkSync(zipPath);
          
          // Step 3: Create deployment record
          const { data, error } = await supabase
            .from('epsilon_model_deployments')
            .upsert({
              model_id: modelId,
              model_data: null, // Using storage_path instead
              storage_path: storagePath,
              stats: stats,
              temperature: stats.temperature || 0.9,
              version: version,
              status: 'pending', // Requires owner approval
              deployed_at: null, // Will be set when approved
              deployed_by: 'deployment-script'
            }, {
              onConflict: 'model_id'
            })
            .select();
          
          if (error) {
            const errorStr = error?.message || error?.toString() || '';
            const isHtmlError = errorStr.includes('<!DOCTYPE html>') || 
                               errorStr.includes('Cloudflare') || 
                               errorStr.includes('522') || 
                               errorStr.includes('521');
            if (isHtmlError) {
              throw new Error('Supabase connection timeout - please try again later');
            } else {
              throw new Error(`Supabase upload failed: ${error.message || 'Unknown error'}`);
            }
          }
          
          resolve();
        } catch (err) {
          reject(err);
        }
      });
      
      archive.pipe(output);
      
      // Add all required files to zip
      archive.file(modelPath, { name: 'model.pt' });
      archive.file(configPath, { name: 'config.json' });
      
      const tokenizerPath = path.join(exportDir, 'tokenizer.json');
      if (fs.existsSync(tokenizerPath)) {
        archive.file(tokenizerPath, { name: 'tokenizer.json' });
      }
      
      if (fs.existsSync(metadataPath)) {
        archive.file(metadataPath, { name: 'run_meta.json' });
      }
      
      const specialTokensPath = path.join(exportDir, 'special_tokens.json');
      if (fs.existsSync(specialTokensPath)) {
        archive.file(specialTokensPath, { name: 'special_tokens.json' });
      }
      
      archive.finalize();
    });

  } catch (error) {
    console.error('[DEPLOY] Deployment failed:', error.message);
    
    if (error.message.includes('Export directory not found') || error.message.includes('Model file not found')) {
      console.error('\n[DEPLOY] Model export not found:');
      console.error('   1. Train a model: cd ml_local && python train/pretrain.py ...');
      console.error('   2. Export the model: python train/export.py --checkpoint runs/checkpoint.pt --tokenizer tokenizer/tokenizer.json --output exports/latest');
      console.error('   3. Then run this deployment script again');
    }
    
    if (error.message.includes('Supabase')) {
      console.error('\n[DEPLOY] Supabase connection issue:');
      console.error('   - Check your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY');
      console.error('   - Verify the epsilon_model_deployments table exists in Supabase');
    }
    
    process.exit(1);
  }
}

async function checkTableExists() {
  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
      auth: { persistSession: false }
    });
    
    const { error } = await supabase
      .from('epsilon_model_deployments')
      .select('model_id')
      .limit(1);
    
    if (error && error.code === '42P01') {
      console.error('\n[DEPLOY] Table epsilon_model_deployments does not exist!');
      console.error('   Run this SQL in Supabase SQL Editor (see schema file for table definition)');
      // Silent - no console.log (SQL schema output removed)
      process.exit(1);
    }
  } catch (err) {
    // Table check failed, but continue - might be permissions issue
  }
}

// Run deployment
(async () => {
  await checkTableExists();
  await deployModel();
})();

