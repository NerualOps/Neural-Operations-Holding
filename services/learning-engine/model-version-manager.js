/**
 * Model Version Manager
 * =====================
 * Manages model versions, rollback, and A/B testing
 */

const { createClient } = require('@supabase/supabase-js');

class ModelVersionManager {
  constructor() {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_KEY) {
      throw new Error('SUPABASE_URL and SUPABASE_SERVICE_KEY must be set');
    }
    
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );
    
    this.currentVersion = null;
    this.versionHistory = [];
    this.abTestVersions = new Map();
  }

  /**
   * Get current production model version
   */
  async getCurrentVersion() {
    try {
      const { data, error } = await this.supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, deployed_at, created_by, storage_path')
        .eq('model_id', 'production')
        .eq('status', 'approved')
        .order('deployed_at', { ascending: false })
        .limit(1)
        .limit(1).maybeSingle();

      if (error && error.code !== 'PGRST116') {
        throw error;
      }

      if (data) {
        this.currentVersion = data;
        return data;
      }

      return null;
    } catch (error) {
      console.error('[MODEL VERSION] Error getting current version:', error);
      return null;
    }
  }

  /**
   * Create new model version
   */
  async createVersion(modelData, stats, metadata = {}) {
    try {
      const version = `v${Date.now()}`;
      const { data, error } = await this.supabase
        .from('epsilon_model_deployments')
        .insert({
          model_id: `version_${version}`,
          model_data: modelData,
          stats: {
            ...stats,
            ...metadata,
            created_at: new Date().toISOString(),
            version_type: 'incremental'
          },
          version: version,
          status: 'pending'
        })
        .select()
        .limit(1).maybeSingle();

      if (error) throw error;

      return data;
    } catch (error) {
      console.error('[MODEL VERSION] Error creating version:', error);
      throw error;
    }
  }

  /**
   * Rollback to previous version
   */
  async rollbackToVersion(versionId) {
    try {
      // Get the version to rollback to
      const { data: targetVersion, error: fetchError } = await this.supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, deployed_at, created_by, storage_path, model_data')
        .eq('id', versionId)
        .limit(1).maybeSingle();

      if (fetchError) throw fetchError;

      // Create new deployment with previous version's data
      const { data: rollbackVersion, error: createError } = await this.supabase
        .from('epsilon_model_deployments')
        .insert({
          model_id: 'production',
          model_data: targetVersion.model_data,
          storage_path: targetVersion.storage_path,
          stats: {
            ...(targetVersion.stats || {}),
            rollback_from: this.currentVersion?.id,
            rollback_at: new Date().toISOString(),
            original_version: targetVersion.version
          },
          version: `rollback_${targetVersion.version}`,
          status: 'approved',
          quality_score: targetVersion.quality_score
        })
        .select()
        .limit(1).maybeSingle();

      if (createError) throw createError;

      this.currentVersion = rollbackVersion;
      return rollbackVersion;
    } catch (error) {
      console.error('[MODEL VERSION] Error rolling back:', error);
      throw error;
    }
  }

  /**
   * Get version history
   */
  async getVersionHistory(limit = 20) {
    try {
      const { data, error } = await this.supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, deployed_at, created_by, storage_path')
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;

      this.versionHistory = data || [];
      return data;
    } catch (error) {
      console.error('[MODEL VERSION] Error getting version history:', error);
      return [];
    }
  }

  /**
   * Start A/B test with two model versions
   */
  async startABTest(versionAId, versionBId, splitRatio = 0.5) {
    try {
      const testId = `ab_test_${Date.now()}`;
      
      // Store A/B test configuration
      const { data, error } = await this.supabase
        .from('epsilon_model_deployments')
        .update({
          stats: {
            ab_test_id: testId,
            ab_test_split: splitRatio,
            ab_test_started_at: new Date().toISOString()
          }
        })
        .eq('id', versionAId)
        .select()
        .limit(1).maybeSingle();

      if (error) throw error;

      // Store B version
      await this.supabase
        .from('epsilon_model_deployments')
        .update({
          stats: {
            ab_test_id: testId,
            ab_test_split: 1 - splitRatio,
            ab_test_started_at: new Date().toISOString()
          }
        })
        .eq('id', versionBId);

      this.abTestVersions.set(testId, {
        versionA: versionAId,
        versionB: versionBId,
        splitRatio,
        startedAt: new Date().toISOString()
      });

      return { testId, versionA: versionAId, versionB: versionBId };
    } catch (error) {
      console.error('[MODEL VERSION] Error starting A/B test:', error);
      throw error;
    }
  }

  /**
   * Get model version for user (A/B test routing)
   */
  async getVersionForUser(userId) {
    try {
      const activeTests = Array.from(this.abTestVersions.entries());
      
      for (const [testId, config] of activeTests) {
        // Simple hash-based routing
        const userHash = this.hashUserId(userId);
        const useVersionA = (userHash % 100) < (config.splitRatio * 100);
        
        return {
          versionId: useVersionA ? config.versionA : config.versionB,
          testId,
          variant: useVersionA ? 'A' : 'B'
        };
      }

      const current = await this.getCurrentVersion();
      return {
        versionId: current?.id,
        testId: null,
        variant: 'production'
      };
    } catch (error) {
      console.error('[MODEL VERSION] Error getting version for user:', error);
      return { versionId: null, testId: null, variant: 'production' };
    }
  }

  /**
   * Hash user ID for consistent A/B test routing
   */
  hashUserId(userId) {
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}

module.exports = ModelVersionManager;

