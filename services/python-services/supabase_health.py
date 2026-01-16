"""
Supabase Health Check and Schema Validation Service
"""
import os
import aiohttp
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supabase Health Check")

SUPABASE_URL = os.getenv('SUPABASE_URL')
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "supabase-health-check",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/supabase-config")
async def supabase_config():
    """
    Test Supabase connection and return schema validation info
    Checks key tables and returns status
    """
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_SERVICE_KEY not configured"
        )
    
    headers = {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "supabase_url": SUPABASE_URL,
        "checks": {}
    }
    
    # Test tables that must exist
    test_tables = [
        'profiles',
        'epsilon_conversations',
        'epsilon_feedback',
        'knowledge_documents',
        'document_learning_sessions',
        'document_learning_insights',
        'epsilon_learning_patterns',
        'epsilon_model_weights',
        'epsilon_semantic_memory'
    ]
    
    async with aiohttp.ClientSession() as session:
        for table in test_tables:
            if not table or not isinstance(table, str):
                continue
            if not table.replace('_', '').replace('-', '').isalnum():
                continue
            if len(table) > 100:  # Prevent DoS
                continue
            
            try:
                url = f"{SUPABASE_URL}/rest/v1/{table}?limit=1"
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        results["checks"][table] = {
                            "status": "ok",
                            "exists": True,
                            "accessible": True
                        }
                    else:
                        try:
                            resp_text = await response.text()
                            is_html_error = '<!DOCTYPE html>' in resp_text or 'Cloudflare' in resp_text or '522' in resp_text or '521' in resp_text
                        except:
                            is_html_error = False
                        
                        if is_html_error:
                            results["checks"][table] = {
                                "status": "error",
                                "exists": None,
                                "accessible": False,
                                "error": "Supabase connection timeout"
                            }
                        else:
                            results["checks"][table] = {
                                "status": "error",
                                "exists": False,
                                "accessible": False,
                                "error_code": response.status
                            }
            except Exception as e:
                error_str = str(e)
                is_html_error = '<!DOCTYPE html>' in error_str or 'Cloudflare' in error_str or '522' in error_str or '521' in error_str
                results["checks"][table] = {
                    "status": "error",
                    "exists": None,
                    "accessible": False,
                    "error": "Supabase connection timeout" if is_html_error else str(e)
                }
    
    # Count total failures
    failed_checks = sum(1 for check in results["checks"].values() if not check.get("accessible", True))
    
    if failed_checks == 0:
        results["overall_status"] = "healthy"
    elif failed_checks < len(test_tables) / 2:
        results["overall_status"] = "degraded"
    else:
        results["overall_status"] = "critical"
        raise HTTPException(
            status_code=503,
            detail=f"Supabase health check failed: {failed_checks}/{len(test_tables)} tables inaccessible"
        )
    
    return results

@app.get("/supabase-config/detailed")
async def detailed_schema_check():
    """Detailed schema validation with table counts"""
    if not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_KEY not configured")
    
    headers = {
        'apikey': SUPABASE_SERVICE_KEY,
        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
        'Content-Type': 'application/json'
    }
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "supabase_url": SUPABASE_URL,
        "table_details": {}
    }
    
    tables_to_check = {
        'profiles': 'User profiles',
        'epsilon_conversations': 'Conversation logs',
        'epsilon_feedback': 'Feedback data',
        'knowledge_documents': 'Document storage',
        'document_learning_sessions': 'Learning sessions',
        'document_learning_insights': 'Learning insights',
        'epsilon_learning_patterns': 'Learning patterns',
        'epsilon_model_weights': 'Model weights',
        'epsilon_semantic_memory': 'Semantic memory (pgvector)'
    }
    
    async with aiohttp.ClientSession() as session:
        for table, description in tables_to_check.items():
            if not table or not isinstance(table, str):
                continue
            if not table.replace('_', '').replace('-', '').isalnum():
                continue
            if len(table) > 100:  # Prevent DoS
                continue
            if not description or not isinstance(description, str):
                description = 'Unknown'
            if len(description) > 200:  # Prevent DoS
                description = description[:200]
            
            try:
                url = f"{SUPABASE_URL}/rest/v1/{table}?select=id&limit=100"
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except Exception as json_error:
                            resp_text = await response.text()
                            if '<!DOCTYPE html>' in resp_text or 'Cloudflare' in resp_text or '522' in resp_text or '521' in resp_text:
                                results["table_details"][table] = {
                                    "description": description,
                                    "exists": None,
                                    "accessible": False,
                                    "error": "Supabase connection timeout"
                                }
                                continue
                            else:
                                raise json_error
                        
                        row_count = len(data) if isinstance(data, list) else 1
                        results["table_details"][table] = {
                            "description": description,
                            "exists": True,
                            "accessible": True,
                            "row_count": row_count if table != 'profiles' or row_count > 0 else "check manually"
                        }
                    else:
                        try:
                            resp_text = await response.text()
                            is_html_error = '<!DOCTYPE html>' in resp_text or 'Cloudflare' in resp_text or '522' in resp_text or '521' in resp_text
                        except:
                            is_html_error = False
                        
                        if is_html_error:
                            results["table_details"][table] = {
                                "description": description,
                                "exists": None,
                                "accessible": False,
                                "error": "Supabase connection timeout"
                            }
                        else:
                            results["table_details"][table] = {
                                "description": description,
                                "exists": False,
                                "accessible": False,
                                "error_code": response.status
                            }
            except Exception as e:
                error_str = str(e)
                is_html_error = '<!DOCTYPE html>' in error_str or 'Cloudflare' in error_str or '522' in error_str or '521' in error_str
                results["table_details"][table] = {
                    "description": description,
                    "exists": None,
                    "accessible": False,
                    "error": "Supabase connection timeout" if is_html_error else str(e)
                }
    
    return results

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Supabase Health Check Service on port 8005")
    uvicorn.run(app, host="0.0.0.0", port=8005)

