"""
Epsilon AI Metadata Query System
--------------------------
Fast query system for retrieving structured metadata from Supabase.
Allows Epsilon AI to instantly access learned concepts, facts, processes, and rules.
"""

import os
import aiohttp
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EpsilonMetadataQuery:
    """Fast metadata query system for instant knowledge retrieval"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY', '')
        if not self.supabase_key:
            logger.warning("SUPABASE_SERVICE_KEY not set - metadata queries may fail")
        self.cache: Dict[str, Any] = {}  # Simple in-memory cache
    
    def _is_html_error(self, text: str) -> bool:
        """Check if response text contains HTML error page (Supabase downtime)"""
        if not text:
            return False
        return '<!DOCTYPE html>' in text or 'Cloudflare' in text or '522' in text or '521' in text
    
    async def _handle_response(self, response, query_name: str):
        """Handle HTTP response with HTML error detection"""
        if response.status == 200:
            try:
                return await response.json()
            except Exception as json_error:
                resp_text = await response.text()
                if self._is_html_error(resp_text):
                    logger.warning(f"[METADATA] Supabase connection issue while {query_name}")
                else:
                    logger.error(f"Error parsing JSON response for {query_name}: {json_error}")
                return []
        else:
            resp_text = await response.text()
            if self._is_html_error(resp_text):
                logger.warning(f"[METADATA] Supabase connection issue while {query_name}")
            else:
                logger.warning(f"[METADATA] {query_name} failed with status {response.status}")
            return []
    
    async def query_concepts(self, concept_name: str) -> List[Dict]:
        """Query for a specific concept across all documents"""
        if not concept_name or not isinstance(concept_name, str):
            return []
        if len(concept_name) > 200:  # Prevent DoS
            concept_name = concept_name[:200]
        # Sanitize to prevent SQL injection (remove special characters)
        concept_name = ''.join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"concept:{concept_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                # Query knowledge_documents for extracted_metadata containing this concept
                # Include chunked flag for consistency (even though we only query metadata)
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'extracted_metadata->concepts->>': f'%{concept_name}%',
                    'limit': '10'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying concepts")
                    if not docs:
                        return []
                    
                    concepts = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        concepts_dict = metadata.get('concepts', {})
                        if concept_name.lower() in concepts_dict:
                            concepts.append({
                                'concept': concept_name,
                                'definition': concepts_dict.get(concept_name.lower(), {}).get('definition', ''),
                                'source': doc.get('title', ''),
                                'category': doc.get('learning_category', '')
                            })
                    
                    self.cache[cache_key] = concepts
                    return concepts
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying concepts")
            else:
                logger.error(f"Error querying concepts: {e}")
        
        return []
    
    async def query_facts(self, topic: str) -> List[Dict]:
        """Query for facts about a topic"""
        if not topic or not isinstance(topic, str):
            return []
        if len(topic) > 200:  # Prevent DoS
            topic = topic[:200]
        # Sanitize to prevent SQL injection (remove special characters)
        topic = ''.join(c for c in topic if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"facts:{topic}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                # Query for documents with facts about this topic
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'extracted_metadata->facts->>statement': f'%{topic}%',
                    'limit': '20'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying facts")
                    if not docs:
                        return []
                    
                    facts = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        facts_list = metadata.get('facts', [])
                        for fact in facts_list:
                            if topic.lower() in fact.get('statement', '').lower():
                                facts.append({
                                    'fact': fact.get('statement', ''),
                                    'type': fact.get('type', 'declarative'),
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = facts[:10]  # Limit to top 10
                    return facts[:10]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying facts")
            else:
                logger.error(f"Error querying facts: {e}")
        
        return []
    
    async def query_processes(self, process_name: str) -> List[Dict]:
        """Query for processes/steps"""
        if not process_name or not isinstance(process_name, str):
            return []
        if len(process_name) > 200:  # Prevent DoS
            process_name = process_name[:200]
        # Sanitize to prevent SQL injection (remove special characters)
        process_name = ''.join(c for c in process_name if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"process:{process_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'extracted_metadata->processes->>steps': f'%{process_name}%',
                    'limit': '5'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying processes")
                    if not docs:
                        return []
                    
                    processes = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        processes_list = metadata.get('processes', [])
                        for process in processes_list:
                            steps = process.get('steps', [])
                            # Check if any step mentions the process name
                            if any(process_name.lower() in str(step).lower() for step in steps):
                                processes.append({
                                    'process': process,
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = processes[:5]
                    return processes[:5]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying processes")
            else:
                logger.error(f"Error querying processes: {e}")
        
        return []
    
    async def query_rules(self, rule_type: str = None) -> List[Dict]:
        """Query for rules and guidelines"""
        if rule_type is not None:
            if not isinstance(rule_type, str):
                rule_type = None
            elif len(rule_type) > 50:  # Prevent DoS
                rule_type = rule_type[:50]
            # Sanitize to prevent SQL injection
            rule_type = ''.join(c for c in rule_type if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"rules:{rule_type or 'all'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'extracted_metadata->rules->>rule': 'is.not.null',
                    'limit': '30'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying rules")
                    if not docs:
                        return []
                    
                    rules = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        rules_list = metadata.get('rules', [])
                        for rule in rules_list:
                            if not rule_type or rule.get('type') == rule_type:
                                rules.append({
                                    'rule': rule.get('rule', ''),
                                    'type': rule.get('type', 'rule'),
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = rules[:30]
                    return rules[:30]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying rules")
            else:
                logger.error(f"Error querying rules: {e}")
        
        return []
    
    async def query_examples(self, example_type: str = None) -> List[Dict]:
        """Query for examples and case studies"""
        if example_type is not None:
            if not isinstance(example_type, str):
                example_type = None
            elif len(example_type) > 50:  # Prevent DoS
                example_type = example_type[:50]
            # Sanitize to prevent SQL injection
            example_type = ''.join(c for c in example_type if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"examples:{example_type or 'all'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'extracted_metadata->examples->>example': 'is.not.null',
                    'limit': '20'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying examples")
                    if not docs:
                        return []
                    
                    examples = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        examples_list = metadata.get('examples', [])
                        for example in examples_list:
                            if not example_type or example.get('type') == example_type:
                                examples.append({
                                    'example': example.get('example', ''),
                                    'type': example.get('type', 'example'),
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = examples[:20]
                    return examples[:20]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying examples")
            else:
                logger.error(f"Error querying examples: {e}")
        
        return []
    
    async def query_business_insights(self, topic: str = None) -> List[Dict]:
        """Query for business insights"""
        if topic is not None:
            if not isinstance(topic, str):
                topic = None
            elif len(topic) > 200:  # Prevent DoS
                topic = topic[:200]
            # Sanitize to prevent SQL injection
            topic = ''.join(c for c in topic if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"insights:{topic or 'all'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category,is_chunked',
                    'learning_category': 'eq.knowledge',
                    'extracted_metadata->business_insights->>insight': 'is.not.null',
                    'limit': '15'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying business insights")
                    if not docs:
                        return []
                    
                    insights = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        insights_list = metadata.get('business_insights', [])
                        for insight in insights_list:
                            insight_text = insight.get('insight', '')
                            if not topic or topic.lower() in insight_text.lower():
                                insights.append({
                                    'insight': insight_text,
                                    'type': insight.get('type', 'insight'),
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = insights[:15]
                    return insights[:15]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying business insights")
            else:
                logger.error(f"Error querying business insights: {e}")
        
        return []
    
    async def query_sales_insights(self, topic: str = None) -> List[Dict]:
        """Query for sales insights"""
        if topic is not None:
            if not isinstance(topic, str):
                topic = None
            elif len(topic) > 200:  # Prevent DoS
                topic = topic[:200]
            # Sanitize to prevent SQL injection
            topic = ''.join(c for c in topic if c.isalnum() or c in (' ', '-', '_'))
        
        cache_key = f"sales_insights:{topic or 'all'}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.supabase_url}/rest/v1/knowledge_documents"
                params = {
                    'select': 'id,title,extracted_metadata,learning_category',
                    'learning_category': 'eq.sales_training',
                    'extracted_metadata->sales_insights->>insight': 'is.not.null',
                    'limit': '15'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    docs = await self._handle_response(response, "querying sales insights")
                    if not docs:
                        return []
                    
                    insights = []
                    for doc in docs:
                        metadata = doc.get('extracted_metadata', {})
                        insights_list = metadata.get('sales_insights', [])
                        for insight in insights_list:
                            insight_text = insight.get('insight', '')
                            if not topic or topic.lower() in insight_text.lower():
                                insights.append({
                                    'insight': insight_text,
                                    'type': insight.get('type', 'sales_technique'),
                                    'source': doc.get('title', ''),
                                    'category': doc.get('learning_category', '')
                                })
                    
                    self.cache[cache_key] = insights[:15]
                    return insights[:15]
        except Exception as e:
            error_str = str(e)
            if self._is_html_error(error_str):
                logger.warning(f"[METADATA] Supabase connection issue while querying sales insights")
            else:
                logger.error(f"Error querying sales insights: {e}")
        
        return []
    
    def clear_cache(self):
        """Clear the metadata cache"""
        self.cache = {}

