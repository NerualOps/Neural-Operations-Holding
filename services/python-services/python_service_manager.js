/**
 * Epsilon AI - Python Service Manager
 * © 2025 Neural Operation's & Holding's LLC. All rights reserved.
 */

const { spawn } = require('child_process');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

class PythonServiceManager {
    constructor() {
        this.services = {
            nlp: { port: 8001, process: null, ready: false },
            analytics: { port: 8002, process: null, ready: false },
            content: { port: 8003, process: null, ready: false },
            document_learning: { port: 8004, process: null, ready: false },
            language_model: { port: 8005, process: null, ready: false },
            learning_service: { port: 0, process: null, ready: false }
        };
        this.initialized = false;
    }

    async initialize() {
        console.log('[PYTHON MANAGER] Starting Python services initialization...');
        console.log('[PYTHON MANAGER] Environment:', process.env.NODE_ENV || 'development');
        
        try {
            const isProduction = process.env.NODE_ENV === 'production';
            
            // In production, only start inference service (and optionally document extraction)
            if (isProduction) {
                console.log('[PYTHON MANAGER] Production mode: Starting inference service only...');
                await this.startLanguageModelService();
                
                // Optionally start document extraction if needed
                if (process.env.ENABLE_DOCUMENT_EXTRACTION === 'true') {
                    await this.startDocumentLearningService();
                }
            } else {
                // Development: start all services
                console.log('[PYTHON MANAGER] Development mode: Starting all services...');
                await this.startNLPService();
                await this.startLanguageModelService();
                
                setTimeout(async () => {
                    try {
                        await this.startAnalyticsService();
                        await this.startContentService();
                        await this.startDocumentLearningService();
                        await this.startLearningService();
                    } catch (error) {
                        console.error('[PYTHON MANAGER] Deferred services failed to start:', error.message);
                        throw error;
                    }
                }, 10000);
            }
            
            console.log('[PYTHON MANAGER] Waiting for critical services to be ready...');
            
            let timeoutId = null;
            let timeoutFired = false;
            let servicesBecameReady = false;
            
            const timeoutPromise = new Promise((resolve, reject) => {
                timeoutId = setTimeout(() => {
                    timeoutFired = true;
                    if (!servicesBecameReady) {
                        reject(new Error('Timeout waiting for Python services to initialize'));
                    } else {
                        resolve(false);
                    }
                }, 90000);
            });
            
            // Wait for services based on environment
            const requiredServices = isProduction 
                ? ['language_model'] 
                : ['nlp', 'language_model'];
            
            const servicesReady = await Promise.race([
                this.waitForServices(requiredServices).then((ready) => {
                    servicesBecameReady = ready;
                    if (timeoutId && !timeoutFired) {
                        clearTimeout(timeoutId);
                    }
                    return ready;
                }),
                timeoutPromise
            ]);
            
            if (!servicesReady) {
                throw new Error('Critical Python services failed to initialize');
            }
            
            this.initialized = servicesReady;
            console.log('[PYTHON MANAGER] All critical Python services are ready');
            return servicesReady;
        } catch (error) {
            console.error('[PYTHON MANAGER] Failed to initialize Python services:', error);
            throw error;
        }
    }

    resolvePythonPath() {
        if (process.env.PYTHON_EXECUTABLE && fs.existsSync(process.env.PYTHON_EXECUTABLE)) {
            return process.env.PYTHON_EXECUTABLE;
        }

        if (process.platform === 'win32') {
            const venvPython = path.join(process.cwd(), '.venv', 'Scripts', 'python.exe');
            if (fs.existsSync(venvPython)) {
                return venvPython;
            }
            const launcher = 'py';
            return launcher;
        }

        return process.env.PYTHON || 'python3';
    }

    async startNLPService() {
        return new Promise(async (resolve, reject) => {
            try {
                const response = await axios.get(`http://localhost:${this.services.nlp.port}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    console.log('[PYTHON MANAGER] NLP service already running on port', this.services.nlp.port);
                    this.services.nlp.ready = true;
                    resolve();
                    return;
                }
            } catch (error) {
            }
            
            const pythonPath = this.resolvePythonPath();
            const pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'nlp_processor:app', '--host', '0.0.0.0', '--port', '8001'], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env, PORT: '8001' }
            });

            this.services.nlp.process = pythonProcess;

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                if (!output.includes('INFO:') && !output.includes('Uvicorn running') && output.length > 0) {
                    console.error(`[NLP SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.nlp.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`NLP service exited with code ${code}`));
                }
            });

            setTimeout(async () => {
                try {
                    const response = await axios.get(`http://localhost:${this.services.nlp.port}/health`, { timeout: 2000 });
                    if (response.status === 200) {
                        console.log('[PYTHON MANAGER] NLP service is ready on port', this.services.nlp.port);
                        this.services.nlp.ready = true;
                        resolve();
                    } else {
                        reject(new Error('NLP service health check failed'));
                    }
                } catch (error) {
                    if (pythonProcess.exitCode === null) {
                        this.services.nlp.ready = true;
                        resolve();
                    } else {
                        reject(new Error(`NLP service failed to start: ${error.message}`));
                    }
                }
            }, 5000);
        });
    }

    async startAnalyticsService() {
        return new Promise(async (resolve, reject) => {
            try {
                const response = await axios.get(`http://localhost:${this.services.analytics.port}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    this.services.analytics.ready = true;
                    resolve();
                    return;
                }
            } catch (error) {
            }
            
            const pythonPath = this.resolvePythonPath();
            const pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'learning_analytics:app', '--host', '0.0.0.0', '--port', '8002'], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env, PORT: '8002' }
            });

            this.services.analytics.process = pythonProcess;

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                if (!output.includes('INFO:') && !output.includes('Uvicorn running') && output.length > 0) {
                    console.error(`[ANALYTICS SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.analytics.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`Analytics service exited with code ${code}`));
                }
            });

            setTimeout(async () => {
                try {
                    const response = await axios.get(`http://localhost:${this.services.analytics.port}/health`, { timeout: 2000 });
                    if (response.status === 200) {
                        this.services.analytics.ready = true;
                        resolve();
                    } else {
                        reject(new Error('Analytics service health check failed'));
                    }
                } catch (error) {
                    if (pythonProcess.exitCode === null) {
                        this.services.analytics.ready = true;
                        resolve();
                    } else {
                        reject(new Error(`Analytics service failed to start: ${error.message}`));
                    }
                }
            }, 5000);
        });
    }

    async startContentService() {
        return new Promise(async (resolve, reject) => {
            try {
                const response = await axios.get(`http://localhost:${this.services.content.port}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    this.services.content.ready = true;
                    resolve();
                    return;
                }
            } catch (error) {
            }
            
            const pythonPath = this.resolvePythonPath();
            const pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'content_generator:app', '--host', '0.0.0.0', '--port', '8003'], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env, PORT: '8003' }
            });

            this.services.content.process = pythonProcess;

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                if (!output.includes('INFO:') && !output.includes('Uvicorn running') && output.length > 0) {
                    console.error(`[CONTENT SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.content.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`Content service exited with code ${code}`));
                }
            });

            setTimeout(async () => {
                try {
                    const response = await axios.get(`http://localhost:${this.services.content.port}/health`, { timeout: 2000 });
                    if (response.status === 200) {
                        this.services.content.ready = true;
                        resolve();
                    } else {
                        reject(new Error('Content service health check failed'));
                    }
                } catch (error) {
                    if (pythonProcess.exitCode === null) {
                        this.services.content.ready = true;
                        resolve();
                    } else {
                        reject(new Error(`Content service failed to start: ${error.message}`));
                    }
                }
            }, 5000);
        });
    }

    async startDocumentLearningService() {
        return new Promise(async (resolve, reject) => {
            try {
                const response = await axios.get(`http://localhost:${this.services.document_learning.port}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    this.services.document_learning.ready = true;
                    resolve();
                    return;
                }
            } catch (error) {
            }
            
            const pythonPath = this.resolvePythonPath();
            const pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'document_learning:app', '--host', '0.0.0.0', '--port', '8004'], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { 
                    ...process.env, 
                    PORT: '8004',
                    PYTHONPATH: __dirname,
                    PYTHONUNBUFFERED: '1'
                }
            });

            this.services.document_learning.process = pythonProcess;

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                const isInfoLog = output.includes('INFO:') || output.includes('Uvicorn running') || output.includes('__main__ - INFO');
                if (!isInfoLog && output.length > 0 && !output.startsWith('2025-')) {
                    console.error(`[DOCUMENT LEARNING SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.document_learning.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`Document learning service exited with code ${code}`));
                }
            });

            setTimeout(async () => {
                try {
                    const response = await axios.get(`http://localhost:${this.services.document_learning.port}/health`, { timeout: 2000 });
                    if (response.status === 200) {
                        this.services.document_learning.ready = true;
                        resolve();
                    } else {
                        reject(new Error('Document learning service health check failed'));
                    }
                } catch (error) {
                    if (pythonProcess.exitCode === null) {
                        this.services.document_learning.ready = true;
                        resolve();
                    } else {
                        reject(new Error(`Document learning service failed to start: ${error.message}`));
                    }
                }
            }, 5000);
        });
    }

    async startLanguageModelService() {
        return new Promise(async (resolve, reject) => {
            try {
                const response = await axios.get(`http://localhost:${this.services.language_model.port}/health`, { timeout: 1000 });
                if (response.status === 200) {
                    console.log('[PYTHON MANAGER] Language model service already running on port', this.services.language_model.port);
                    this.services.language_model.ready = true;
                    resolve();
                    return;
                }
            } catch (error) {
            }

            const pythonPath = this.resolvePythonPath();
            const modelDir = path.join(__dirname, 'models', 'latest');
            const pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'inference_service:app', '--host', '0.0.0.0', '--port', '8005'], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: {
                    ...process.env,
                    PORT: '8005',
                    PYTHONPATH: __dirname,
                    PYTHONUNBUFFERED: '1',
                    EPSILON_MODEL_DIR: modelDir
                }
            });

            this.services.language_model.process = pythonProcess;

            // Capture stdout to see Python service logs
            pythonProcess.stdout.on('data', (data) => {
                const output = data.toString().trim();
                if (output.length > 0) {
                    console.log(`[PYTHON LANGUAGE MODEL SERVICE] ${output}`);
                }
            });

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                const isInfoLog = output.includes('INFO:') || output.includes('Uvicorn running') || output.includes('__main__ - INFO');
                if (!isInfoLog && output.length > 0) {
                    console.error(`[LANGUAGE MODEL SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.language_model.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`Language model service exited with code ${code}`));
                }
            });

            let retries = 0;
            const maxRetries = 10;
            const checkInterval = 1000;
            
            const checkReady = async () => {
                try {
                    const response = await axios.get(`http://localhost:${this.services.language_model.port}/health`, { timeout: 2000 });
                    if (response.status === 200) {
                        console.log('[PYTHON MANAGER] Language model service is ready on port', this.services.language_model.port);
                        this.services.language_model.ready = true;
                        resolve();
                        return;
                    }
                } catch (error) {
                    retries++;
                    if (pythonProcess.exitCode !== null) {
                        reject(new Error(`Language model service failed to start: ${error.message}`));
                        return;
                    }
                    
                    if (retries < maxRetries) {
                        setTimeout(checkReady, checkInterval);
                    } else {
                        if (pythonProcess.exitCode === null) {
                            this.services.language_model.ready = true;
                            resolve();
                        } else {
                            reject(new Error('Language model service failed to start after max retries'));
                        }
                    }
                }
            };
            
            setTimeout(checkReady, 3000);
        });
    }

    async startLearningService() {
        return new Promise((resolve, reject) => {
            const pythonPath = this.resolvePythonPath();
            const servicePath = path.join(__dirname, 'epsilon_learning_service.py');
            
            const pythonProcess = spawn(pythonPath, [servicePath], {
                cwd: __dirname,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            this.services.learning_service.process = pythonProcess;

            pythonProcess.stderr.on('data', (data) => {
                const output = data.toString().trim();
                const isInfoLog = output.includes('INFO:') || output.includes('__main__ - INFO');
                if (!isInfoLog && output.length > 0 && !output.startsWith('2025-')) {
                    console.error(`[LEARNING SERVICE ERROR] ${output}`);
                }
            });

            pythonProcess.on('close', (code) => {
                this.services.learning_service.ready = false;
                if (code !== 0 && code !== null) {
                    reject(new Error(`Learning service exited with code ${code}`));
                }
            });

            setTimeout(() => {
                if (pythonProcess.exitCode === null) {
                    this.services.learning_service.ready = true;
                    resolve();
                } else {
                    reject(new Error('Learning service failed to start'));
                }
            }, 3000);
        });
    }

    async waitForServices(serviceNames = null) {
        const servicesToWait = serviceNames || Object.keys(this.services);
        const maxWaitTime = 90000;
        const checkInterval = 1000;
        let waitTime = 0;

        while (waitTime < maxWaitTime) {
            const servicesToWaitFor = servicesToWait
                .map(name => this.services[name])
                .filter(service => service && service.port > 0);
            
            const allReady = servicesToWaitFor.length > 0 && servicesToWaitFor.every(service => service.ready);
            
            if (allReady) {
                return true;
            }
            
            await new Promise(resolve => setTimeout(resolve, checkInterval));
            waitTime += checkInterval;
        }

        const servicesToWaitFor = servicesToWait
            .map(name => this.services[name])
            .filter(service => service && service.port > 0);
        
        const allReady = servicesToWaitFor.length > 0 && servicesToWaitFor.every(service => service.ready);
        
        if (allReady) {
            return true;
        }

        const notReadyServices = Object.entries(this.services)
            .filter(([name, service]) => servicesToWait.includes(name) && !service.ready && service.port > 0)
            .map(([name]) => name);
        
        if (notReadyServices.length > 0) {
            throw new Error(`Services failed to become ready: ${notReadyServices.join(', ')}`);
        }
        
        return false;
    }

    async analyzeText(text, analysisType = 'full') {
        if (!text || typeof text !== 'string') {
            throw new Error('text must be a non-empty string');
        }
        if (text.length > 100000) {
            text = text.substring(0, 100000);
        }
        if (!analysisType || typeof analysisType !== 'string') {
            analysisType = 'full';
        }
        
        if (!this.services.nlp.ready) {
            throw new Error('NLP service not ready');
        }

        const response = await axios.post(`http://localhost:${this.services.nlp.port}/analyze`, {
            text: text,
            analysis_type: analysisType
        }, {
            timeout: 10000
        });

        return response.data;
    }

    async generateResponse(userMessage, context = {}) {
        if (!userMessage || typeof userMessage !== 'string') {
            throw new Error('userMessage must be a non-empty string');
        }
        if (userMessage.length > 100000) {
            userMessage = userMessage.substring(0, 100000);
        }
        if (!context || typeof context !== 'object' || Array.isArray(context)) {
            context = {};
        }
        
        if (!this.services.content.ready) {
            throw new Error('Content service not ready');
        }

        const response = await axios.post(`http://localhost:${this.services.content.port}/generate-response`, {
            user_message: userMessage,
            context: context
        }, {
            timeout: 10000
        });

        return response.data;
    }

    async analyzeConversation(conversationData) {
        if (!conversationData || typeof conversationData !== 'object' || Array.isArray(conversationData)) {
            throw new Error('conversationData must be a non-empty object');
        }
        
        if (!this.services.analytics.ready) {
            throw new Error('Analytics service not ready');
        }

        const response = await axios.post(`http://localhost:${this.services.analytics.port}/analyze-conversation`, {
            conversation_data: conversationData
        }, {
            timeout: 10000
        });

        return response.data;
    }

    async generateInsights(timePeriod = '7d') {
        if (!timePeriod || typeof timePeriod !== 'string') {
            timePeriod = '7d';
        }
        if (timePeriod.length > 50) {
            timePeriod = timePeriod.substring(0, 50);
        }
        
        if (!this.services.analytics.ready) {
            throw new Error('Analytics service not ready');
        }

        const response = await axios.post(`http://localhost:${this.services.analytics.port}/generate-insights`, {
            time_period: timePeriod
        }, {
            timeout: 10000
        });

        return response.data;
    }

    async optimizeResponse(userProfile, conversationContext) {
        if (!userProfile || typeof userProfile !== 'object' || Array.isArray(userProfile)) {
            userProfile = {};
        }
        if (!conversationContext || typeof conversationContext !== 'object' || Array.isArray(conversationContext)) {
            conversationContext = {};
        }
        
        if (!this.services.analytics.ready) {
            throw new Error('Analytics service not ready');
        }

        const response = await axios.post(`http://localhost:${this.services.analytics.port}/optimize-response`, {
            user_profile: userProfile,
            conversation_context: conversationContext
        }, {
            timeout: 10000
        });

        return response.data;
    }

    async healthCheck() {
        const health = {};
        
        for (const [serviceName, service] of Object.entries(this.services)) {
            try {
                if (service.ready) {
                    const response = await axios.get(`http://localhost:${service.port}/health`, {
                        timeout: 5000
                    });
                    health[serviceName] = {
                        status: 'healthy',
                        ready: true,
                        response: response.data
                    };
                } else {
                    health[serviceName] = {
                        status: 'not_ready',
                        ready: false
                    };
                }
            } catch (error) {
                health[serviceName] = {
                    status: 'error',
                    ready: false,
                    error: error.message
                };
            }
        }

        return health;
    }

    async shutdown() {
        for (const [serviceName, service] of Object.entries(this.services)) {
            if (service.process) {
                service.process.kill();
                service.ready = false;
            }
        }

        this.initialized = false;
    }

    isReady() {
        return this.initialized && Object.values(this.services).every(service => service.ready);
    }

    isServiceReady(serviceName) {
        if (!this.services[serviceName]) {
            return false;
        }
        return this.services[serviceName].ready === true;
    }

    getServiceStatus() {
        return {
            initialized: this.initialized,
            services: Object.fromEntries(
                Object.entries(this.services).map(([name, service]) => [
                    name,
                    {
                        ready: service.ready,
                        port: service.port,
                        process_running: service.process !== null
                    }
                ])
            )
        };
    }
}

module.exports = PythonServiceManager;
