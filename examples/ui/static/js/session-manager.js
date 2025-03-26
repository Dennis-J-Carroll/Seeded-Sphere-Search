/**
 * Session management module for storing and retrieving search sessions
 */

class SessionManager {
    constructor() {
        this.currentSession = null;
        this.sessionsList = [];
    }
    
    /**
     * Load all available sessions
     * @returns {Promise} Promise resolving to list of sessions
     */
    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            if (data.success) {
                this.sessionsList = data.sessions;
                return this.sessionsList;
            } else {
                throw new Error(data.error || 'Failed to load sessions');
            }
        } catch (error) {
            console.error('Error loading sessions:', error);
            throw error;
        }
    }
    
    /**
     * Create a new session
     * @param {string} name - Session name
     * @returns {Promise} Promise resolving to the new session
     */
    async createSession(name) {
        try {
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSession = data.session;
                // Add to sessions list
                if (!this.sessionsList.find(s => s.id === data.session.id)) {
                    this.sessionsList.push(data.session);
                }
                return data.session;
            } else {
                throw new Error(data.error || 'Failed to create session');
            }
        } catch (error) {
            console.error('Error creating session:', error);
            throw error;
        }
    }
    
    /**
     * Load a session by ID
     * @param {string} sessionId - Session ID
     * @returns {Promise} Promise resolving to the session
     */
    async loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentSession = data.session;
                return data.session;
            } else {
                throw new Error(data.error || 'Failed to load session');
            }
        } catch (error) {
            console.error(`Error loading session ${sessionId}:`, error);
            throw error;
        }
    }
    
    /**
     * Delete a session
     * @param {string} sessionId - Session ID
     * @returns {Promise} Promise resolving to success status
     */
    async deleteSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Remove from sessions list
                this.sessionsList = this.sessionsList.filter(s => s.id !== sessionId);
                
                // Reset current session if it was deleted
                if (this.currentSession && this.currentSession.id === sessionId) {
                    this.currentSession = null;
                }
                
                return true;
            } else {
                throw new Error(data.error || 'Failed to delete session');
            }
        } catch (error) {
            console.error(`Error deleting session ${sessionId}:`, error);
            throw error;
        }
    }
    
    /**
     * Export a session
     * @param {string} sessionId - Session ID
     */
    exportSession(sessionId) {
        // Create a download link for the session
        const exportUrl = `/api/sessions/${sessionId}/export`;
        
        // Create a temporary link element
        const downloadLink = document.createElement('a');
        downloadLink.href = exportUrl;
        downloadLink.download = `session-${sessionId}.json`;
        downloadLink.style.display = 'none';
        
        // Add to document, click, and remove
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }
    
    /**
     * Import a session from a file
     * @param {File} file - The JSON file to import
     * @returns {Promise} Promise resolving to the imported session
     */
    async importSession(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/sessions/import', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSession = data.session;
                
                // Add to sessions list
                if (!this.sessionsList.find(s => s.id === data.session.id)) {
                    this.sessionsList.push(data.session);
                }
                
                return data.session;
            } else {
                throw new Error(data.error || 'Failed to import session');
            }
        } catch (error) {
            console.error('Error importing session:', error);
            throw error;
        }
    }
    
    /**
     * Populate a select element with session options
     * @param {HTMLSelectElement} selectElement - The select element to populate
     * @param {function} onChangeHandler - Optional handler for select change events
     */
    populateSessionsSelect(selectElement, onChangeHandler = null) {
        // Clear existing options
        selectElement.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '-- Select a session --';
        selectElement.appendChild(defaultOption);
        
        // Add sessions
        this.sessionsList.forEach(session => {
            const option = document.createElement('option');
            option.value = session.id;
            option.textContent = `${session.name} (${new Date(session.created_at).toLocaleString()})`;
            selectElement.appendChild(option);
        });
        
        // Set current session if available
        if (this.currentSession) {
            selectElement.value = this.currentSession.id;
        }
        
        // Add change handler
        if (onChangeHandler) {
            selectElement.addEventListener('change', onChangeHandler);
        }
    }
    
    /**
     * Display search results from a session
     * @param {string} resultId - Result ID
     * @param {HTMLElement} container - Container to display results in
     * @returns {Promise} Promise resolving to the results
     */
    async displayResults(resultId, container) {
        if (!this.currentSession) {
            throw new Error('No session loaded');
        }
        
        try {
            const response = await fetch(`/api/sessions/${this.currentSession.id}/results/${resultId}`);
            const data = await response.json();
            
            if (data.success) {
                const results = data.results;
                
                // Clear container
                container.innerHTML = '';
                
                // Create results list
                if (results.results.length > 0) {
                    const resultsHtml = results.results.map((result, index) => {
                        const tags = result.tags && result.tags.length > 0 
                            ? `<div class="flex flex-wrap gap-1 mt-2">
                                ${result.tags.map(tag => 
                                    `<span class="bg-indigo-100 text-indigo-800 text-xs px-2 py-0.5 rounded">${tag}</span>`
                                ).join('')}
                               </div>`
                            : '';
                            
                        return `
                            <div class="result-card bg-white p-4 rounded-lg border border-gray-200 hover:border-indigo-300">
                                <div class="flex justify-between items-start">
                                    <h3 class="text-lg font-medium text-gray-900">${index + 1}. ${result.title}</h3>
                                    <span class="bg-indigo-100 text-indigo-800 text-xs px-2 py-0.5 rounded-full font-medium">
                                        ${result.similarity ? result.similarity.toFixed(3) : (result.score ? result.score.toFixed(3) : 'N/A')}
                                    </span>
                                </div>
                                <p class="mt-2 text-gray-600">${result.description || 'No description'}</p>
                                ${tags}
                            </div>
                        `;
                    }).join('');
                    
                    container.innerHTML = resultsHtml;
                } else {
                    container.innerHTML = `
                        <div class="text-gray-600 text-center py-8">
                            No results found in this session.
                        </div>
                    `;
                }
                
                return results;
            } else {
                throw new Error(data.error || 'Failed to load results');
            }
        } catch (error) {
            console.error(`Error displaying results ${resultId}:`, error);
            container.innerHTML = `
                <div class="text-red-600 text-center py-8">
                    Error: ${error.message}
                </div>
            `;
            throw error;
        }
    }
    
    /**
     * Display analytics data from a session
     * @param {string} analyticsId - Analytics ID
     * @param {object} chartManager - Chart manager instance
     * @param {HTMLElement} container - Container to display analytics in
     * @returns {Promise} Promise resolving to the analytics data
     */
    async displayAnalytics(analyticsId, chartManager, container) {
        if (!this.currentSession) {
            throw new Error('No session loaded');
        }
        
        try {
            const response = await fetch(`/api/sessions/${this.currentSession.id}/analytics/${analyticsId}`);
            const data = await response.json();
            
            if (data.success) {
                const analytics = data.analytics;
                
                // Clear container
                container.innerHTML = '';
                
                // Create analytics display
                const metricsHtml = `
                    <div class="bg-white p-4 rounded-lg shadow-md mb-4">
                        <h3 class="text-lg font-medium text-gray-900 mb-2">Metrics Summary</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <p class="text-sm text-gray-600">Average Precision</p>
                                <p class="text-xl font-semibold">${analytics.metrics.average_precision.toFixed(3)}</p>
                            </div>
                            ${Object.entries(analytics.metrics.precision_at_k).map(([k, v]) => `
                                <div>
                                    <p class="text-sm text-gray-600">${k}</p>
                                    <p class="text-xl font-semibold">${v.toFixed(3)}</p>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                
                // Create chart containers
                const chartsHtml = `
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <canvas id="pr-curve-chart" height="300"></canvas>
                        </div>
                        <div>
                            <canvas id="f1-threshold-chart" height="300"></canvas>
                        </div>
                        <div class="col-span-2">
                            <canvas id="precision-at-k-chart" height="200"></canvas>
                        </div>
                    </div>
                `;
                
                container.innerHTML = metricsHtml + chartsHtml;
                
                // Create charts
                chartManager.createPrecisionRecallCurve(
                    'pr-curve-chart', 
                    analytics.metrics.precision_recall_curve,
                    analytics.metrics.average_precision
                );
                
                chartManager.createF1ThresholdChart(
                    'f1-threshold-chart',
                    analytics.metrics.threshold_metrics
                );
                
                chartManager.createPrecisionAtKChart(
                    'precision-at-k-chart',
                    analytics.metrics.precision_at_k
                );
                
                return analytics;
            } else {
                throw new Error(data.error || 'Failed to load analytics');
            }
        } catch (error) {
            console.error(`Error displaying analytics ${analyticsId}:`, error);
            container.innerHTML = `
                <div class="text-red-600 text-center py-8">
                    Error: ${error.message}
                </div>
            `;
            throw error;
        }
    }
} 