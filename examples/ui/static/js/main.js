/**
 * Main JavaScript file for the Seeded Sphere Search Evaluation Platform
 * 
 * This file coordinates all the UI functionality, including:
 * - Tab navigation
 * - Search functionality
 * - Analytics processing
 * - Algorithm comparison
 * - Embedding visualization
 * - Session management
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    const chartManager = new ChartManager();
    const sessionManager = new SessionManager();
    
    // Initialize tabs
    initTabs();
    
    // Initialize UI components
    initSearchUI();
    initAnalyticsUI(chartManager, sessionManager);
    initComparisonUI(chartManager, sessionManager);
    initVisualizationUI(sessionManager);
    initSessionUI(sessionManager, chartManager);
    
    // Load sessions
    sessionManager.loadSessions().then(sessions => {
        // Populate session selects
        const sessionSelects = document.querySelectorAll('#session-select');
        sessionSelects.forEach(select => {
            sessionManager.populateSessionsSelect(select);
        });
        
        // Refresh sessions list in UI
        refreshSessionsList();
    }).catch(error => {
        console.error('Error loading sessions:', error);
    });
    
    /**
     * Initialize tab navigation
     */
    function initTabs() {
        const tabButtons = document.querySelectorAll('[data-tab]');
        const tabContents = document.querySelectorAll('[data-tab-content]');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.getAttribute('data-tab');
                switchTab(tabId);
            });
        });
        
        // Default to first tab
        switchTab('search');
    }
    
    /**
     * Switch to a specific tab
     * @param {string} tabId - ID of the tab to switch to
     */
    function switchTab(tabId) {
        // Update tab buttons
        const tabButtons = document.querySelectorAll('[data-tab]');
        tabButtons.forEach(button => {
            const buttonTabId = button.getAttribute('data-tab');
            if (buttonTabId === tabId) {
                button.classList.add('border-indigo-600', 'bg-indigo-600', 'text-white');
                button.classList.remove('border-transparent', 'bg-white', 'text-gray-700');
            } else {
                button.classList.remove('border-indigo-600', 'bg-indigo-600', 'text-white');
                button.classList.add('border-transparent', 'bg-white', 'text-gray-700');
            }
        });
        
        // Update tab contents
        const tabContents = document.querySelectorAll('[data-tab-content]');
        tabContents.forEach(content => {
            const contentTabId = content.getAttribute('data-tab-content');
            if (contentTabId === tabId) {
                content.classList.remove('hidden');
            } else {
                content.classList.add('hidden');
            }
        });
    }
});

/**
 * Initialize the search tab UI
 */
function initSearchUI() {
    // Get DOM elements
    const queryInput = document.getElementById('query');
    const seedSelect = document.getElementById('seed');
    const toggleAdvancedBtn = document.getElementById('toggle-advanced');
    const advancedOptions = document.getElementById('advanced-options');
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const maxResultsSlider = document.getElementById('max-results');
    const maxResultsValue = document.getElementById('max-results-value');
    const useEllipsoidal = document.getElementById('use-ellipsoidal');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('results-container');
    const resultsStats = document.getElementById('results-stats');
    const loading = document.getElementById('loading');
    
    // Load documents for seed selection
    loadDocuments(seedSelect);
    
    // Check if ellipsoidal transformation is available
    checkTransformations(useEllipsoidal);
    
    // Toggle advanced options
    toggleAdvancedBtn.addEventListener('click', () => {
        advancedOptions.classList.toggle('hidden');
        const isHidden = advancedOptions.classList.contains('hidden');
        toggleAdvancedBtn.innerHTML = isHidden 
            ? '<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>Advanced Options'
            : '<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 12H6" /></svg>Advanced Options';
    });
    
    // Update slider values
    thresholdSlider.addEventListener('input', () => {
        thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    });
    
    maxResultsSlider.addEventListener('input', () => {
        maxResultsValue.textContent = maxResultsSlider.value;
    });
    
    // Search button click handler
    searchButton.addEventListener('click', () => {
        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a search query');
            return;
        }
        
        const seedId = seedSelect.value;
        const threshold = parseFloat(thresholdSlider.value);
        const maxResults = parseInt(maxResultsSlider.value);
        const useEllipsoidalValue = useEllipsoidal.checked;
        
        performSearch(query, seedId, threshold, maxResults, useEllipsoidalValue, resultsContainer, resultsStats, loading);
    });
    
    // Add enter key handler for search
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            searchButton.click();
        }
    });
}

/**
 * Load documents for seed selection
 * @param {HTMLSelectElement} seedSelect - Select element for seed documents
 */
async function loadDocuments(seedSelect) {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        if (data.success) {
            // Clear existing options except "No seed document"
            seedSelect.innerHTML = '<option value="">No seed document</option>';
            
            // Add document options
            data.documents.forEach(doc => {
                const option = document.createElement('option');
                option.value = doc.id;
                option.textContent = doc.title;
                seedSelect.appendChild(option);
            });
        } else {
            console.error('Error loading documents:', data.error);
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

/**
 * Check if ellipsoidal transformation is available
 * @param {HTMLInputElement} useEllipsoidal - Checkbox for using ellipsoidal transformation
 */
async function checkTransformations(useEllipsoidal) {
    try {
        const response = await fetch('/api/transformations');
        const data = await response.json();
        
        if (data.success) {
            // Enable or disable ellipsoidal option
            useEllipsoidal.disabled = !data.ellipsoidal_available;
            
            // Add a note if disabled
            if (!data.ellipsoidal_available) {
                const note = document.createElement('small');
                note.className = 'block text-gray-500 mt-1';
                note.textContent = 'Ellipsoidal transformation not available. Please provide weights file.';
                useEllipsoidal.parentNode.appendChild(note);
            }
        }
    } catch (error) {
        console.error('Error checking transformations:', error);
        useEllipsoidal.disabled = true;
    }
}

/**
 * Perform search with the given parameters
 * @param {string} query - Search query
 * @param {string} seedId - Seed document ID (optional)
 * @param {number} threshold - Similarity threshold
 * @param {number} maxResults - Maximum number of results
 * @param {boolean} useEllipsoidal - Whether to use ellipsoidal transformation
 * @param {HTMLElement} resultsContainer - Container for results
 * @param {HTMLElement} resultsStats - Container for results stats
 * @param {HTMLElement} loading - Loading indicator
 */
async function performSearch(query, seedId, threshold, maxResults, useEllipsoidal, resultsContainer, resultsStats, loading) {
    // Show loading indicator
    loading.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    resultsStats.textContent = '';
    
    try {
        // Prepare request body
        const requestBody = {
            query: query,
            threshold: threshold,
            max_results: maxResults,
            use_ellipsoidal: useEllipsoidal
        };
        
        // Add seed document if selected
        if (seedId) {
            requestBody.seed_doc_id = seedId;
        }
        
        // Get selected session
        const sessionSelect = document.getElementById('session-select');
        if (sessionSelect.value) {
            requestBody.session_id = sessionSelect.value;
        }
        
        // Send search request
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update results stats
            resultsStats.textContent = `${data.results.length} results (${data.execution_time.toFixed(3)} seconds)`;
            
            // Display results
            displaySearchResults(data.results, resultsContainer);
        } else {
            resultsContainer.innerHTML = `
                <div class="text-red-600 text-center py-8">
                    Error: ${data.error || 'Failed to perform search'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error performing search:', error);
        resultsContainer.innerHTML = `
            <div class="text-red-600 text-center py-8">
                Error: ${error.message || 'Failed to perform search'}
            </div>
        `;
    } finally {
        // Hide loading indicator
        loading.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
    }
}

/**
 * Display search results in the UI
 * @param {Array} results - Search results
 * @param {HTMLElement} container - Container for results
 */
function displaySearchResults(results, container) {
    if (results.length === 0) {
        container.innerHTML = `
            <div class="text-gray-600 text-center py-8">
                No results found
            </div>
        `;
        return;
    }
    
    // Generate HTML for results
    const resultsHtml = results.map((result, index) => {
        // Generate tags HTML if available
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
                <p class="mt-2 text-gray-600">${result.content ? result.content.substring(0, 200) + '...' : 'No content'}</p>
                ${tags}
            </div>
        `;
    }).join('');
    
    container.innerHTML = resultsHtml;
}

/**
 * Initialize the analytics UI
 * @param {ChartManager} chartManager - Chart manager instance
 * @param {SessionManager} sessionManager - Session manager instance
 */
function initAnalyticsUI(chartManager, sessionManager) {
    const analyticsForm = document.getElementById('analytics-form');
    if (!analyticsForm) return;
    
    const analyticsQuery = document.getElementById('analytics-query');
    const groundTruthInput = document.getElementById('ground-truth-file');
    const createGroundTruthBtn = document.getElementById('create-ground-truth');
    const runAnalyticsBtn = document.getElementById('run-analytics');
    const analyticsLoading = document.getElementById('analytics-loading');
    const analyticsResults = document.getElementById('analytics-results');
    
    // Create ground truth
    if (createGroundTruthBtn) {
        createGroundTruthBtn.addEventListener('click', async () => {
            const query = analyticsQuery.value.trim();
            if (!query) {
                alert('Please enter a query for ground truth creation');
                return;
            }
            
            analyticsLoading.classList.remove('hidden');
            
            try {
                const response = await fetch('/api/analytics/ground-truth', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert(`Ground truth created for query: ${query}`);
                } else {
                    alert(`Error creating ground truth: ${data.error}`);
                }
            } catch (error) {
                console.error('Error creating ground truth:', error);
                alert('An error occurred while creating ground truth');
            } finally {
                analyticsLoading.classList.add('hidden');
            }
        });
    }
    
    // Run analytics
    if (runAnalyticsBtn) {
        analyticsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = analyticsQuery.value.trim();
            if (!query) {
                alert('Please enter a query for analytics');
                return;
            }
            
            const formData = new FormData();
            formData.append('query', query);
            
            if (groundTruthInput.files.length > 0) {
                formData.append('ground_truth_file', groundTruthInput.files[0]);
            }
            
            analyticsLoading.classList.remove('hidden');
            analyticsResults.innerHTML = '';
            
            try {
                const response = await fetch('/api/analytics/metrics', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Create metrics display
                    const metricsHtml = `
                        <div class="bg-white p-4 rounded-lg shadow-md mb-4">
                            <h3 class="text-lg font-medium text-gray-900 mb-2">Metrics Summary</h3>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <p class="text-sm text-gray-600">Average Precision</p>
                                    <p class="text-xl font-semibold">${data.metrics.average_precision.toFixed(3)}</p>
                                </div>
                                ${Object.entries(data.metrics.precision_at_k).map(([k, v]) => `
                                    <div>
                                        <p class="text-sm text-gray-600">${k}</p>
                                        <p class="text-xl font-semibold">${v.toFixed(3)}</p>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
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
                    
                    analyticsResults.innerHTML = metricsHtml;
                    
                    // Create charts
                    chartManager.createPrecisionRecallCurve(
                        'pr-curve-chart', 
                        data.curve_data,
                        data.metrics.average_precision
                    );
                    
                    chartManager.createF1ThresholdChart(
                        'f1-threshold-chart',
                        data.metrics.threshold_metrics
                    );
                    
                    chartManager.createPrecisionAtKChart(
                        'precision-at-k-chart',
                        data.metrics.precision_at_k
                    );
                    
                    // Save to session if available
                    if (sessionManager.currentSession) {
                        const sessionId = sessionManager.currentSession.id;
                        
                        // Save analytics to session
                        await fetch(`/api/sessions/${sessionId}/analytics`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                query: query,
                                metrics: data.metrics
                            })
                        });
                    }
                } else {
                    analyticsResults.innerHTML = `
                        <div class="text-red-600 text-center py-8">
                            Error: ${data.error}
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error running analytics:', error);
                analyticsResults.innerHTML = `
                    <div class="text-red-600 text-center py-8">
                        An error occurred while running analytics.
                    </div>
                `;
            } finally {
                analyticsLoading.classList.add('hidden');
            }
        });
    }
}

/**
 * Initialize the comparison UI
 * @param {ChartManager} chartManager - Chart manager instance
 * @param {SessionManager} sessionManager - Session manager instance
 */
function initComparisonUI(chartManager, sessionManager) {
    const comparisonForm = document.getElementById('comparison-form');
    if (!comparisonForm) return;
    
    const comparisonQuery = document.getElementById('comparison-query');
    const standardSearchCheck = document.getElementById('comparison-standard');
    const seededSearchCheck = document.getElementById('comparison-seeded');
    const ellipsoidalCheck = document.getElementById('comparison-ellipsoidal');
    const seedDocSelect = document.getElementById('comparison-seed');
    const runComparisonBtn = document.getElementById('run-comparison');
    const comparisonLoading = document.getElementById('comparison-loading');
    const comparisonResults = document.getElementById('comparison-results');
    
    // Load seed documents
    if (seedDocSelect) {
        loadDocuments(seedDocSelect);
    }
    
    // Run comparison
    if (runComparisonBtn) {
        comparisonForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = comparisonQuery.value.trim();
            if (!query) {
                alert('Please enter a query for comparison');
                return;
            }
            
            // Get selected algorithms
            const algorithms = [];
            if (standardSearchCheck.checked) {
                algorithms.push('standard');
            }
            if (seededSearchCheck.checked) {
                algorithms.push('seeded');
            }
            if (ellipsoidalCheck.checked) {
                algorithms.push('ellipsoidal');
            }
            
            if (algorithms.length === 0) {
                alert('Please select at least one algorithm for comparison');
                return;
            }
            
            // Get seed document
            const seedDoc = seedDocSelect.value;
            if (algorithms.includes('seeded') && !seedDoc) {
                alert('Please select a seed document for seeded search comparison');
                return;
            }
            
            comparisonLoading.classList.remove('hidden');
            comparisonResults.innerHTML = '';
            
            try {
                const response = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        algorithms: algorithms,
                        seed: seedDoc,
                        session_id: sessionManager.currentSession?.id
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Create comparison table
                    const tableHtml = `
                        <div class="bg-white p-4 rounded-lg shadow-md mb-4 overflow-x-auto">
                            <h3 class="text-lg font-medium text-gray-900 mb-2">Comparison Results</h3>
                            <table class="min-w-full">
                                <thead class="bg-gray-50">
                                    <tr>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Algorithm</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. Precision</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P@5</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Exec. Time (s)</th>
                                    </tr>
                                </thead>
                                <tbody class="bg-white divide-y divide-gray-200">
                                    ${data.results.map(result => `
                                        <tr>
                                            <td class="px-6 py-4 whitespace-nowrap">${result.name}</td>
                                            <td class="px-6 py-4 whitespace-nowrap">${result.metrics.average_precision.toFixed(3)}</td>
                                            <td class="px-6 py-4 whitespace-nowrap">${result.metrics.precision_at_k['p@5']?.toFixed(3) || 'N/A'}</td>
                                            <td class="px-6 py-4 whitespace-nowrap">${result.execution_time.toFixed(3)}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="bg-white p-4 rounded-lg shadow-md mb-4">
                            <canvas id="comparison-chart" height="400"></canvas>
                        </div>
                    `;
                    
                    comparisonResults.innerHTML = tableHtml;
                    
                    // Create comparison chart
                    chartManager.createComparisonChart('comparison-chart', data.results);
                } else {
                    comparisonResults.innerHTML = `
                        <div class="text-red-600 text-center py-8">
                            Error: ${data.error}
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error running comparison:', error);
                comparisonResults.innerHTML = `
                    <div class="text-red-600 text-center py-8">
                        An error occurred while running comparison.
                    </div>
                `;
            } finally {
                comparisonLoading.classList.add('hidden');
            }
        });
    }
}

/**
 * Initialize the visualization UI
 * @param {SessionManager} sessionManager - Session manager instance
 */
function initVisualizationUI(sessionManager) {
    const visualizationForm = document.getElementById('visualization-form');
    if (!visualizationForm) return;
    
    const visualizationMethod = document.getElementById('visualization-method');
    const visualizationType = document.getElementById('visualization-type');
    const visualizationQuery = document.getElementById('visualization-query');
    const dimensions = document.getElementById('visualization-dimensions');
    const runVisualizationBtn = document.getElementById('run-visualization');
    const visualizationLoading = document.getElementById('visualization-loading');
    const visualizationResults = document.getElementById('visualization-results');
    
    // Run visualization
    visualizationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const method = visualizationMethod.value;
        const type = visualizationType.value;
        const query = type === 'query' ? visualizationQuery.value.trim() : '';
        const dims = parseInt(dimensions.value);
        
        if (type === 'query' && !query) {
            alert('Please enter a query for visualization');
            return;
        }
        
        visualizationLoading.classList.remove('hidden');
        visualizationResults.innerHTML = '';
        
        try {
            const response = await fetch('/api/visualize/embeddings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    method: method,
                    type: type,
                    query: query,
                    dimensions: dims,
                    session_id: sessionManager.currentSession?.id
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Create visualization container
                visualizationResults.innerHTML = `
                    <div class="bg-white p-4 rounded-lg shadow-md mb-4">
                        <h3 class="text-lg font-medium text-gray-900 mb-2">
                            Embedding Visualization (${method.toUpperCase()}, ${dims}D)
                        </h3>
                        <div id="visualization-container" style="height: 500px;"></div>
                    </div>
                `;
                
                // Create visualization
                const container = document.getElementById('visualization-container');
                const visualizer = new EmbeddingVisualizer(container);
                
                if (dims === 2) {
                    visualizer.create2DScatterPlot(data.visualization, (point) => {
                        alert(`Selected document: ${point.title}`);
                    });
                } else if (dims === 3) {
                    visualizer.create3DScatterPlot(data.visualization, (point) => {
                        alert(`Selected document: ${point.title}`);
                    });
                }
            } else {
                visualizationResults.innerHTML = `
                    <div class="text-red-600 text-center py-8">
                        Error: ${data.error}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error creating visualization:', error);
            visualizationResults.innerHTML = `
                <div class="text-red-600 text-center py-8">
                    An error occurred while creating visualization.
                </div>
            `;
        } finally {
            visualizationLoading.classList.add('hidden');
        }
    });
}

/**
 * Initialize the session management UI
 * @param {SessionManager} sessionManager - Session manager instance
 * @param {ChartManager} chartManager - Chart manager instance
 */
function initSessionUI(sessionManager, chartManager) {
    const createSessionForm = document.getElementById('create-session-form');
    const sessionsList = document.getElementById('sessions-list');
    const sessionSelect = document.getElementById('session-select');
    
    // Load sessions on startup
    sessionManager.loadSessions().then(() => {
        if (sessionSelect) {
            sessionManager.populateSessionsSelect(sessionSelect, () => {
                const sessionId = sessionSelect.value;
                if (sessionId) {
                    sessionManager.loadSession(sessionId);
                }
            });
        }
        
        refreshSessionsList();
    }).catch(error => {
        console.error('Error loading sessions:', error);
    });
    
    // Create session
    if (createSessionForm) {
        createSessionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const sessionName = document.getElementById('session-name').value.trim();
            if (!sessionName) {
                alert('Please enter a session name');
                return;
            }
            
            try {
                await sessionManager.createSession(sessionName);
                alert(`Session "${sessionName}" created successfully`);
                
                // Update session selects
                if (sessionSelect) {
                    sessionManager.populateSessionsSelect(sessionSelect);
                }
                
                refreshSessionsList();
                
                // Clear form
                document.getElementById('session-name').value = '';
            } catch (error) {
                console.error('Error creating session:', error);
                alert(`Error creating session: ${error.message}`);
            }
        });
    }
    
    // Session import
    const importSessionInput = document.getElementById('import-session');
    if (importSessionInput) {
        importSessionInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    await sessionManager.importSession(file);
                    alert(`Session imported successfully`);
                    
                    // Update session selects
                    if (sessionSelect) {
                        sessionManager.populateSessionsSelect(sessionSelect);
                    }
                    
                    refreshSessionsList();
                    
                    // Clear input
                    importSessionInput.value = '';
                } catch (error) {
                    console.error('Error importing session:', error);
                    alert(`Error importing session: ${error.message}`);
                }
            }
        });
    }
    
    // Refresh sessions list
    function refreshSessionsList() {
        if (!sessionsList) return;
        
        sessionsList.innerHTML = '';
        
        if (sessionManager.sessionsList.length === 0) {
            sessionsList.innerHTML = `
                <div class="text-gray-600 text-center py-4">
                    No sessions available
                </div>
            `;
            return;
        }
        
        sessionManager.sessionsList.forEach(session => {
            const sessionCard = document.createElement('div');
            sessionCard.className = 'bg-white p-4 rounded-lg shadow-md mb-4';
            sessionCard.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <h3 class="text-lg font-medium text-gray-900">${session.name}</h3>
                        <p class="text-sm text-gray-600">Created: ${new Date(session.created_at).toLocaleString()}</p>
                        <p class="text-sm text-gray-600">
                            Contents: ${session.queries.length} queries, 
                            ${session.results_count} results, 
                            ${session.analytics_count} analytics, 
                            ${session.visualizations_count} visualizations
                        </p>
                    </div>
                    <div class="flex space-x-2">
                        <button class="session-load text-sm bg-indigo-100 text-indigo-800 px-3 py-1 rounded hover:bg-indigo-200"
                                data-session-id="${session.id}">
                            Load
                        </button>
                        <button class="session-export text-sm bg-green-100 text-green-800 px-3 py-1 rounded hover:bg-green-200"
                                data-session-id="${session.id}">
                            Export
                        </button>
                        <button class="session-delete text-sm bg-red-100 text-red-800 px-3 py-1 rounded hover:bg-red-200"
                                data-session-id="${session.id}">
                            Delete
                        </button>
                    </div>
                </div>
            `;
            
            sessionsList.appendChild(sessionCard);
        });
        
        // Add event listeners
        document.querySelectorAll('.session-load').forEach(button => {
            button.addEventListener('click', async () => {
                const sessionId = button.getAttribute('data-session-id');
                try {
                    await sessionManager.loadSession(sessionId);
                    alert(`Session loaded successfully`);
                    
                    // Update session select
                    if (sessionSelect) {
                        sessionSelect.value = sessionId;
                    }
                } catch (error) {
                    console.error(`Error loading session ${sessionId}:`, error);
                    alert(`Error loading session: ${error.message}`);
                }
            });
        });
        
        document.querySelectorAll('.session-export').forEach(button => {
            button.addEventListener('click', () => {
                const sessionId = button.getAttribute('data-session-id');
                sessionManager.exportSession(sessionId);
            });
        });
        
        document.querySelectorAll('.session-delete').forEach(button => {
            button.addEventListener('click', async () => {
                const sessionId = button.getAttribute('data-session-id');
                if (confirm('Are you sure you want to delete this session?')) {
                    try {
                        await sessionManager.deleteSession(sessionId);
                        alert(`Session deleted successfully`);
                        
                        // Update session select
                        if (sessionSelect) {
                            sessionManager.populateSessionsSelect(sessionSelect);
                        }
                        
                        refreshSessionsList();
                    } catch (error) {
                        console.error(`Error deleting session ${sessionId}:`, error);
                        alert(`Error deleting session: ${error.message}`);
                    }
                }
            });
        });
    }
}