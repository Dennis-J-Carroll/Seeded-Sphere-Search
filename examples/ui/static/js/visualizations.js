/**
 * Visualizations module for displaying embeddings in 2D/3D space
 */

class EmbeddingVisualizer {
    constructor(container) {
        this.container = container;
        this.chart = null;
        this.currentVisualization = null;
    }
    
    /**
     * Create a 2D scatter plot of embeddings
     * @param {object} visualizationData - Data returned from the API
     * @param {string} pointClickHandler - Function to call when a point is clicked
     */
    create2DScatterPlot(visualizationData, pointClickHandler = null) {
        // Clean up any existing visualization
        if (this.chart) {
            this.chart.destroy();
        }
        
        const points = visualizationData.points;
        
        // Extract data
        const data = points.map(point => ({
            x: point.coords[0],
            y: point.coords[1],
            id: point.id,
            title: point.title,
            color: point.color,
            tag: point.tag,
            isQuery: point.isQuery || false,
            score: point.score || null,
            radius: point.isQuery ? 8 : (point.score ? 4 + point.score * 4 : 5)
        }));
        
        // Group points by tag
        const tagGroups = {};
        data.forEach(point => {
            if (!tagGroups[point.tag]) {
                tagGroups[point.tag] = [];
            }
            tagGroups[point.tag].push(point);
        });
        
        // Create datasets for each tag
        const datasets = Object.keys(tagGroups).map(tag => ({
            label: tag,
            data: tagGroups[tag].map(point => ({ x: point.x, y: point.y })),
            backgroundColor: tagGroups[tag].map(point => point.color),
            pointRadius: tagGroups[tag].map(point => point.radius),
            pointHoverRadius: tagGroups[tag].map(point => point.radius + 2),
            pointStyle: tagGroups[tag].map(point => point.isQuery ? 'triangle' : 'circle')
        }));
        
        // Create chart
        const ctx = document.createElement('canvas');
        ctx.width = this.container.offsetWidth;
        ctx.height = 500;
        ctx.style.margin = '0 auto';
        
        // Clear container and add canvas
        this.container.innerHTML = '';
        this.container.appendChild(ctx);
        
        // Create chart
        this.chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Dimension 1'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Dimension 2'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Embedding Space Visualization (${visualizationData.method.toUpperCase()})`
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const index = context.dataIndex;
                                const tag = context.dataset.label;
                                const point = tagGroups[tag][index];
                                let label = point.title;
                                
                                if (point.score !== null) {
                                    label += ` (Score: ${point.score.toFixed(3)})`;
                                }
                                
                                return label;
                            }
                        }
                    },
                    legend: {
                        position: 'right'
                    }
                },
                onClick: (e, elements) => {
                    if (elements.length > 0 && pointClickHandler) {
                        const element = elements[0];
                        const datasetIndex = element.datasetIndex;
                        const index = element.index;
                        const tag = Object.keys(tagGroups)[datasetIndex];
                        const point = tagGroups[tag][index];
                        pointClickHandler(point);
                    }
                }
            }
        });
        
        // Store current visualization
        this.currentVisualization = visualizationData;
        
        return this.chart;
    }
    
    /**
     * Create a 3D scatter plot of embeddings using Plotly
     * @param {object} visualizationData - Data returned from the API
     * @param {function} pointClickHandler - Function to call when a point is clicked
     */
    create3DScatterPlot(visualizationData, pointClickHandler = null) {
        // Clean up any existing visualization
        if (this.chart) {
            if (typeof this.chart.destroy === 'function') {
                this.chart.destroy();
            } else {
                // Plotly cleanup
                Plotly.purge(this.container);
            }
        }
        
        const points = visualizationData.points;
        
        // Extract data
        const data = points.map(point => ({
            x: point.coords[0],
            y: point.coords[1],
            z: point.coords[2],
            id: point.id,
            title: point.title,
            color: point.color,
            tag: point.tag,
            isQuery: point.isQuery || false,
            score: point.score || null
        }));
        
        // Group points by tag
        const tagGroups = {};
        data.forEach(point => {
            if (!tagGroups[point.tag]) {
                tagGroups[point.tag] = {
                    x: [],
                    y: [],
                    z: [],
                    text: [],
                    ids: [],
                    scores: []
                };
            }
            
            tagGroups[point.tag].x.push(point.x);
            tagGroups[point.tag].y.push(point.y);
            tagGroups[point.tag].z.push(point.z);
            tagGroups[point.tag].text.push(point.title);
            tagGroups[point.tag].ids.push(point.id);
            tagGroups[point.tag].scores.push(point.score);
        });
        
        // Create traces for Plotly
        const traces = Object.keys(tagGroups).map(tag => {
            const group = tagGroups[tag];
            const trace = {
                type: 'scatter3d',
                mode: 'markers',
                name: tag,
                x: group.x,
                y: group.y,
                z: group.z,
                text: group.text,
                ids: group.ids,
                scores: group.scores,
                marker: {
                    size: tag === 'query' ? 8 : 5,
                    color: points.filter(p => p.tag === tag)[0].color,
                    opacity: 0.8
                },
                hovertemplate: '%{text}<br>ID: %{ids}<extra></extra>'
            };
            
            return trace;
        });
        
        // Clear container
        this.container.innerHTML = '';
        
        // Create layout
        const layout = {
            title: `Embedding Space Visualization (${visualizationData.method.toUpperCase()})`,
            height: 600,
            scene: {
                xaxis: { title: 'Dimension 1' },
                yaxis: { title: 'Dimension 2' },
                zaxis: { title: 'Dimension 3' }
            },
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 50
            }
        };
        
        // Create Plotly chart
        Plotly.newPlot(this.container, traces, layout);
        
        // Add click handler
        if (pointClickHandler) {
            this.container.on('plotly_click', function(data) {
                const point = data.points[0];
                const pointData = {
                    id: point.data.ids[point.pointNumber],
                    title: point.data.text[point.pointNumber],
                    tag: point.data.name,
                    isQuery: point.data.name === 'query',
                    score: point.data.scores[point.pointNumber]
                };
                pointClickHandler(pointData);
            });
        }
        
        // Store current visualization
        this.currentVisualization = visualizationData;
        
        return this.container;
    }
    
    /**
     * Display points with different colors based on type or tags
     * @param {string} colorType - How to color points ('tag', 'score', or 'custom')
     */
    updateColors(colorType) {
        if (!this.chart || !this.currentVisualization) {
            return;
        }
        
        const points = this.currentVisualization.points;
        
        if (colorType === 'tag') {
            // Already colored by tag
            return;
            
        } else if (colorType === 'score') {
            // Color by score
            this.chart.data.datasets.forEach((dataset, datasetIndex) => {
                const updatedColors = dataset.data.map((point, pointIndex) => {
                    const tag = dataset.label;
                    const originalPoint = points.find(p => 
                        p.tag === tag && 
                        Math.abs(p.coords[0] - point.x) < 0.0001 && 
                        Math.abs(p.coords[1] - point.y) < 0.0001
                    );
                    
                    if (originalPoint.score) {
                        // Scale from blue (low) to red (high)
                        const r = Math.floor(originalPoint.score * 255);
                        const b = Math.floor((1 - originalPoint.score) * 255);
                        return `rgba(${r}, 0, ${b}, 0.8)`;
                    } else {
                        return 'rgba(200, 200, 200, 0.8)';
                    }
                });
                
                this.chart.data.datasets[datasetIndex].backgroundColor = updatedColors;
            });
            
            this.chart.update();
        }
    }
} 