/**
 * Charts module for visualizing search analytics
 */

class ChartManager {
    constructor() {
        this.charts = {};
    }
    
    /**
     * Create or update a precision-recall curve chart
     * @param {string} elementId - ID of the canvas element
     * @param {object} curveData - Precision-recall curve data
     * @param {number} averagePrecision - Average precision value
     */
    createPrecisionRecallCurve(elementId, curveData, averagePrecision) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
        }
        
        // Create new chart
        this.charts[elementId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: curveData.labels,
                datasets: curveData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Recall'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Precision'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Precision-Recall Curve (AP: ${averagePrecision.toFixed(3)})`
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Precision: ${context.parsed.y.toFixed(3)}, Recall: ${context.parsed.x.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[elementId];
    }
    
    /**
     * Create or update an F1 score vs threshold chart
     * @param {string} elementId - ID of the canvas element
     * @param {Array} thresholdMetrics - Metrics at different thresholds
     */
    createF1ThresholdChart(elementId, thresholdMetrics) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Extract data from threshold metrics
        const thresholds = thresholdMetrics.map(m => m.threshold);
        const precisionValues = thresholdMetrics.map(m => m.precision);
        const recallValues = thresholdMetrics.map(m => m.recall);
        const f1Values = thresholdMetrics.map(m => m.f1);
        
        // Destroy existing chart if it exists
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
        }
        
        // Create new chart
        this.charts[elementId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: thresholds,
                datasets: [
                    {
                        label: 'F1 Score',
                        data: f1Values,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        fill: false
                    },
                    {
                        label: 'Precision',
                        data: precisionValues,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        fill: false
                    },
                    {
                        label: 'Recall',
                        data: recallValues,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Threshold'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Score'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Metrics vs Threshold'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const datasetLabel = context.dataset.label;
                                const value = context.parsed.y.toFixed(3);
                                const threshold = context.parsed.x.toFixed(2);
                                return `${datasetLabel}: ${value} (threshold: ${threshold})`;
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[elementId];
    }
    
    /**
     * Create or update a precision@k chart
     * @param {string} elementId - ID of the canvas element
     * @param {object} precisionAtK - Precision at different k values
     */
    createPrecisionAtKChart(elementId, precisionAtK) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Extract data from precision@k
        const kValues = Object.keys(precisionAtK);
        const precisionValues = Object.values(precisionAtK);
        
        // Destroy existing chart if it exists
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
        }
        
        // Create new chart
        this.charts[elementId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: kValues,
                datasets: [
                    {
                        label: 'Precision@K',
                        data: precisionValues,
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Precision'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'k'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Precision@K'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Precision: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts[elementId];
    }
    
    /**
     * Create or update a comparison chart
     * @param {string} elementId - ID of the canvas element
     * @param {object} comparisonData - Algorithm comparison data
     */
    createComparisonChart(elementId, comparisonData) {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[elementId]) {
            this.charts[elementId].destroy();
        }
        
        // Extract algorithms and metrics
        const algorithms = comparisonData.map(item => item.name);
        const averagePrecision = comparisonData.map(item => item.metrics.average_precision);
        const executionTime = comparisonData.map(item => item.execution_time);
        
        // Create new chart - bar chart with dual y-axis
        this.charts[elementId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: algorithms,
                datasets: [
                    {
                        label: 'Average Precision',
                        data: averagePrecision,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Execution Time (s)',
                        data: executionTime,
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        type: 'line',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Average Precision'
                        },
                        min: 0,
                        max: 1
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Execution Time (s)'
                        },
                        min: 0,
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Algorithm Comparison'
                    }
                }
            }
        });
        
        return this.charts[elementId];
    }
} 