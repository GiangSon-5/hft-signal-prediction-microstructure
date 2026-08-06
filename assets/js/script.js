/**
 * HFT Market Data Analytics - Dashboard Script
 * Manages tab switching, task state, progress tracking, and chart rendering.
 */

// All Subtask Identifiers
const SUBTASKS = [
    'st-1-1', 'st-1-2', 'st-1-3', 'st-1-4', 'st-1-5',
    'st-2-1', 'st-2-2', 'st-2-3', 'st-2-4', 'st-2-5',
    'st-3-1', 'st-3-2', 'st-3-3'
];

// Mapping Subtasks to Tasks
const TASK_MAP = {
    'task1': ['st-1-1', 'st-1-2', 'st-1-3', 'st-1-4', 'st-1-5'],
    'task2': ['st-2-1', 'st-2-2', 'st-2-3', 'st-2-4', 'st-2-5'],
    'task3': ['st-3-1', 'st-3-2', 'st-3-3']
};

let returnsChart = null;
let regimesChart = null;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    loadChecklistState();
    updateAllProgressBars();
    initCharts();
});

/**
 * Tab Switching Logic
 */
function switchTab(tabName) {
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.nav-tab').forEach(el => {
        el.classList.remove('text-white', 'bg-indigo-600/30', 'border', 'border-indigo-500/40', 'shadow-sm');
        el.classList.add('text-slate-400');
    });

    const targetTab = document.getElementById(`tab-content-${tabName}`);
    if (targetTab) {
        targetTab.classList.remove('hidden');
    }
    
    const activeNav = document.getElementById(`tab-${tabName}`);
    if (activeNav) {
        activeNav.classList.remove('text-slate-400');
        activeNav.classList.add('text-white', 'bg-indigo-600/30', 'border', 'border-indigo-500/40', 'shadow-sm');
    }
}

/**
 * Checkbox Toggle Handler
 */
function toggleSubtask(checkbox) {
    saveChecklistState();
    updateAllProgressBars();
}

/**
 * LocalStorage State Management
 */
function saveChecklistState() {
    const state = {};
    SUBTASKS.forEach(id => {
        const el = document.getElementById(id);
        if (el) state[id] = el.checked;
    });
    localStorage.setItem('hft_dashboard_subtasks', JSON.stringify(state));
}

function loadChecklistState() {
    const saved = localStorage.getItem('hft_dashboard_subtasks');
    if (saved) {
        try {
            const state = JSON.parse(saved);
            Object.keys(state).forEach(id => {
                const el = document.getElementById(id);
                if (el) el.checked = state[id];
            });
        } catch (e) {
            console.error('Error restoring checklist state:', e);
        }
    }
}

function resetAllChecklists() {
    SUBTASKS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.checked = false;
    });
    saveChecklistState();
    updateAllProgressBars();
}

/**
 * Progress Calculation Engine
 */
function updateAllProgressBars() {
    let totalChecked = 0;

    Object.keys(TASK_MAP).forEach(taskKey => {
        const list = TASK_MAP[taskKey];
        let taskChecked = 0;
        list.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.checked) {
                taskChecked++;
                totalChecked++;
            }
        });

        const pct = Math.round((taskChecked / list.length) * 100);
        const bar = document.getElementById(`bar-${taskKey}`);
        if (bar) bar.style.width = `${pct}%`;

        // Update task status dropdown if 100% completed
        const statusSelect = document.getElementById(`status-${taskKey}`);
        if (statusSelect) {
            if (pct === 100) {
                statusSelect.value = 'completed';
            } else if (pct > 0 && statusSelect.value === 'not_started') {
                statusSelect.value = 'in_progress';
            }
        }
    });

    // Global Statistics
    const globalPct = Math.round((totalChecked / SUBTASKS.length) * 100);
    const globalBar = document.getElementById('global-progress-bar');
    const globalText = document.getElementById('global-percent');
    
    if (globalBar) globalBar.style.width = `${globalPct}%`;
    if (globalText) globalText.innerText = `${globalPct}%`;

    const totalEl = document.getElementById('total-subtasks-count');
    const completedEl = document.getElementById('completed-subtasks-count');
    const remainingEl = document.getElementById('remaining-subtasks-count');

    if (totalEl) totalEl.innerText = SUBTASKS.length;
    if (completedEl) completedEl.innerText = totalChecked;
    if (remainingEl) remainingEl.innerText = SUBTASKS.length - totalChecked;
}

/**
 * Task Status Change Handler
 */
function updateTaskStatus(taskKey, value) {
    console.log(`Updated status for ${taskKey}: ${value}`);
}

/**
 * Task Filter Handler
 */
function filterTasks(filter) {
    document.querySelectorAll('.task-filter-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-slate-800', 'text-white');
    });
    
    if (event && event.target) {
        event.target.classList.add('active', 'bg-slate-800', 'text-white');
    }

    const cards = document.querySelectorAll('.task-card');
    cards.forEach(card => {
        if (filter === 'all' || card.getAttribute('data-task') === filter) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

/**
 * Chart.js Visualization Engine
 */
function initCharts() {
    const ctx1 = document.getElementById('chart-returns-dist');
    if (ctx1 && typeof Chart !== 'undefined') {
        returnsChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: Array.from({length: 40}, (_, i) => (i - 20) * 0.1),
                datasets: [
                    {
                        label: '1m Log Return (Fat-Tailed Empirical)',
                        data: [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.2, 7.5, 18.0, 35.0, 18.0, 7.5, 3.2, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Normal Distribution (Gaussian Baseline)',
                        data: [0.01, 0.03, 0.1, 0.3, 0.8, 2.0, 4.5, 9.0, 16.0, 24.0, 28.0, 24.0, 16.0, 9.0, 4.5, 2.0, 0.8, 0.3, 0.1, 0.03, 0.01],
                        borderColor: '#64748b',
                        borderDash: [4, 4],
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } }
                }
            }
        });
    }

    const ctx2 = document.getElementById('chart-volatility-regimes');
    if (ctx2 && typeof Chart !== 'undefined') {
        regimesChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: Array.from({length: 50}, (_, i) => `t+${i}`),
                datasets: [
                    {
                        label: 'Rolling Volatility (60m Window)',
                        data: [12, 11, 13, 14, 12, 15, 14, 13, 15, 16, 28, 42, 55, 68, 62, 58, 65, 70, 64, 59, 22, 18, 15, 14, 12, 13, 15, 14, 16, 17, 15, 14, 45, 58, 62, 75, 80, 71, 66, 60, 25, 20, 18, 15, 14, 13, 12, 14, 15, 13],
                        borderColor: '#818cf8',
                        backgroundColor: 'rgba(129, 140, 248, 0.1)',
                        fill: true,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } }
                }
            }
        });
    }
}

/**
 * Simulate Live Chart Updates for Testing
 */
function simulateEdaUpdate() {
    if (returnsChart) {
        returnsChart.data.datasets[0].data = returnsChart.data.datasets[0].data.map(v => v * (0.9 + Math.random() * 0.2));
        returnsChart.update();
    }
    if (regimesChart) {
        regimesChart.data.datasets[0].data = regimesChart.data.datasets[0].data.map(v => v * (0.9 + Math.random() * 0.2));
        regimesChart.update();
    }
}
