/**
 * HFT Market Data Analytics - Dashboard Script
 * Manages tab switching, task filtering, and Chart.js rendering.
 */

let returnsChart = null;
let regimesChart = null;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
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

    // Auto toggle dataset schema details visibility based on active tab
    const schemaContainer = document.getElementById('dataset-schema-container');
    const toggleBtnText = document.getElementById('schema-toggle-text');
    const toggleBtnIcon = document.getElementById('schema-toggle-icon');
    if (schemaContainer) {
        if (tabName === 'roadmap') {
            // Overview Tab: Show dataset schema & glossary by default
            schemaContainer.classList.remove('hidden');
            if (toggleBtnText) toggleBtnText.textContent = 'Ẩn Chi Tiết Dữ Liệu & Thuật Ngữ';
            if (toggleBtnIcon) toggleBtnIcon.className = 'fa-solid fa-chevron-up text-xs';
        } else {
            // Child Tabs (Task 1, 2, 3, Report): Hide by default, user can expand via toggle button
            schemaContainer.classList.add('hidden');
            if (toggleBtnText) toggleBtnText.textContent = 'Xem Chi Tiết 9 Cột Dữ Liệu & Thuật Ngữ OHLCV';
            if (toggleBtnIcon) toggleBtnIcon.className = 'fa-solid fa-chevron-down text-xs';
        }
    }

    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
        window.MathJax.typesetPromise();
    }
}

/**
 * Toggle Dataset Schema & Glossary Container manually
 */
function toggleDatasetSchema() {
    const schemaContainer = document.getElementById('dataset-schema-container');
    const toggleBtnText = document.getElementById('schema-toggle-text');
    const toggleBtnIcon = document.getElementById('schema-toggle-icon');
    if (schemaContainer) {
        const isHidden = schemaContainer.classList.contains('hidden');
        if (isHidden) {
            schemaContainer.classList.remove('hidden');
            if (toggleBtnText) toggleBtnText.textContent = 'Ẩn Chi Tiết Dữ Liệu & Thuật Ngữ';
            if (toggleBtnIcon) toggleBtnIcon.className = 'fa-solid fa-chevron-up text-xs';
        } else {
            schemaContainer.classList.add('hidden');
            if (toggleBtnText) toggleBtnText.textContent = 'Xem Chi Tiết 9 Cột Dữ Liệu & Thuật Ngữ OHLCV';
            if (toggleBtnIcon) toggleBtnIcon.className = 'fa-solid fa-chevron-down text-xs';
        }
    }
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
 * View Mode Toggle Handler (Executive vs Technical)
 */
function toggleViewMode(mode) {
    const execBtn = document.getElementById('btn-mode-executive');
    const techBtn = document.getElementById('btn-mode-technical');
    
    if (mode === 'executive') {
        document.querySelectorAll('.tech-only').forEach(el => el.classList.add('hidden'));
        document.querySelectorAll('.exec-only').forEach(el => el.classList.remove('hidden'));
        
        if (execBtn && techBtn) {
            execBtn.classList.add('bg-indigo-600', 'text-white', 'shadow-md');
            execBtn.classList.remove('text-slate-400', 'hover:text-white');
            techBtn.classList.remove('bg-indigo-600', 'text-white', 'shadow-md');
            techBtn.classList.add('text-slate-400', 'hover:text-white');
        }
    } else {
        document.querySelectorAll('.tech-only').forEach(el => el.classList.remove('hidden'));
        document.querySelectorAll('.exec-only').forEach(el => el.classList.add('hidden'));
        
        if (execBtn && techBtn) {
            techBtn.classList.add('bg-indigo-600', 'text-white', 'shadow-md');
            techBtn.classList.remove('text-slate-400', 'hover:text-white');
            execBtn.classList.remove('bg-indigo-600', 'text-white', 'shadow-md');
            execBtn.classList.add('text-slate-400', 'hover:text-white');
        }
    }
}

/**
 * Quick Preview Modal Component
 */
const MODAL_DATA = {
    signal: {
        title: 'Đặc Trưng Signal & Phân Phối Return 1m',
        img: 'reports/figures/01_returns_distribution.png',
        desc: 'Phân phối tỷ suất lợi nhuận 1m (close-to-close) có đỉnh cực nhọn (Kurtosis = 65.49) và đuôi béo nặng được khớp chính xác bởi phân phối Student-t (df = 2.665 < 3). Kiểm định Jarque-Bera p = 0.0 bác bỏ hoàn toàn giả thuyết phân phối chuẩn.',
        tab: 'task1'
    },
    predictive: {
        title: 'Hiệu Năng Mô Hình XGBoost & Calibration',
        img: 'reports/figures/02_roc_pr_calibration.png',
        desc: 'Mô hình XGBoost dự báo bùng nổ biến động 15m đạt ROC-AUC 0.748 và PR-AUC 0.525 out-of-fold từ kiểm lỗi chéo Purged & Embargoed Time-Series CV 5-Fold. Hiệu chỉnh xác suất Isotonic Calibration giảm Brier Score xuống 0.124.',
        tab: 'task2'
    },
    deepdive: {
        title: 'Tác Động Giá Kyle\'s Lambda & Bootstrap 95% CI',
        img: 'reports/figures/03_kyles_lambda_bootstrap.png',
        desc: 'Hệ số tác động giá Kyle\'s Lambda tăng vọt 4.4 lần ở Regime High Volatility (lambda_high = 0.00185 vs lambda_low = 0.00042). Phương pháp Block Bootstrap 1,000 lượt khẳng định ý nghĩa thống kê p = 0.0001.',
        tab: 'task3'
    },
    schema: {
        title: 'Cấu Trúc Ma Trận Feature & Data Quality',
        img: 'reports/figures/01_returns_qqplot.png',
        desc: 'Tập dữ liệu 264,961 bản ghi OHLCV 1m H2 2024 được làm sạch 100% (0% gap time-series). Trích xuất 6 đặc trưng vi mô domain-informed: Parkinson Vol 15m, Garman-Klass Vol 15m, OFI Ratio, Trade Density, Volume Spike Z-score 60m, Return Momentum 15m.',
        tab: 'docs'
    }
};

function openPreviewModal(type) {
    const data = MODAL_DATA[type];
    if (!data) return;

    const modal = document.getElementById('preview-modal');
    const titleEl = document.getElementById('modal-title-text');
    const imgEl = document.getElementById('modal-img');
    const descEl = document.getElementById('modal-desc');
    const jumpBtn = document.getElementById('modal-jump-btn');

    if (modal && titleEl && descEl && jumpBtn) {
        titleEl.textContent = data.title;
        descEl.textContent = data.desc;
        
        if (imgEl && data.img) {
            imgEl.src = data.img;
            imgEl.classList.remove('hidden');
        } else if (imgEl) {
            imgEl.classList.add('hidden');
        }

        jumpBtn.onclick = () => {
            closePreviewModal();
            switchTab(data.tab);
        };

        modal.classList.remove('hidden');
    }
}

function closePreviewModal() {
    const modal = document.getElementById('preview-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

