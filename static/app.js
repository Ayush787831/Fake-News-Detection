/**
 * TruthLens Fake News Detector - Frontend Application Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabButtons = document.querySelectorAll('.nav-tab');
    const viewPanels = document.querySelectorAll('.view-panel');
    
    const newsInput = document.getElementById('news-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    
    const detectionForm = document.getElementById('detection-form');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnSpinner = btnAnalyze.querySelector('.btn-spinner');
    
    const btnPaste = document.getElementById('btn-paste');
    const btnClear = document.getElementById('btn-clear');
    
    const resultEmptyState = document.getElementById('result-empty-state');
    const resultContent = document.getElementById('result-content');
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-description');
    const confidenceVal = document.getElementById('confidence-value');
    
    const truthPercent = document.getElementById('truth-percent');
    const truthBar = document.getElementById('truth-bar');
    const fakePercent = document.getElementById('fake-percent');
    const fakeBar = document.getElementById('fake-bar');
    
    const keywordsContainer = document.getElementById('keywords-container');
    const metaTimestamp = document.getElementById('meta-timestamp');
    const metaWords = document.getElementById('meta-words');
    const btnCopyResult = document.getElementById('btn-copy-result');
    
    const historyContainer = document.getElementById('history-container');
    const btnClearHistory = document.getElementById('btn-clear-history');
    const toast = document.getElementById('toast');
    
    // In-memory / LocalStorage history
    let predictionHistory = JSON.parse(localStorage.getItem('truthlens_history') || '[]');
    let currentResultData = null;

    // --- Tab Switching ---
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.getAttribute('data-target');
            
            tabButtons.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            viewPanels.forEach(p => p.classList.remove('active'));

            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');
            
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });

    // --- Text Counters ---
    function updateCounters() {
        const text = newsInput.value || '';
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        
        charCounter.textContent = `${chars} chars`;
        wordCounter.textContent = `${words} words`;
    }

    newsInput.addEventListener('input', updateCounters);
    updateCounters();

    // --- Presets Click Handler ---
    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const text = chip.getAttribute('data-text');
            newsInput.value = text;
            updateCounters();
            showToast('Preset loaded! Click Analyze to test.');
            newsInput.focus();
        });
    });

    // --- Paste from Clipboard ---
    btnPaste.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            if (text) {
                newsInput.value = text;
                updateCounters();
                showToast('Pasted from clipboard!');
            }
        } catch (err) {
            newsInput.focus();
            showToast('Clipboard access unavailable. Please use Ctrl+V / Cmd+V.');
        }
    });

    // --- Clear Input ---
    btnClear.addEventListener('click', () => {
        newsInput.value = '';
        updateCounters();
        newsInput.focus();
    });

    // --- Submit Analysis Form ---
    detectionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = newsInput.value.trim();
        if (!text) {
            showToast('Please enter or paste a statement to analyze.');
            newsInput.focus();
            return;
        }

        // Show Loading State
        btnAnalyze.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnSpinner.classList.remove('hidden');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({ news: text })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Failed to process statement.');
            }

            // Render Results
            renderResults(data);
            
            // Save to History
            saveToHistory(data);

        } catch (err) {
            console.error('Analysis error:', err);
            showToast(`Error: ${err.message}`);
        } finally {
            // Reset Loading State
            btnAnalyze.disabled = false;
            btnText.textContent = 'Analyze Statement';
            btnSpinner.classList.add('hidden');
        }
    });

    // --- Render Results ---
    function renderResults(data) {
        currentResultData = data;

        // Hide empty state, show content
        resultEmptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');

        // Verdict Banner
        verdictBanner.classList.remove('real-verdict', 'fake-verdict');
        if (data.is_real) {
            verdictBanner.classList.add('real-verdict');
            verdictIcon.className = 'fa-solid fa-circle-check';
            verdictTitle.textContent = 'VERIFIED REAL NEWS';
            verdictDesc.textContent = 'Linguistic patterns match verified factual reporting from the dataset.';
        } else {
            verdictBanner.classList.add('fake-verdict');
            verdictIcon.className = 'fa-solid fa-triangle-exclamation';
            verdictTitle.textContent = 'FLAGGED AS FAKE / MISLEADING';
            verdictDesc.textContent = 'High resemblance to false claims, misinformation, or sensational rhetoric.';
        }

        confidenceVal.textContent = `${data.confidence_score}%`;

        // Probability Meters
        truthPercent.textContent = `${data.truth_probability}%`;
        fakePercent.textContent = `${data.fake_probability}%`;

        // Trigger animation
        truthBar.style.width = '0%';
        fakeBar.style.width = '0%';
        setTimeout(() => {
            truthBar.style.width = `${data.truth_probability}%`;
            fakeBar.style.width = `${data.fake_probability}%`;
        }, 50);

        // Keywords Cloud (Explainable AI)
        keywordsContainer.innerHTML = '';
        if (data.explanations && data.explanations.length > 0) {
            data.explanations.forEach(item => {
                const pill = document.createElement('span');
                const isRealWord = item.direction === 'real';
                pill.className = `keyword-pill ${isRealWord ? 'pill-real' : 'pill-fake'}`;
                pill.innerHTML = `
                    <i class="fa-solid ${isRealWord ? 'fa-arrow-up-right-dots' : 'fa-arrow-down-right-dots'}"></i>
                    <span>"${item.word}"</span>
                    <span class="keyword-weight">${item.weight > 0 ? '+' : ''}${item.weight}</span>
                `;
                keywordsContainer.appendChild(pill);
            });
        } else {
            keywordsContainer.innerHTML = '<span class="text-dim" style="font-size: 0.8rem;">No high-frequency benchmark n-grams detected in this vocabulary.</span>';
        }

        // Meta
        metaTimestamp.textContent = data.timestamp || new Date().toLocaleTimeString();
        metaWords.textContent = data.metrics?.word_count || (newsInput.value.split(/\s+/).length);

        // Scroll result card into view on small screens
        if (window.innerWidth < 960) {
            resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // --- History Management ---
    function saveToHistory(item) {
        // Prepend to list
        predictionHistory.unshift({
            statement: item.statement,
            is_real: item.is_real,
            prediction: item.prediction,
            confidence: item.confidence_score,
            timestamp: item.timestamp
        });

        // Limit to 20 items
        if (predictionHistory.length > 20) {
            predictionHistory.pop();
        }

        localStorage.setItem('truthlens_history', JSON.stringify(predictionHistory));
        renderHistory();
    }

    function renderHistory() {
        if (!predictionHistory || predictionHistory.length === 0) {
            historyContainer.innerHTML = '<div class="history-empty">No statements tested yet in this session.</div>';
            return;
        }

        historyContainer.innerHTML = '';
        predictionHistory.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = `
                <div class="history-text" title="${escapeHtml(item.statement)}">
                    "${escapeHtml(item.statement)}"
                </div>
                <span class="history-badge ${item.is_real ? 'badge-real' : 'badge-fake'}">
                    ${item.is_real ? 'REAL' : 'FAKE'} (${item.confidence}%)
                </span>
                <button type="button" class="btn btn-secondary btn-sm history-reload-btn" data-index="${index}" title="Re-test this statement">
                    <i class="fa-solid fa-arrow-rotate-right"></i>
                </button>
            `;
            historyContainer.appendChild(div);
        });

        // Attach reload listeners
        document.querySelectorAll('.history-reload-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.getAttribute('data-index'), 10);
                const item = predictionHistory[idx];
                if (item) {
                    newsInput.value = item.statement;
                    updateCounters();
                    detectionForm.dispatchEvent(new Event('submit'));
                }
            });
        });
    }

    btnClearHistory.addEventListener('click', () => {
        predictionHistory = [];
        localStorage.removeItem('truthlens_history');
        renderHistory();
        showToast('History cleared.');
    });

    // --- Copy Result Report ---
    btnCopyResult.addEventListener('click', () => {
        if (!currentResultData) return;
        const report = `TruthLens Fake News Analysis Report:
Statement: "${currentResultData.statement}"
Verdict: ${currentResultData.prediction}
Confidence: ${currentResultData.confidence_score}%
Truth Probability: ${currentResultData.truth_probability}%
Fake Probability: ${currentResultData.fake_probability}%
Timestamp: ${currentResultData.timestamp}`;

        navigator.clipboard.writeText(report).then(() => {
            showToast('Analysis report copied to clipboard!');
        }).catch(() => {
            showToast('Failed to copy report.');
        });
    });

    // --- Toast Notification ---
    let toastTimeout = null;
    function showToast(msg) {
        if (toastTimeout) clearTimeout(toastTimeout);
        toast.textContent = msg;
        toast.classList.remove('hidden');
        toastTimeout = setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }

    function escapeHtml(string) {
        return String(string)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // Initial history render
    renderHistory();
});
