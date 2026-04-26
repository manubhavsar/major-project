const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorBox = document.getElementById('error-box');
const resetBtn = document.getElementById('reset-btn');
const downloadBtn = document.getElementById('download-btn');
const overallBadge = document.getElementById('overall-badge');
const transcriptBox = document.getElementById('transcript-box');
const wordCountMsg = document.getElementById('word-count');
const mediaPlayerBox = document.getElementById('media-player-box');

// ─── Drag & Drop ───
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
    uploadZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false)
);
['dragenter', 'dragover'].forEach(ev =>
    uploadZone.addEventListener(ev, () => uploadZone.classList.add('dragover'), false)
);
['dragleave', 'drop'].forEach(ev =>
    uploadZone.addEventListener(ev, () => uploadZone.classList.remove('dragover'), false)
);
uploadZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));

function handleFiles(files) {
    if (files.length > 0) uploadFile(files[0]);
}

function scoreClass(level) {
    if (!level) return 'score-unknown';
    const l = level.toLowerCase();
    if (l === 'low') return 'score-low';
    if (l === 'medium') return 'score-medium';
    if (l === 'high') return 'score-high';
    return 'score-unknown';
}

// ─── Render Pipeline Card ───
function renderPipeline(data, prefix) {
    const cb = document.getElementById(`${prefix}-consensus`);
    cb.textContent = data.consensus;
    cb.className = `consensus-badge ${scoreClass(data.consensus)}`;

    const fvContainer = document.getElementById(`${prefix}-feature-values`);
    fvContainer.innerHTML = '';
    const fv = data.feature_values || {};
    Object.entries(fv).forEach(([label, value]) => {
        const div = document.createElement('div');
        div.className = 'feat-item';
        div.innerHTML = `<div class="feat-label">${label}</div><div class="feat-value">${value}</div>`;
        fvContainer.appendChild(div);
    });

    const tbody = document.getElementById(`${prefix}-models-tbody`);
    tbody.innerHTML = '';
    (data.models || []).forEach(m => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${m.model}</td>
            <td><span class="badge ${scoreClass(m.prediction)}">${m.prediction}</span></td>
            <td class="conf-cell">${m.confidence || '0.0%'}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ─── Render Gemini AI Coach ───
function renderGeminiReport(report) {
    const section = document.getElementById('gemini-section');
    section.style.display = 'block'; // Always show

    if (!report || report.engagement_rating === 'N/A' || report.engagement_rating === 'Error') {
        document.getElementById('gemini-rating').textContent = 'Error';
        document.getElementById('gemini-summary').innerHTML = `<span style="color:red;">${report?.summary || 'AI Report failed to generate. Check console/API key.'}</span>`;
        return;
    }

    const rBadge = document.getElementById('gemini-rating');
    rBadge.textContent = report.engagement_rating;
    rBadge.className = `consensus-badge ${scoreClass(report.engagement_rating)}`;

    document.getElementById('gemini-summary').textContent = report.summary;
    document.getElementById('gemini-confidence').textContent = report.confidence_score;
    document.getElementById('gemini-fillers').textContent = report.filler_analysis;

    const renderList = (id, items) => {
        const ul = document.getElementById(id);
        ul.innerHTML = '';
        (items || []).forEach(it => { const li = document.createElement('li'); li.textContent = it; ul.appendChild(li); });
    };

    renderList('gemini-strengths', report.strengths);
    renderList('gemini-improvements', report.improvements);
    renderList('gemini-tips', report.coaching_tips);
}

// ─── Inject Media Player ───
function injectMedia(file) {
    mediaPlayerBox.innerHTML = '';
    const url = URL.createObjectURL(file);
    const type = file.type;

    if (type.startsWith('video/')) {
        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.autoplay = false;
        mediaPlayerBox.appendChild(video);
    } else if (type.startsWith('audio/')) {
        const audio = document.createElement('audio');
        audio.src = url;
        audio.controls = true;
        audio.autoplay = false;
        mediaPlayerBox.appendChild(audio);
    } else {
        mediaPlayerBox.innerHTML = '<p style="color:white; font-size:0.8rem;">Media preview unavailable</p>';
    }
}

let currentResults = null;

// ─── Upload Process ───
async function uploadFile(file) {
    uploadZone.classList.add('hidden');
    results.classList.add('hidden');
    errorBox.classList.add('hidden');
    loading.classList.remove('hidden');
    resetBtn.classList.add('hidden');
    downloadBtn.classList.add('hidden');

    const loadingTexts = [
        "Transcribing audio with Whisper...",
        "Extracting 29 acoustic variables...",
        "Computing linguistic structure...",
        "Running 10 ML model predictions...",
        "Consulting Gemini AI coach..."
    ];
    let idx = 0;
    const interval = setInterval(() => {
        idx = (idx + 1) % loadingTexts.length;
        document.getElementById('loading-text').textContent = loadingTexts[idx];
    }, 2500);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        clearInterval(interval);
        loading.classList.add('hidden');

        if (data.success) {
            results.classList.remove('hidden');
            resetBtn.classList.remove('hidden');
            downloadBtn.classList.remove('hidden');

            const r = data.results;
            currentResults = r;
            
            overallBadge.textContent = r.overall_engagement;
            overallBadge.className = `master-badge ${scoreClass(r.overall_engagement)}`;
            transcriptBox.textContent = r.transcript;
            
            const words = r.transcript.split(/\s+/).filter(w => w.length > 0).length;
            wordCountMsg.textContent = `(${words} words)`;

            injectMedia(file);
            renderPipeline(r.text_pipeline, 'text');
            renderPipeline(r.audio_pipeline, 'audio');
            renderGeminiReport(r.gemini_report);
        } else {
            errorBox.textContent = data.error || 'Processing error';
            errorBox.classList.remove('hidden');
            uploadZone.classList.remove('hidden');
        }
    } catch (err) {
        clearInterval(interval);
        loading.classList.add('hidden');
        errorBox.textContent = 'Network communication failed';
        errorBox.classList.remove('hidden');
        uploadZone.classList.remove('hidden');
    }
}

// ─── Evaluation Tabs ───
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        
        btn.classList.add('active');
        const target = `tab-${btn.dataset.tab}`;
        document.getElementById(target).classList.remove('hidden');
    });
});

// ─── New Analysis ───
resetBtn.addEventListener('click', () => {
    results.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    resetBtn.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    fileInput.value = '';
    mediaPlayerBox.innerHTML = '';
});

// ─── Download Logic ───
downloadBtn.addEventListener('click', () => {
    if (!currentResults) return;
    let report = "ENGAGEMENT REPORT\n================\n";
    report += `Overall: ${currentResults.overall_engagement}\n\n`;
    report += `Transcript:\n${currentResults.transcript}\n\n`;
    
    if (currentResults.gemini_report) {
        const g = currentResults.gemini_report;
        report += `AI COACH SUMMARY:\n${g.summary}\n\n`;
        report += `Rating: ${g.engagement_rating} | Confidence: ${g.confidence_score}\n\n`;
        report += "STRENGTHS:\n" + (g.strengths || []).map(s => `- ${s}`).join('\n') + "\n\n";
        report += "TIPS:\n" + (g.coaching_tips || []).map(s => `- ${s}`).join('\n') + "\n";
    }

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'engagement-report.txt';
    a.click();
});
