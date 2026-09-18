// ---------------- Scroll-reveal ----------------
const revealEls = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
revealEls.forEach(el => revealObserver.observe(el));

// ---------------- Animated hero stat counters ----------------
function animateCount(el) {
  const target = parseFloat(el.dataset.countTo);
  const suffix = el.dataset.suffix || '';
  const isDecimal = target % 1 !== 0;
  const duration = 1400;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = (isDecimal ? value.toFixed(2) : Math.round(value)) + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
document.querySelectorAll('[data-count-to]').forEach(animateCount);

// ---------------- Active nav link on scroll ----------------
const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
const navSections = ['about', 'architecture', 'performance', 'analyze']
  .map(id => document.getElementById(id))
  .filter(Boolean);

const navObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      navLinks.forEach(link => {
        link.classList.toggle('nav-active', link.getAttribute('href') === `#${id}`);
      });
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });
navSections.forEach(sec => navObserver.observe(sec));

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadError = document.getElementById('upload-error');

const uploadView = document.getElementById('upload-view');
const scanView = document.getElementById('scan-view');
const resultsView = document.getElementById('results-view');

const scanSlideName = document.getElementById('scan-slide-name');
const scanPhaseLabel = document.getElementById('scan-phase-label');
const scanGrid = document.getElementById('scan-grid');
const statProcessed = document.getElementById('stat-processed');
const statTotal = document.getElementById('stat-total');

const dominantStageValue = document.getElementById('dominant-stage-value');
const dominantStageSub = document.getElementById('dominant-stage-sub');
const slideCallValue = document.getElementById('slide-call-value');
const recommendationText = document.getElementById('recommendation-text');
const fractionRingFill = document.getElementById('fraction-ring-fill');
const fractionValue = document.getElementById('fraction-value');
const fractionCaption = document.getElementById('fraction-caption');
const distributionBars = document.getElementById('distribution-bars');
const resetBtn = document.getElementById('reset-btn');

const STAGE_LABELS = {
  stage0: 'Stage 0 · Mucus tissue',
  stage1: 'Stage 1 · Stroma tissue',
  stage2: 'Stage 2 · Muscle tissue',
  stage3: 'Stage 3 · Adipose tissue',
  unidentified: 'Unidentified',
};

const STAGE_COLORS = {
  stage0: '#2FBF9F',
  stage1: '#3EA6D8',
  stage2: '#E0A63D',
  stage3: '#E2574C',
  unidentified: '#5B6472',
};

const RING_CIRCUMFERENCE = 2 * Math.PI * 52;

const TOTAL_CELLS = 240;
let filledCells = 0;
let pollTimer = null;

function showView(view) {
  [uploadView, scanView, resultsView].forEach(v => v.classList.remove('active'));
  view.classList.add('active');
}

function buildGrid() {
  scanGrid.innerHTML = '';
  for (let i = 0; i < TOTAL_CELLS; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    scanGrid.appendChild(cell);
  }
  filledCells = 0;
}

function updateGrid(processed, total, normalCount, tumorCount) {
  if (!total) return;
  const targetFilled = Math.min(TOTAL_CELLS, Math.round((processed / total) * TOTAL_CELLS));
  const cells = scanGrid.children;
  const processedSoFar = normalCount + tumorCount || 1;
  const normalRatio = normalCount / processedSoFar;

  for (let i = filledCells; i < targetFilled; i++) {
    const isNormal = Math.random() < normalRatio;
    cells[i].classList.add(isNormal ? 'filled-normal' : 'filled-tumor');
  }
  filledCells = targetFilled;
}

async function uploadFile(file) {
  uploadError.hidden = true;

  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!['.svs', '.tif', '.tiff', '.ndpi'].includes(ext)) {
    uploadError.textContent = `Unsupported file type "${ext}". Upload a .svs, .tif, .tiff, or .ndpi whole slide image.`;
    uploadError.hidden = false;
    return;
  }

  scanSlideName.textContent = file.name;
  scanPhaseLabel.textContent = 'Uploading slide\u2026';
  buildGrid();
  statProcessed.textContent = '0';
  statTotal.textContent = '0';
  showView(scanView);

  const formData = new FormData();
  formData.append('slide', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      showUploadError(data.error || 'Upload failed.');
      return;
    }

    pollStatus(data.job_id);
  } catch (err) {
    showUploadError('Could not reach the analysis server. Is app.py running?');
  }
}

function showUploadError(message) {
  showView(uploadView);
  uploadError.textContent = message;
  uploadError.hidden = false;
}

function pollStatus(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      const state = await res.json();

      if (state.phase === 'tiling') {
        scanPhaseLabel.textContent = 'Tiling slide and filtering background\u2026';
      } else if (state.phase === 'classifying') {
        scanPhaseLabel.textContent = 'Classifying tissue patches\u2026';
        statTotal.textContent = state.total_patches;
        statProcessed.textContent = state.processed;
        updateGrid(state.processed, state.total_patches, state.normal_count, state.tumor_count);
      } else if (state.phase === 'done') {
        clearInterval(pollTimer);
        renderResults(state.result);
      } else if (state.phase === 'error') {
        clearInterval(pollTimer);
        showUploadError(state.error || 'Analysis failed.');
      }
    } catch (err) {
      clearInterval(pollTimer);
      showUploadError('Lost connection to the analysis server.');
    }
  }, 1000);
}

function renderResults(result) {
  const deepest = result.deepest_invasion_stage;
  if (deepest) {
    const baseLabel = STAGE_LABELS[deepest];
    dominantStageValue.textContent = result.invasion_beyond_flag
      ? `${baseLabel} and beyond`
      : baseLabel;
  } else {
    dominantStageValue.textContent = 'None (no tumor detected)';
  }
  dominantStageSub.textContent = result.tumor_patches > 0
    ? `${result.tumor_patches} of ${result.total_patches} patches classified as tumor`
    : 'No tumor patches were detected on this slide.';
  if (result.invasion_beyond_flag) {
    dominantStageSub.textContent += ' — a large share of tumor patches were unidentified, so true depth may exceed this stage.';
  }

  slideCallValue.textContent = result.slide_level_call;
  slideCallValue.className = 'slide-call ' + (result.slide_level_call === 'TUMOR DETECTED' ? 'tumor' : 'normal');
  recommendationText.textContent = result.recommendation;

  const pct = Math.round(result.tumor_fraction * 100);
  fractionValue.textContent = `${pct}%`;
  const offset = RING_CIRCUMFERENCE - (result.tumor_fraction * RING_CIRCUMFERENCE);
  fractionRingFill.style.strokeDashoffset = offset;
  fractionRingFill.style.stroke = pct > 40 ? '#E2574C' : (pct > 10 ? '#E0A63D' : '#2FBF9F');
  fractionCaption.textContent = `${result.tumor_patches} tumor / ${result.normal_patches} normal patches`;

  distributionBars.innerHTML = '';
  const maxCount = Math.max(1, ...Object.values(result.stage_distribution));
  ['stage0', 'stage1', 'stage2', 'stage3', 'unidentified'].forEach(key => {
    const count = result.stage_distribution[key] || 0;
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${STAGE_LABELS[key]}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(count / maxCount) * 100}%; background:${STAGE_COLORS[key]}"></div></div>
      <div class="bar-count">${count}</div>
    `;
    distributionBars.appendChild(row);
  });

  showView(resultsView);
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});

resetBtn.addEventListener('click', () => {
  fileInput.value = '';
  showView(uploadView);
});
