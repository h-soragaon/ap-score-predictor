/* AP Score Predictor — Static Site Frontend Logic */

(function () {
  'use strict';

  const courseSelect = document.getElementById('course-select');
  const inputSection = document.getElementById('input-section');
  const resultsSection = document.getElementById('results-section');
  const scoreForm = document.getElementById('score-form');
  const mcqRange = document.getElementById('mcq-range');
  const mcqInput = document.getElementById('mcq-input');
  const mcqMeta = document.getElementById('mcq-meta');
  const frqContainer = document.getElementById('frq-container');
  const predictBtn = document.getElementById('predict-btn');

  let currentConfig = null;
  let currentKey = null;

  // ── Data stores (loaded from static JSON) ──────

  let courseConfigs = null;
  let constrainedPriors = null;
  let scoringStatistics = null;

  async function loadData() {
    try {
      const [configResp, priorsResp, statsResp] = await Promise.all([
        fetch('data/course_config.json'),
        fetch('data/constrained_priors.json'),
        fetch('data/scoring_statistics.json'),
      ]);
      courseConfigs = await configResp.json();
      constrainedPriors = await priorsResp.json();
      scoringStatistics = await statsResp.json();
    } catch (e) {
      console.error('Failed to load data:', e);
      throw e;
    }
  }

  // ── Load Courses ──────────────────────────────

  async function loadCourses() {
    try {
      await loadData();

      const courses = Object.entries(courseConfigs)
        .map(([key, cfg]) => ({ key, title: cfg.title }))
        .sort((a, b) => a.key.localeCompare(b.key));

      courseSelect.innerHTML = '<option value="">Choose an AP course\u2026</option>';
      courses.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.key;
        opt.textContent = c.title;
        courseSelect.appendChild(opt);
      });
      courseSelect.disabled = false;
    } catch (e) {
      courseSelect.innerHTML = '<option value="">Failed to load courses</option>';
    }
  }

  // ── Course Selection ──────────────────────────

  courseSelect.addEventListener('change', () => {
    const key = courseSelect.value;
    if (!key) {
      inputSection.style.display = 'none';
      resultsSection.style.display = 'none';
      return;
    }

    currentKey = key;
    currentConfig = courseConfigs[key];
    currentConfig.key = key;
    buildForm(currentConfig);
    inputSection.style.display = '';
    resultsSection.style.display = 'none';
    inputSection.style.animation = 'none';
    inputSection.offsetHeight; // reflow
    inputSection.style.animation = '';
  });

  // ── Build Dynamic Form ────────────────────────

  function buildForm(config) {
    // MCQ setup
    mcqRange.max = config.mcq_total;
    mcqInput.max = config.mcq_total;
    const midVal = Math.round(config.mcq_total * 0.6);
    mcqRange.value = midVal;
    mcqInput.value = midVal;
    mcqMeta.textContent = `out of ${config.mcq_total}  \u00b7  weight ${(config.mcq_weight * 100).toFixed(0)}%`;

    // FRQ sections
    frqContainer.innerHTML = '';

    config.frq_sections.forEach(section => {
      const sectionDiv = document.createElement('div');
      sectionDiv.className = 'frq-section';

      const sectionName = formatSectionName(section.name);
      sectionDiv.innerHTML = `
        <div class="frq-section-header">
          <span>${sectionName}</span>
          <span class="frq-section-weight">weight ${(section.weight * 100).toFixed(0)}%</span>
        </div>
        <div class="frq-grid"></div>
      `;

      const grid = sectionDiv.querySelector('.frq-grid');

      section.question_max.forEach((max, i) => {
        const item = document.createElement('div');
        item.className = 'frq-item';

        const qLabel = section.question_max.length === 1
          ? sectionName
          : `Q${i + 1}`;

        item.innerHTML = `
          <label>${qLabel} <strong>/ ${max}</strong></label>
          <input type="number"
            class="frq-q-input"
            data-section="${section.name}"
            data-index="${i}"
            min="0"
            max="${max}"
            value="${Math.round(max * 0.6)}"
            step="1">
        `;
        grid.appendChild(item);
      });

      frqContainer.appendChild(sectionDiv);
    });
  }

  function formatSectionName(name) {
    const map = {
      'frq': 'Free Response',
      'essays': 'Essays',
      'written_frq': 'Written FRQ',
      'sight_singing': 'Sight Singing',
      'saq': 'Short Answer (SAQ)',
      'dbq': 'Document-Based (DBQ)',
      'leq': 'Long Essay (LEQ)',
    };
    return map[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  // ── Sync Range ↔ Number ───────────────────────

  mcqRange.addEventListener('input', () => {
    mcqInput.value = mcqRange.value;
  });

  mcqInput.addEventListener('input', () => {
    mcqRange.value = mcqInput.value;
  });

  // ── Form Submission ───────────────────────────

  scoreForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!currentConfig || !currentKey) return;

    const mcqCorrect = parseInt(mcqInput.value, 10);

    // Gather FRQ scores by section
    const frqSections = {};
    document.querySelectorAll('.frq-q-input').forEach(input => {
      const section = input.dataset.section;
      const idx = parseInt(input.dataset.index, 10);
      if (!frqSections[section]) frqSections[section] = [];
      frqSections[section][idx] = parseFloat(input.value) || 0;
    });

    // Look up priors: try year-specific first, fall back to base
    const examYear = 2025;
    const priorsKey = `${currentKey}_${examYear}`;
    const priors = constrainedPriors[priorsKey] || constrainedPriors[currentKey];

    if (!priors) {
      alert('No priors available for this course.');
      return;
    }

    // Look up scoring statistics (may be null)
    const statsKey = `${currentKey}_${examYear}`;
    const stats = scoringStatistics[statsKey] || null;

    // Run prediction locally
    const result = Predictor.predict(
      { mcqCorrect, frqSections },
      currentConfig,
      priors,
      stats,
    );

    displayResults(result);
  });

  // ── Display Results ───────────────────────────

  function displayResults(result) {
    resultsSection.style.display = '';
    resultsSection.style.animation = 'none';
    resultsSection.offsetHeight;
    resultsSection.style.animation = '';

    // Hero
    document.getElementById('hero-score').textContent = result.most_likely_score;
    document.getElementById('hero-expected').textContent =
      `Expected value: ${result.expected_score.toFixed(2)}`;

    // Probability chart
    const chart = document.getElementById('prob-chart');
    chart.innerHTML = '';

    const probs = result.predicted_distribution;
    const maxProb = Math.max(...Object.values(probs));
    const scoreLabels = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'];

    for (let k = 1; k <= 5; k++) {
      const prob = probs[String(k)];
      const pct = (prob * 100).toFixed(1);
      const isMax = Math.abs(prob - maxProb) < 0.0001;

      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-label">${scoreLabels[k - 1]}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill score-${k} ${isMax ? 'is-max' : ''}"
               style="width: 0%"
               data-width="${Math.max(prob * 100, 0.5)}%"></div>
        </div>
        <span class="prob-value">${pct}%</span>
      `;
      chart.appendChild(row);
    }

    // Animate bars after a tick
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        chart.querySelectorAll('.prob-bar-fill').forEach(bar => {
          bar.style.width = bar.dataset.width;
        });
      });
    });

    // Stats
    document.getElementById('stat-composite').textContent =
      (result.weighted_composite * 100).toFixed(1) + '%';
    document.getElementById('stat-difficulty').textContent =
      (result.difficulty_adjustment >= 0 ? '+' : '') +
      (result.difficulty_adjustment * 100).toFixed(2) + '%';
    document.getElementById('stat-confidence').textContent =
      result.confidence_band.p10 + ' \u2013 ' + result.confidence_band.p90;

    // Explanations
    const explDiv = document.getElementById('explanations');
    explDiv.innerHTML = '';
    result.explanations.forEach(text => {
      const item = document.createElement('div');
      item.className = 'explanation-item';
      item.textContent = text;
      explDiv.appendChild(item);
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Init ──────────────────────────────────────

  loadCourses();
})();
