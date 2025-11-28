// static/js/main.js
// Safe, resilient frontend script for SmartStudy app.
// - collects form values
// - POSTs to /api/predict and displays results
// - avoids touching non-existent DOM nodes (null-safe)

document.addEventListener('DOMContentLoaded', () => {
  // DOM references (may be null if HTML changed)
  const predictForm = document.getElementById('predictForm');
  const resultCard = document.getElementById('resultCard');
  const predictionText = document.getElementById('predictionText');
  const suggestionsList = document.getElementById('suggestionsList');
  const ttTableBody = document.querySelector('#ttTable tbody');
  const resetBtn = document.getElementById('resetBtn');
  const iqField = document.getElementById('iqField');

  // Utility: clear result area
  function clearResultArea() {
    if (predictionText) predictionText.textContent = '';
    if (suggestionsList) suggestionsList.innerHTML = '';
    if (ttTableBody) ttTableBody.innerHTML = '';
    if (resultCard) resultCard.style.display = 'none';
  }

  // Utility: show error message in result area
  function showError(msg) {
    if (!resultCard) return alert(msg);
    resultCard.style.display = 'block';
    if (predictionText) predictionText.textContent = 'Error: ' + msg;
    if (suggestionsList) suggestionsList.innerHTML = '';
    if (ttTableBody) ttTableBody.innerHTML = '';
  }

  // Convert form data to the JSON shape expected by /api/predict
  function collectFormData(formEl) {
    const fd = new FormData(formEl);
    const obj = {};

    // Subjects and numeric fields
    const numericNames = [
      'kan', 'english', 'hindi', 'maths', 'chem', 'bio_or_cs', 'physics',
      'study_hours', 'time_extra', 'courses', 'attendance'
    ];

    numericNames.forEach(name => {
      if (fd.has(name)) {
        const raw = fd.get(name);
        const val = raw === null || raw === '' ? 0 : Number(raw);
        obj[name] = Number.isFinite(val) ? val : 0;
      }
    });

    // IQ field (from section 1)
    if (iqField && iqField.value !== '') {
      const val = Number(iqField.value);
      obj['iq'] = Number.isFinite(val) ? val : 100;
    } else if (fd.has('iq')) {
      const raw = fd.get('iq');
      const val = raw === null || raw === '' ? 100 : Number(raw);
      obj['iq'] = Number.isFinite(val) ? val : 100;
    } else {
      obj['iq'] = 100;
    }

    // Boolean selects
    function boolFrom(value) {
      if (typeof value === 'boolean') return value;
      if (value === null) return false;
      const s = String(value).toLowerCase();
      return ['yes','true','1','on'].includes(s);
    }

    if (fd.has('extra_activity')) obj['extra_activity'] = boolFrom(fd.get('extra_activity'));
    if (fd.has('attended_contest')) obj['attended_academic'] = boolFrom(fd.get('attended_contest'));
    if (fd.has('attended_contest') === false && fd.has('attended_academic')) obj['attended_academic'] = boolFrom(fd.get('attended_academic'));
    if (fd.has('extra_activity') === false && fd.has('extra')) obj['extra_activity'] = boolFrom(fd.get('extra'));

    const mapping = {
      'kan': 'kan',
      'english': 'english',
      'hindi': 'hindi',
      'maths': 'maths',
      'chem': 'chem',
      'bio_or_cs': 'bio_or_cs',
      'physics': 'physics',
      'study_hours': 'study_hours_per_day',
      'time_extra': 'time_extra',
      'courses': 'courses',
      'attendance': 'attendance'
    };

    const payload = { inputs: {} };
    Object.keys(obj).forEach(k => {
      const mapped = mapping[k] || k;
      payload.inputs[mapped] = obj[k];
    });

    if (typeof payload.inputs.attended_academic === 'undefined') payload.inputs.attended_academic = Boolean(obj.attended_academic || false);
    if (typeof payload.inputs.extra_activity === 'undefined') payload.inputs.extra_activity = Boolean(obj.extra_activity || false);

    return payload;
  }

  // Render suggestions (array of strings) as list items
  function renderSuggestions(suggestions) {
    if (!suggestionsList) return;
    suggestionsList.innerHTML = '';
    suggestions.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      suggestionsList.appendChild(li);
    });
  }

  // Render timetable: supports new format (slot1/slot2/slot3) and legacy 'plan' format
  function renderTimetable(tt) {
    if (!ttTableBody) return;
    ttTableBody.innerHTML = '';

    if (!Array.isArray(tt) || tt.length === 0) {
      ttTableBody.innerHTML = '<tr><td colspan="3">No timetable returned.</td></tr>';
      return;
    }

    // Detect new format: objects with slot1 keys
    const first = tt[0];
    const isNewFormat = first && (typeof first.slot1 !== 'undefined' || typeof first.slot2 !== 'undefined' || typeof first.slot3 !== 'undefined');

    if (isNewFormat) {
      // Expect exactly 7 rows (Monday..Sunday) but handle any length
      tt.forEach(row => {
        const tr = document.createElement('tr');
        const day = document.createElement('td');
        day.textContent = row.day || '-';
        const slot1 = document.createElement('td');
        slot1.textContent = row.slot1 || '-';
        const slot2 = document.createElement('td');
        slot2.textContent = row.slot2 || '-';
        const slot3 = document.createElement('td');
        slot3.textContent = row.slot3 || '-';
        tr.appendChild(day);
        tr.appendChild(slot1);
        tr.appendChild(slot2);
        tr.appendChild(slot3);
        ttTableBody.appendChild(tr);
      });
      return;
    }

    // Legacy format: each item has 'day' and 'plan' array of {subject,hours}
    tt.forEach(dayObj => {
      const day = dayObj.day || '';
      const plan = Array.isArray(dayObj.plan) ? dayObj.plan : [];

      if (plan.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${day}</td><td>-</td><td>-</td>`;
        ttTableBody.appendChild(tr);
      } else {
        // render each plan row as its own table row (legacy behavior)
        plan.forEach((blk, idx) => {
          const tr = document.createElement('tr');
          const subject = blk.subject || blk.sub || '-';
          const hours = (typeof blk.hours !== 'undefined') ? blk.hours : (blk.h || '-');
          tr.innerHTML = `<td>${day}</td><td>${subject}</td><td>${hours}</td>`;
          ttTableBody.appendChild(tr);
        });
      }
    });
  }

  // Submit handler
  async function handlePredict(event) {
    if (event && event.preventDefault) event.preventDefault();
    clearResultArea();

    if (!predictForm) {
      showError('Form not found on page.');
      return;
    }

    // Prepare payload
    let payload;
    try {
      payload = collectFormData(predictForm);
    } catch (err) {
      showError('Failed to read form inputs: ' + err.message);
      return;
    }

    // call backend
    const endpoint = '/api/predict'; // assumed API route
    try {
      // show interim message
      if (resultCard) resultCard.style.display = 'block';
      if (predictionText) predictionText.textContent = 'Predicting...';

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(() => null);
        throw new Error(`Server returned ${resp.status} ${resp.statusText}${txt ? ': ' + txt : ''}`);
      }

      const json = await resp.json();
      if (!json || json.status !== 'ok' || !json.result) {
        // some servers may return direct result without wrapper
        const fallback = json.result || json;
        if (!fallback) throw new Error('Unexpected response format');
        displayResult(fallback);
        return;
      }
      displayResult(json.result);
    } catch (err) {
      showError(err.message || String(err));
    }
  }

  // Display result object structure { prediction, probability, suggestions, timetable, inputs, ... }
  function displayResult(res) {
    if (!resultCard) return;
    resultCard.style.display = 'block';

    // Prediction text
    let predLabel = res.prediction || res.pred || 'N/A';
    let prob = (typeof res.probability !== 'undefined') ? res.probability : res.prob || res.probability_estimate;
    if (typeof prob === 'number') {
      prob = Math.round(prob * 100) / 100;
    }
    const conf = res.confidence ? ` • Confidence: ${res.confidence}` : '';

    if (predictionText) {
      predictionText.textContent = `${predLabel}${(prob ? ' — ' + prob : '')}${conf}`;
    }

    // Suggestions
    if (res.suggestions && Array.isArray(res.suggestions)) {
      renderSuggestions(res.suggestions);
    } else if (res.suggestion && Array.isArray(res.suggestion)) {
      renderSuggestions(res.suggestion);
    } else {
      if (suggestionsList) suggestionsList.innerHTML = '<li>No suggestions provided.</li>';
    }

    // Timetable (supports new slot1..slot3 format or legacy plan)
    const tt = res.timetable || res.timetable_blocks || res.timetable_plan || null;
    if (tt) renderTimetable(tt);
    else if (ttTableBody) ttTableBody.innerHTML = '<tr><td colspan="3">No timetable returned.</td></tr>';
  }

  // Attach event listeners safely
  if (predictForm) {
    predictForm.addEventListener('submit', handlePredict);
  } else {
    // If predictForm not present, try to attach to a button with class .btn.primary
    const altBtn = document.querySelector('button.btn.primary');
    if (altBtn) altBtn.addEventListener('click', handlePredict);
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (predictForm) predictForm.reset();
      if (iqField) iqField.value = '100';
      clearResultArea();
    });
  }

  // initial clear
  clearResultArea();
});
