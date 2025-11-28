from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import requests
import logging

# XGBoost import (optional) — if not installed we'll gracefully fallback to rule-based prediction
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-secure-key'

# ---- Configuration ----
IQ_SERVICE_API = "http://localhost:5001/api/iq"
MODEL_PATH = "xgb_model.json"   # expected model path (change as needed)
# Subjects
SUBJECTS = ['kan', 'english', 'maths', 'chem', 'bio_or_cs', 'physics']

# Logging
logging.basicConfig(level=logging.INFO)


# ---- Helpers: rules, capability, suggestions, timetable ----

def compute_activity_score(attendance_pct, extra_activity, time_extra_hrs, attended_academic, courses_count):
    attendance_norm = np.clip(attendance_pct / 100.0, 0.0, 1.0)
    extra_bin = 1 if extra_activity else 0
    time_extra_norm = np.clip(time_extra_hrs / 12.0, 0.0, 1.0)
    attended_bin = 1 if attended_academic else 0
    courses_norm = np.clip(courses_count / 8.0, 0.0, 1.0)

    activity_score = (0.5 * attendance_norm +
                      0.2 * extra_bin +
                      0.15 * time_extra_norm +
                      0.1 * attended_bin +
                      0.05 * courses_norm)
    return float(np.clip(activity_score, 0.0, 1.0))


def capability_score(iq, study_hours_per_day, activity_score):
    iq_norm = np.clip(iq / 100.0, 0.0, 1.0)
    study_norm = np.clip(study_hours_per_day / 4.0, 0.0, 1.0)
    cap = 0.6 * iq_norm + 0.2 * study_norm + 0.2 * activity_score
    return float(cap)


def apply_business_rules(iq, study_hpd, subject_marks, activity_score, cap_score):
    all_subjects_below_35 = all([m < 35 for m in subject_marks.values()])
    avg_marks = sum(subject_marks.values()) / max(1, len(subject_marks))

    if (iq > 40) and (study_hpd >= 1.0):
        return "Pass", "High", "IQ > 40 and study_hours >= 1"
    if (iq < 40) and (study_hpd < 1.0):
        return "Fail", "Low", "IQ < 40 and study_hours < 1"
    if (iq < 40) and (study_hpd > 2.0):
        return "Pass", "Medium", "IQ < 40 but study_hours > 2 (compensated by study)"
    if (iq > 80) and (study_hpd < 1.0) and all_subjects_below_35:
        return "Pass", "Low", "High IQ but low study & low marks (pass with low confidence)"
    if cap_score >= 0.50:
        if cap_score >= 0.80:
            conf = "High"
        elif cap_score >= 0.60:
            conf = "Medium"
        else:
            conf = "Low"
        return "Pass", conf, "Capability score >= 0.50 (fallback)"
    else:
        return "Fail", ("Low" if cap_score < 0.35 else "Medium"), "Capability score < 0.50 (fallback)"


# -------------------------
# XGBoost prediction helper
# -------------------------
_xgb_model = None
def _load_xgb_model(path=MODEL_PATH):
    global _xgb_model
    if not XGB_AVAILABLE:
        return None
    if _xgb_model is not None:
        return _xgb_model
    try:
        # try JSON/Booster load; accept either native Booster JSON or binary
        booster = xgb.Booster()
        booster.load_model(path)
        _xgb_model = booster
        app.logger.info("Loaded XGBoost model from %s", path)
        return _xgb_model
    except Exception as e:
        app.logger.warning("XGBoost model load failed: %s", e)
        _xgb_model = None
        return None


def model_predict(inputs):
    """
    Predict Pass/Fail using XGBoost if available and model file exists.
    Features used (order):
      kan, english, maths, chem, bio_or_cs, physics,
      iq, study_hours_per_day, attendance, extra_activity, time_extra, attended_academic, courses
    Returns (prediction_label, probability)
    Fallback: use apply_business_rules as fallback prediction (probability = capability_score)
    """
    # prepare numeric feature vector
    feat = [
        float(inputs.get('kan', 0.0)),
        float(inputs.get('english', 0.0)),
        float(inputs.get('maths', 0.0)),
        float(inputs.get('chem', 0.0)),
        float(inputs.get('bio_or_cs', 0.0)),
        float(inputs.get('physics', 0.0)),
        float(inputs.get('iq', 100.0)),
        float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
        float(inputs.get('attendance', inputs.get('attendance_percentage', 75.0))),
        1.0 if inputs.get('extra_activity', False) else 0.0,
        float(inputs.get('time_extra', 0.0)),
        1.0 if inputs.get('attended_academic', False) else 0.0,
        float(inputs.get('courses', 0)),
    ]
    booster = _load_xgb_model()
    if booster is None:
        # fallback to rule-based decision using capability_score
        activity = compute_activity_score(
            float(inputs.get('attendance', 75.0)),
            inputs.get('extra_activity', False),
            float(inputs.get('time_extra', 0.0)),
            inputs.get('attended_academic', False),
            int(inputs.get('courses', 0)),
        )
        cap = capability_score(float(inputs.get('iq', 100.0)),
                               float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
                               activity)
        pred, conf, reason = apply_business_rules(float(inputs.get('iq', 100.0)),
                                                  float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
                                                  {s: float(inputs.get(s, 0.0)) for s in SUBJECTS},
                                                  activity, cap)
        # map conf to a rough probability proxy
        prob = float(cap)
        return pred, prob, conf, reason

    try:
        dmat = xgb.DMatrix(np.array([feat], dtype=float))
        proba = float(booster.predict(dmat)[0])
        # For binary classifier many XGBoost setups produce probability of positive class.
        label = "Pass" if proba >= 0.5 else "Fail"
        if proba >= 0.8:
            conf = "High"
        elif proba >= 0.6:
            conf = "Medium"
        else:
            conf = "Low"
        reason = f"Model predicted probability {proba:.3f}"
        return label, proba, conf, reason
    except Exception as e:
        app.logger.exception("XGBoost prediction failed, falling back to rules: %s", e)
        # fallback to rules
        activity = compute_activity_score(
            float(inputs.get('attendance', 75.0)),
            inputs.get('extra_activity', False),
            float(inputs.get('time_extra', 0.0)),
            inputs.get('attended_academic', False),
            int(inputs.get('courses', 0)),
        )
        cap = capability_score(float(inputs.get('iq', 100.0)),
                               float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
                               activity)
        pred, conf, reason = apply_business_rules(float(inputs.get('iq', 100.0)),
                                                  float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
                                                  {s: float(inputs.get(s, 0.0)) for s in SUBJECTS},
                                                  activity, cap)
        prob = float(cap)
        return pred, prob, conf, reason


# -------------------------
# Timetable (IQ-driven) — unchanged logic but ensure we pass IQ where needed
# -------------------------
def generate_timetable(subject_marks, iq):
    """
    Timetable generator driven only by IQ + subject marks:
      - daily hours based on IQ:
          iq < 40      -> 4 hrs/day
          40 <= iq <75 -> 3 hrs/day
          iq >= 75     -> 2 hrs/day
      - weekly_total = daily_hours * 7
      - allocate weekly hours inversely proportional to marks (lower marks get more hours)
      - schedule into 7 days, 3 slots/day prioritizing 2 weakest subjects + 1 other
      - per-slot cap 2.0 hrs, quarter-hour granularity, display to 1 decimal
    """
    iq_val = float(iq or 0.0)

    if iq_val < 40.0:
        daily_hours = 4.0
    elif iq_val < 75.0:
        daily_hours = 3.0
    else:
        daily_hours = 2.0

    total_week_hours = max(1.0, daily_hours * 7.0)
    subj_marks_list = [float(subject_marks.get(s, 0.0)) for s in SUBJECTS]

    marks = np.array(subj_marks_list)
    inv = (100.0 - marks) + 1.0
    weights = inv / inv.sum()
    raw_hours = weights * total_week_hours
    raw_hours = np.round(raw_hours * 4.0) / 4.0

    diff = round(total_week_hours - raw_hours.sum(), 4)
    if abs(diff) >= 0.25:
        order = np.argsort(-inv)
        step = 0.25 if diff > 0 else -0.25
        k = 0
        while abs(round(diff, 4)) >= 0.25 and k < len(order):
            raw_hours[order[k]] += step
            diff -= step
            k += 1
    raw_hours = np.clip(raw_hours, 0.0, None)
    remaining = {SUBJECTS[i]: float(raw_hours[i]) for i in range(len(SUBJECTS))}

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    per_slot_cap = 2.0
    slots_per_day = 3

    if daily_hours >= 4.0:
        slot_targets = [1.5, 1.5, 1.0]
    elif daily_hours >= 3.0:
        slot_targets = [1.0, 1.0, 1.0]
    else:
        slot_targets = [0.75, 0.75, 0.5]

    st_sum = sum(slot_targets)
    if st_sum > daily_hours:
        scale = daily_hours / st_sum if st_sum > 0 else 0.0
        slot_targets = [round(x * scale * 4.0) / 4.0 for x in slot_targets]
        while sum(slot_targets) > daily_hours and any(x >= 0.25 for x in slot_targets):
            for i in range(len(slot_targets)):
                if slot_targets[i] >= 0.25 and sum(slot_targets) > daily_hours:
                    slot_targets[i] = round((slot_targets[i] - 0.25) * 4.0) / 4.0

    timetable = []
    for d in days:
        day_plan = []
        ordered = sorted(remaining.items(), key=lambda kv: -kv[1])
        weak_candidates = [k for k, v in ordered if v >= 0.25][:2]
        other_candidate = None
        for k, v in ordered:
            if k not in weak_candidates and v >= 0.25:
                other_candidate = k
                break
        if len(weak_candidates) < 2:
            for k, v in ordered:
                if k not in weak_candidates:
                    weak_candidates.append(k)
                if len(weak_candidates) >= 2:
                    break
        if other_candidate is None:
            for k, v in ordered:
                if k not in weak_candidates:
                    other_candidate = k
                    break

        chosen = []
        if len(weak_candidates) > 0:
            chosen.append(weak_candidates[0])
        if len(weak_candidates) > 1:
            chosen.append(weak_candidates[1])
        if other_candidate:
            chosen.append(other_candidate)

        pad_list = [s for s in SUBJECTS if s not in chosen]
        idx = 0
        while len(chosen) < slots_per_day:
            if idx < len(pad_list):
                chosen.append(pad_list[idx]); idx += 1
            else:
                chosen.append(None)

        for si in range(slots_per_day):
            subj = chosen[si]
            if subj is None:
                day_plan.append({'subject': None, 'hours': 0.0}); continue
            target = slot_targets[si]
            alloc = min(target, remaining.get(subj, 0.0), per_slot_cap)
            alloc = round(alloc * 4.0) / 4.0
            if alloc < 0.25:
                day_plan.append({'subject': None, 'hours': 0.0})
            else:
                day_plan.append({'subject': subj, 'hours': alloc})
                remaining[subj] = max(0.0, remaining.get(subj, 0.0) - alloc)

        used_today = sum(slot['hours'] for slot in day_plan)
        remaining_to_fill = round(daily_hours - used_today, 4)
        if remaining_to_fill >= 0.25:
            for slot in day_plan:
                if remaining_to_fill < 0.25:
                    break
                if slot['subject'] is None:
                    cand = None
                    for k, v in sorted(remaining.items(), key=lambda kv: -kv[1]):
                        if v >= 0.25:
                            cand = k
                            break
                    if cand:
                        add = min(remaining_to_fill, per_slot_cap, remaining[cand])
                        add = round(add * 4.0) / 4.0
                        if add >= 0.25:
                            slot['subject'] = cand
                            slot['hours'] = add
                            remaining[cand] -= add
                            remaining_to_fill -= add
                else:
                    can_add = min(per_slot_cap - slot['hours'], remaining.get(slot['subject'], 0.0), remaining_to_fill)
                    can_add = round(can_add * 4.0) / 4.0
                    if can_add >= 0.25:
                        slot['hours'] += can_add
                        remaining[slot['subject']] -= can_add
                        remaining_to_fill -= can_add

        while len(day_plan) < slots_per_day:
            day_plan.append({'subject': None, 'hours': 0.0})

        timetable.append({'day': d, 'plan': day_plan})

    final_timetable = []
    for day in timetable:
        plan = day['plan']
        slot_strs = []
        for slot in plan:
            if slot['subject'] is None or slot['hours'] < 0.25:
                slot_strs.append("-")
            else:
                subj_label = slot['subject'].replace('_', '/').title()
                slot_strs.append(f"{subj_label} - {float(round(slot['hours'],1)):.1f} h")
        while len(slot_strs) < 3:
            slot_strs.append("-")
        final_timetable.append({"day": day['day'], "slot1": slot_strs[0], "slot2": slot_strs[1], "slot3": slot_strs[2]})

    return final_timetable


def attractive_suggestions(inputs, prediction, confidence, cap_score):
    sug = []
    subj_marks = {s: float(inputs.get(s, 0)) for s in SUBJECTS}
    avg_marks = sum(subj_marks.values()) / max(1, len(subj_marks))

    if prediction == "Pass":
        if confidence == "High":
            sug.append("✅ Great job — you're on track! Keep doing focused practice and mock tests. 💪📚")
        elif confidence == "Medium":
            sug.append("👍 Good potential — tighten revision on weak topics and do regular timed practice. ⏱️📝")
        else:
            sug.append("🔔 Pass likely but confidence is low — increase focused revision on weak subjects to feel secure. 🔎📘")
    else:
        sug.append("⚠️ Not there yet — you can improve with a targeted plan. Start with the suggestions below. 🔁🔥")

    weakest = sorted(subj_marks.items(), key=lambda x: x[1])[:2]
    for name, score in weakest:
        if score < 35:
            sug.append(f"📉 {name.title()} is low ({score:.0f}). Do concept revision + 5 solved examples daily. 🧠✏️")
        elif score < 60:
            sug.append(f"🔧 {name.title()} ({score:.0f}) — weekly focused practice (3-4 sessions) will help. ✅")
        else:
            sug.append(f"🌟 {name.title()} ({score:.0f}) is good — maintain with brief weekly practice. ✅")

    study = float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0)))
    if study < 1:
        sug.append("⏰ Try to study at least 1–2 hours/day (short focused sessions) — consistency beats marathon sessions. 🕒")
    elif study < 2:
        sug.append("📈 Aim for 2–3 hours/day with active recall & past papers for best gains. 🧪📚")
    else:
        sug.append("🚀 You're putting in good hours — use Pomodoro (25/5) and weekly full-length mocks. 🍅🧾")

    if float(inputs.get('attendance', inputs.get('attendance_percentage', 0))) < 80:
        sug.append("🏫 Improve attendance — in-class doubt clearing is a quick win. 🤝")
    else:
        sug.append("👍 Good attendance — use class time to ask targeted questions. 💬")

    if inputs.get('extra_activity', False):
        te = float(inputs.get('time_extra', inputs.get('time_spent_on_extra_activity_hrs', 0)))
        if te > 8:
            sug.append("⚖️ Balance activities — keep extracurriculars but reduce time near exams to focus on studies. 🧘‍♀️")
        else:
            sug.append("🎯 Extra-curriculars are great — they build skills and relaxation. Maintain a healthy balance. 🌈")

    iq = float(inputs.get('iq', 100))
    if iq < 60:
        sug.append("🧩 Improve reasoning with daily puzzles (15–20 mins) — helps speed and accuracy. 🧠")
    elif iq > 120:
        sug.append("💡 Use higher-order problem sets and time-bound mocks to leverage your strong reasoning. ⚡")

    sug.append(f"🔍 Model confidence: {confidence} • Capability score: {cap_score:.2f}")
    return sug


# ---- Routes ----

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', subjects=SUBJECTS)


@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        form = request.form
        inputs = {}
        for s in SUBJECTS:
            inputs[s] = float(form.get(s, 0))
        inputs['iq'] = float(form.get('iq', 100))
        inputs['study_hours_per_day'] = float(form.get('study_hours_per_day', form.get('study_hours', 0.0)))
        extra_raw = form.get('extra_activity', 'no').lower()
        inputs['extra_activity'] = extra_raw in ('yes', 'true', '1', 'on')
        inputs['time_extra'] = float(form.get('time_extra', 0.0))
        att_raw = form.get('attended_academic', form.get('attended_contest', 'no')).lower()
        inputs['attended_academic'] = att_raw in ('yes', 'true', '1', 'on')
        inputs['courses'] = int(form.get('courses', 0))
        inputs['attendance'] = float(form.get('attendance', 75.0))

        # compute activity & capability
        activity = compute_activity_score(inputs['attendance'], inputs['extra_activity'],
                                          inputs['time_extra'], inputs['attended_academic'], inputs['courses'])
        cap = capability_score(inputs['iq'], inputs['study_hours_per_day'], activity)

        # predict using XGBoost (or fallback)
        pred_label, prob, conf_from_model, reason = model_predict(inputs)

        # if we used model_predict fallback (rules), model_predict returns confidence/reason already
        # use returned pred_label/prob/conf_from_model directly
        tt = generate_timetable({s: inputs[s] for s in SUBJECTS}, inputs['iq'])
        suggestions = attractive_suggestions(inputs, pred_label, conf_from_model, cap)

        result = {
            'prediction': pred_label,
            'confidence': conf_from_model,
            'reason': reason,
            'capability_score': round(cap, 4),
            'probability_estimate': float(prob),
            'timetable': tt,
            'suggestions': suggestions,
            'inputs': inputs
        }

        return render_template('result.html', result=result)
    except Exception as e:
        app.logger.exception("predict_form failed")
        return f"Error: {e}", 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        data = payload.get('inputs', payload)

        inputs = {}
        for s in SUBJECTS:
            inputs[s] = float(data.get(s, 0.0))
        inputs['iq'] = float(data.get('iq', 100.0))
        inputs['study_hours_per_day'] = float(data.get('study_hours_per_day', data.get('study_hours', 0.0)))
        extra = data.get('extra_activity', data.get('extra', False))
        if isinstance(extra, str):
            extra = extra.lower() in ('yes', 'true', '1', 'on')
        inputs['extra_activity'] = bool(extra)
        inputs['time_extra'] = float(data.get('time_extra', 0.0))
        att = data.get('attended_academic', data.get('attended_contest', False))
        if isinstance(att, str):
            att = att.lower() in ('yes', 'true', '1', 'on')
        inputs['attended_academic'] = bool(att)
        inputs['courses'] = int(data.get('courses', 0))
        inputs['attendance'] = float(data.get('attendance', 75.0))

        activity = compute_activity_score(inputs['attendance'], inputs['extra_activity'],
                                          inputs['time_extra'], inputs['attended_academic'], inputs['courses'])
        cap = capability_score(inputs['iq'], inputs['study_hours_per_day'], activity)

        pred_label, prob, conf_from_model, reason = model_predict(inputs)

        tt = generate_timetable({s: inputs[s] for s in SUBJECTS}, inputs['iq'])
        suggestions = attractive_suggestions(inputs, pred_label, conf_from_model, cap)

        result = {
            'prediction': pred_label,
            'confidence': conf_from_model,
            'reason': reason,
            'capability_score': round(cap, 4),
            'probability_estimate': float(prob),
            'timetable': tt,
            'suggestions': suggestions,
            'inputs': inputs
        }

        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        app.logger.exception("api_predict failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---- Optional: IQ callback integration ----
def fetch_iq_from_service(token):
    try:
        r = requests.get(IQ_SERVICE_API, params={'token': token}, timeout=5)
        r.raise_for_status()
        j = r.json()
        return j.get('iq')
    except Exception as e:
        app.logger.error("Failed to fetch IQ from service: %s", e)
        return None


@app.route('/iq-callback')
def iq_callback():
    token = request.args.get('token')
    if not token:
        return "Missing token", 400
    iq = fetch_iq_from_service(token)
    if iq is None:
        return "Unable to fetch IQ from provider", 502
    return redirect(url_for('index', imported_iq=iq))


# ---- Run ----
if __name__ == '__main__':
    if XGB_AVAILABLE:
        _load_xgb_model()   # attempt load at startup (no-op if missing)
    app.run(host='0.0.0.0', port=8080, debug=True)