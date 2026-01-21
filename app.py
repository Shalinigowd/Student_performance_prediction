# app.py
"""
Flask app for Student Performance Predictor.
Safe, robust, and ensures model_predict ALWAYS returns 4 values:
(label, probability, confidence, reason)
Manual IQ entry and float study hours supported.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import logging
import os
import joblib
from jinja2 import Undefined

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace-with-secure-key'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load trained model and scaler (optional)
MODEL_PATH = os.path.join('model', 'xgb_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
MODEL = None
SCALER = None
try:
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        SCALER = joblib.load(SCALER_PATH)
        logger.info("Loaded scaler from %s", SCALER_PATH)
except Exception as e:
    logger.warning("Could not load model or scaler: %s", e)

# Subjects (no Hindi)
SUBJECTS = ['kan', 'english', 'maths', 'chem', 'bio_or_cs', 'physics']

# ---------------------------
# Serialization helper
# ---------------------------
def to_serializable(obj):
    """Convert numpy / jinja / bytes into JSON-serializable python objects."""
    if isinstance(obj, Undefined):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def safe_jsonify(payload):
    return jsonify(to_serializable(payload))

# ---------------------------
# Core helper functions
# ---------------------------
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
    """Return (label, confidence, reason) — used by fallback logic."""
    
    # Immediate fail: no study hours
    if study_hpd == 0:
        return "Fail", "High", "No study hours - Student has not studied."
    
    all_subjects_below_35 = all([m < 35 for m in subject_marks.values()])
    avg_marks = sum(subject_marks.values()) / max(1, len(subject_marks))
    any_subject_below_25 = any([m < 25 for m in subject_marks.values()])

    # Hard fail
    if all_subjects_below_35:
        return "Fail", "High", "All subjects below passing marks (35)."
    if avg_marks < 30:
        return "Fail", "High", "Average marks critically low (<30)."
    if any_subject_below_25:
        return "Fail", "Medium", "At least one subject very low (<25)."

    # Strong pass
    if avg_marks > 60:
        return "Pass", "High", "Average marks good (>60)."

    # Specified rules
    if (iq > 40) and (study_hpd >= 1.0) and (avg_marks >= 35):
        return "Pass", "High", "IQ > 40 and study >=1h/day with passing average."
    if (iq < 40) and (study_hpd < 1.0):
        return "Fail", "Low", "Low IQ and insufficient study (<1h/day)."
    if (iq < 40) and (study_hpd > 2.0):
        return "Pass", "Medium", "Low IQ but compensated by >2h/day study."
    if (iq > 80) and (study_hpd < 1.0) and all_subjects_below_35:
        return "Pass", "Low", "High IQ but low study & marks (pass with low confidence)."

    # fallback capability
    if cap_score >= 0.50:
        conf = "High" if cap_score >= 0.8 else ("Medium" if cap_score >= 0.6 else "Low")
        return "Pass", conf, "Capability score >= 0.50 (fallback)."
    else:
        return "Fail", ("Low" if cap_score < 0.35 else "Medium"), "Capability score < 0.50 (fallback)."

# ---------------------------
# MODEL PREDICT (always returns 4 values)
# ---------------------------
def model_predict(inputs):
    """
    Inputs: dict with subjects, iq, study_hours_per_day, attendance, extra_activity, time_extra, attended_academic, courses
    Returns: (label:str, probability:float, confidence:str, reason:str)
    Guaranteed to ALWAYS return 4 values.
    """
    try:
        feat = [
            float(inputs.get('kan', 0.0)),
            float(inputs.get('english', 0.0)),
            float(inputs.get('maths', 0.0)),
            float(inputs.get('chem', 0.0)),
            float(inputs.get('bio_or_cs', 0.0)),
            float(inputs.get('physics', 0.0)),
            float(inputs.get('iq', 100.0)),
            float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))),
            float(inputs.get('attendance', 75.0)),
            1.0 if inputs.get('extra_activity', False) else 0.0,
            float(inputs.get('time_extra', 0.0)),
            1.0 if inputs.get('attended_academic', False) else 0.0,
            float(inputs.get('courses', 0))
        ]
    except Exception as e:
        logger.exception("Invalid numeric conversion in inputs: %s", e)
        # safe fallback values
        feat = [0.0] * 13

    # If model+scaler available use them
    if MODEL is not None and SCALER is not None:
        try:
            X = np.array([feat], dtype=float)
            Xs = SCALER.transform(X)
            proba = float(MODEL.predict_proba(Xs)[0, 1])
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
            logger.exception("Model prediction failed: %s", e)
            # continue to fallback

    # Fallback: rule-based + capability score
    activity = compute_activity_score(
        float(inputs.get('attendance', 75.0)),
        inputs.get('extra_activity', False),
        float(inputs.get('time_extra', 0.0)),
        inputs.get('attended_academic', False),
        int(inputs.get('courses', 0))
    )
    cap = capability_score(float(inputs.get('iq', 100.0)), float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))), activity)
    pred_label, pred_conf, pred_reason = apply_business_rules(float(inputs.get('iq', 100.0)), float(inputs.get('study_hours_per_day', inputs.get('study_hours', 0.0))), {s: float(inputs.get(s, 0.0)) for s in SUBJECTS}, activity, cap)
    # set probability proxy from capability score
    prob_proxy = float(cap)
    return pred_label, prob_proxy, pred_conf, pred_reason

# ---------------------------
# Timetable generator (3 subjects/day, formatted)
# ---------------------------
def generate_timetable(subject_marks, iq, study_hours_per_day):
    """
    Generate both formatted and raw timetable structures.
    Returns: { 'formatted': [ {day, slot1, slot2, slot3}, ... ],
               'raw': [ {day, plan: [ {subject, hours}, ... ] }, ... ] }
    """
    iq_val = float(iq or 0.0)
    if iq_val < 40:
        daily_hours = 4.0
    elif iq_val < 75:
        daily_hours = 3.0
    else:
        daily_hours = 2.0

    total_week_hours = max(1.0, daily_hours * 7.0)
    marks = np.array([float(subject_marks.get(s, 0.0)) for s in SUBJECTS])
    inv = (100.0 - marks) + 1.0
    weights = inv / inv.sum()
    raw_hours = weights * total_week_hours
    raw_hours = np.round(raw_hours * 4.0) / 4.0
    remaining = {SUBJECTS[i]: float(raw_hours[i]) for i in range(len(SUBJECTS))}

    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    slots = 3
    if daily_hours >= 4.0:
        slot_targets = [1.5, 1.5, 1.0]
    elif daily_hours >= 3.0:
        slot_targets = [1.0, 1.0, 1.0]
    else:
        slot_targets = [0.75, 0.75, 0.5]

    raw_timetable = []
    formatted = []
    for d in days:
        ordered = sorted(remaining.items(), key=lambda kv: -kv[1])
        chosen = [k for k, v in ordered if v >= 0.25][:slots]
        while len(chosen) < slots:
            chosen.append(None)

        day_plan = []
        slot_texts = []
        for si, subj in enumerate(chosen):
            if subj is None:
                day_plan.append({'subject': None, 'hours': 0.0})
                slot_texts.append('-')
                continue
            alloc = min(slot_targets[si], remaining.get(subj, 0.0))
            alloc = round(alloc * 4.0) / 4.0
            if alloc < 0.25:
                day_plan.append({'subject': None, 'hours': 0.0})
                slot_texts.append('-')
            else:
                day_plan.append({'subject': subj, 'hours': float(alloc)})
                slot_texts.append(f"{subj.replace('_',' ').title()} - {alloc:.2f} h")
                remaining[subj] = max(0.0, remaining.get(subj, 0.0) - alloc)

        raw_timetable.append({'day': d, 'plan': day_plan})
        formatted.append({'day': d, 'slot1': slot_texts[0], 'slot2': slot_texts[1], 'slot3': slot_texts[2]})

    return {'formatted': formatted, 'raw': raw_timetable}

# ---------------------------
# Routes
# ---------------------------
@app.route('/', methods=['GET'])
def index():
    # Index form expects manual IQ input and float study hours
    return render_template('index.html', subjects=SUBJECTS)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        form = request.form
        # Server-side validation: IQ score and IQ certificate are required
        iq_raw = (form.get('iq') or '').strip()
        iq_file = request.files.get('iq_certificate') if hasattr(request, 'files') else None
        if not iq_raw:
            return render_template('index.html', subjects=SUBJECTS, error='IQ score is required. Please enter your IQ test marks.')
        if iq_file is None or getattr(iq_file, 'filename', '') == '':
            return render_template('index.html', subjects=SUBJECTS, error='IQ certificate is required. Please upload your certificate file.')
        inputs = {}
        for s in SUBJECTS:
            try:
                inputs[s] = float(form.get(s, 0) or 0)
            except:
                inputs[s] = 0.0
        # manual IQ (required to be numeric)
        try:
            inputs['iq'] = float(iq_raw)
        except:
            return render_template('index.html', subjects=SUBJECTS, error='IQ score must be a valid number between 0 and 100.')
        # study hours per day (float input)
        try:
            inputs['study_hours_per_day'] = float(form.get('study_hours_per_day', form.get('study_hours', 3.5)) or 3.5)
        except:
            inputs['study_hours_per_day'] = 3.5
        extra_raw = (form.get('extra_activity') or 'no').lower()
        inputs['extra_activity'] = extra_raw in ('yes','true','1','on')
        try:
            inputs['time_extra'] = float(form.get('time_extra', 0) or 0)
        except:
            inputs['time_extra'] = 0.0
        att_raw = (form.get('attended_academic') or form.get('attended_contest') or 'no').lower()
        inputs['attended_academic'] = att_raw in ('yes','true','1','on')
        try:
            inputs['courses'] = int(form.get('courses', 0) or 0)
        except:
            inputs['courses'] = 0
        try:
            inputs['attendance'] = float(form.get('attendance', 75) or 75)
        except:
            inputs['attendance'] = 75.0

        # Ensure no NaN or undefined
        inputs = {k: (v if v is not None else 0) for k, v in inputs.items()}

        # Compute
        activity = compute_activity_score(inputs['attendance'], inputs['extra_activity'], inputs['time_extra'], inputs['attended_academic'], inputs['courses'])
        cap = capability_score(inputs['iq'], inputs['study_hours_per_day'], activity)

        # Model prediction (always 4 values)
        label, prob, conf, reason = model_predict(inputs)

        # Also compute rule result
        rule_label, rule_conf, rule_reason = apply_business_rules(inputs['iq'], inputs['study_hours_per_day'], {s: inputs[s] for s in SUBJECTS}, activity, cap)

        # Hybrid decision: prefer rule when it's high-confidence
        if rule_conf == 'High':
            final_label = rule_label
            final_prob = 0.98 if rule_label == "Pass" else 0.02
            final_conf = rule_conf
            final_reason = "Rule: " + rule_reason
        else:
            final_label = label
            final_prob = prob
            final_conf = conf
            final_reason = reason

        tt = generate_timetable({s: inputs[s] for s in SUBJECTS}, inputs['iq'], inputs['study_hours_per_day'])
        timetable = tt.get('formatted') if isinstance(tt, dict) else tt
        timetable_raw = tt.get('raw') if isinstance(tt, dict) else None

        # Suggestions (emoji friendly)
        weakest = sorted([(s, inputs[s]) for s in SUBJECTS], key=lambda x: x[1])[:2]
        strongest = sorted([(s, inputs[s]) for s in SUBJECTS], key=lambda x: -x[1])[:1]
        avg_marks = sum([inputs[s] for s in SUBJECTS]) / len(SUBJECTS)
        
        suggestions = []
        
        # Main prediction feedback
        if final_label == "Pass":
            suggestions.append("✅ Excellent! You're likely to pass — maintain consistency and focus! 🎯")
        else:
            suggestions.append("⚠️  Warning: Risk of failing — implement the action plan below immediately. 🔥")
        
        # Study hours feedback
        study_hours = inputs.get('study_hours_per_day', 0)
        if study_hours == 0:
            suggestions.append("🚨 CRITICAL: You haven't logged any study hours. Start studying TODAY! 📚")
        elif study_hours < 2:
            suggestions.append(f"⏰ Study Time Alert: {study_hours:.1f}h/day is below recommended. Aim for 3-4 hours daily. 📖")
        elif study_hours >= 4:
            suggestions.append(f"🚀 Great! {study_hours:.1f}h/day study habit is excellent. Stay consistent! ⭐")
        
        # Subject-specific recommendations
        for name, score in weakest:
            readable_name = name.replace('_', ' ').title()
            if score < 25:
                suggestions.append(f"🔴 URGENT: {readable_name} ({score:.0f}/100) — Start with NCERT basics, solve 10 simple problems daily, get peer/tutor help. 👥")
            elif score < 35:
                suggestions.append(f"📉 Priority: {readable_name} ({score:.0f}/100) — Review fundamentals, practice 5 solved examples daily, focus on weak topics. 🎓")
            elif score < 50:
                suggestions.append(f"⚙️ Moderate: {readable_name} ({score:.0f}/100) — Practice 3-4 problems daily, review last 5 chapters, strengthen formulas. 📝")
            elif score < 60:
                suggestions.append(f"🔧 Improvement: {readable_name} ({score:.0f}/100) — 3 focused practice sessions/week, attempt previous year papers. 📄")
        
        # Strength acknowledgment
        if strongest:
            name, score = strongest[0]
            readable_name = name.replace('_', ' ').title()
            suggestions.append(f"⭐ Strength: {readable_name} ({score:.0f}/100) is strong — help peers, teach others to reinforce! 🏆")
        
        # Overall performance feedback
        if avg_marks < 30:
            suggestions.append(f"📊 Average: {avg_marks:.0f}/100 (Critical) — Join a study group, get tutoring support, revise syllabus systematically. 💡")
        elif avg_marks < 45:
            suggestions.append(f"📊 Average: {avg_marks:.0f}/100 (Below Expected) — Increase focus time, practice consistently, review mistake patterns. 📋")
        elif avg_marks < 60:
            suggestions.append(f"📊 Average: {avg_marks:.0f}/100 (Moderate) — Solid foundation — refine problem-solving skills, practice timed tests. ⏱️")
        else:
            suggestions.append(f"📊 Average: {avg_marks:.0f}/100 (Excellent) — You're on track! Focus on retention & accuracy in exams. 🎖️")
        
        # IQ and capability feedback
        iq = inputs.get('iq', 100)
        if iq < 40:
            suggestions.append(f"🧠 IQ Score: {iq} (Consider) — Leverage extended study time and structured learning methods. 🎯")
        elif iq >= 80:
            suggestions.append(f"🧠 IQ Score: {iq} (Excellent) — You have strong analytical ability — use it wisely for exam strategies! 🔬")
        
        # Activity & attendance
        if inputs.get('attendance', 75) < 75:
            suggestions.append(f"🎓 Attendance: {inputs.get('attendance', 75):.0f}% — Improve class presence, don't miss important lectures! 📍")
        
        # Extra-curricular balance
        if inputs.get('extra_activity'):
            suggestions.append(f"⚖️ Balance: Extra activities can help — manage time wisely between studies & activities! ⏳")
        
        # Final motivational & action-oriented message
        suggestions.append(f"🎯 Action Plan: Follow the timetable below, set weekly milestones, review progress every Sunday. 📅")
        suggestions.append(f"💪 You've got this! Small daily efforts lead to big results. Stay focused & motivated! 🌟")

        result = {
            'prediction': final_label,
            'probability': float(round(final_prob, 3)),
            'probability_estimate': float(round(final_prob, 3)),
            'confidence': final_conf,
            'reason': final_reason,
            'capability_score': float(round(cap, 4)),
            'timetable': timetable,
            'timetable_raw': timetable_raw,
            'suggestions': suggestions,
            'inputs': inputs
        }

        return render_template('result.html', result=to_serializable(result))
    except Exception as e:
        logger.exception("predict_form error: %s", e)
        return f"Error: {e}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        data = payload.get('inputs', payload)

        inputs = {}
        for s in SUBJECTS:
            inputs[s] = float(data.get(s, 0.0) or 0.0)
        inputs['iq'] = float(data.get('iq', 100.0) or 100.0)
        # study hours per day (float input)
        try:
            inputs['study_hours_per_day'] = float(data.get('study_hours_per_day', data.get('study_hours', 3.5)) or 3.5)
        except:
            inputs['study_hours_per_day'] = 3.5
        extra = data.get('extra_activity', False)
        if isinstance(extra, str):
            extra = extra.lower() in ('yes','true','1','on')
        inputs['extra_activity'] = bool(extra)
        try:
            inputs['time_extra'] = float(data.get('time_extra', 0.0) or 0.0)
        except:
            inputs['time_extra'] = 0.0
        att = data.get('attended_academic', data.get('attended_contest', False))
        if isinstance(att, str):
            att = att.lower() in ('yes','true','1','on')
        inputs['attended_academic'] = bool(att)
        inputs['courses'] = int(data.get('courses', 0) or 0)
        try:
            inputs['attendance'] = float(data.get('attendance', 75.0) or 75.0)
        except:
            inputs['attendance'] = 75.0

        label, prob, conf, reason = model_predict(inputs)

        activity = compute_activity_score(inputs['attendance'], inputs['extra_activity'], inputs['time_extra'], inputs['attended_academic'], inputs['courses'])
        cap = capability_score(inputs['iq'], inputs['study_hours_per_day'], activity)
        rule_label, rule_conf, rule_reason = apply_business_rules(inputs['iq'], inputs['study_hours_per_day'], {s: inputs[s] for s in SUBJECTS}, activity, cap)

        if rule_conf == 'High':
            final_label = rule_label
            final_prob = 0.98 if rule_label == "Pass" else 0.02
            final_conf = rule_conf
            final_reason = "Rule: " + rule_reason
        else:
            final_label = label
            final_prob = prob
            final_conf = conf
            final_reason = reason

        tt = generate_timetable({s: inputs[s] for s in SUBJECTS}, inputs['iq'], inputs['study_hours_per_day'])
        timetable = tt.get('formatted') if isinstance(tt, dict) else tt
        timetable_raw = tt.get('raw') if isinstance(tt, dict) else None
        suggestions = ["(See UI for personalized suggestions)"]

        result = {
            'prediction': final_label,
            'probability': float(round(final_prob, 3)),
            'probability_estimate': float(round(final_prob, 3)),
            'confidence': final_conf,
            'reason': final_reason,
            'capability_score': float(round(cap, 4)),
            'timetable': timetable,
            'timetable_raw': timetable_raw,
            'suggestions': suggestions,
            'inputs': inputs
        }

        return safe_jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        logger.exception("api_predict failed: %s", e)
        return safe_jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
