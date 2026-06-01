"""Enhanced rule-based form assessment with scoring system.

Each exercise has:
- Fault checks with specific angle thresholds (researched biomechanics)
- Deduction-based scoring (base 100, subtract per fault)
- Pass/fail threshold per exercise
- Detailed feedback strings

Angle indices (from normalize.py ANGLE_DEFINITIONS):
  0: L-Shoulder  (Hip-Shoulder-Elbow)    = arm-to-torso angle
  1: R-Shoulder  (Hip-Shoulder-Elbow)
  2: L-Elbow     (Shoulder-Elbow-Wrist)  = elbow flexion
  3: R-Elbow     (Shoulder-Elbow-Wrist)
  4: L-Hip       (Shoulder-Hip-Knee)     = hip flexion
  5: R-Hip       (Shoulder-Hip-Knee)
  6: L-Knee      (Hip-Knee-Ankle)        = knee flexion
  7: R-Knee      (Hip-Knee-Ankle)
  8: L-Trunk     (Knee-Hip-Shoulder)     = trunk angle (≡ idx 4 geometrically)
  9: R-Trunk     (Knee-Hip-Shoulder)     (≡ idx 5 geometrically)
 10: L-Ankle     (Hip-Ankle-Knee)
 11: R-Ankle     (Hip-Ankle-Knee)

Convention: 180° = fully straight/extended, 90° = right angle
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise name list and mappings
# ---------------------------------------------------------------------------

EXERCISE_NAMES = [
    'bench_press', 'biceps', 'shoulder_press', 'triceps',
]

DIR_TO_EXERCISE = {
    'bench press':    0,
    'bench_press':    0,
    'biceps':         1,
    'shoulder press': 2,
    'shoulder_press': 2,
    'triceps':        3,
}


def get_exercise_names(config=None):
    """Return exercise name list: from config if provided, else built-in EXERCISE_NAMES.

    Use this so the pipeline works with any data: set config['data']['exercises']
    to your class list and num_exercises is derived from its length.
    """
    if config and isinstance(config.get('data'), dict):
        exercises = config['data'].get('exercises')
        if exercises and len(exercises) > 0:
            return list(exercises)
    return EXERCISE_NAMES


# ---------------------------------------------------------------------------
# Helper: compute stats from angle windows
# ---------------------------------------------------------------------------

def _angle_stats(angles):
    """Compute per-angle min, max, mean, range from a (T, 12) window.

    Also computes averaged L/R versions for bilateral checks.
    Returns a dict with all stats for easy access.
    """
    if angles.ndim == 1:
        angles = angles[np.newaxis, :]

    stats = {
        'min': angles.min(axis=0),       # (12,)
        'max': angles.max(axis=0),       # (12,)
        'mean': angles.mean(axis=0),     # (12,)
        'range': angles.max(axis=0) - angles.min(axis=0),  # (12,)
    }

    # Averaged bilateral angles (mean of L and R per frame, then stats)
    avg_knee = (angles[:, 6] + angles[:, 7]) / 2.0
    avg_elbow = (angles[:, 2] + angles[:, 3]) / 2.0
    avg_shoulder = (angles[:, 0] + angles[:, 1]) / 2.0
    avg_hip = (angles[:, 4] + angles[:, 5]) / 2.0
    avg_trunk = (angles[:, 8] + angles[:, 9]) / 2.0

    stats['avg_knee_min'] = avg_knee.min()
    stats['avg_knee_max'] = avg_knee.max()
    stats['avg_knee_range'] = avg_knee.max() - avg_knee.min()

    stats['avg_elbow_min'] = avg_elbow.min()
    stats['avg_elbow_max'] = avg_elbow.max()
    stats['avg_elbow_range'] = avg_elbow.max() - avg_elbow.min()

    stats['avg_shoulder_min'] = avg_shoulder.min()
    stats['avg_shoulder_max'] = avg_shoulder.max()
    stats['avg_shoulder_range'] = avg_shoulder.max() - avg_shoulder.min()

    stats['avg_hip_min'] = avg_hip.min()
    stats['avg_hip_max'] = avg_hip.max()
    stats['avg_hip_range'] = avg_hip.max() - avg_hip.min()

    stats['avg_trunk_min'] = avg_trunk.min()
    stats['avg_trunk_max'] = avg_trunk.max()
    stats['avg_trunk_range'] = avg_trunk.max() - avg_trunk.min()

    # L/R asymmetry (max difference over the window)
    stats['elbow_asymmetry'] = np.abs(angles[:, 2] - angles[:, 3]).max()
    stats['knee_asymmetry'] = np.abs(angles[:, 6] - angles[:, 7]).max()
    stats['shoulder_asymmetry'] = np.abs(angles[:, 0] - angles[:, 1]).max()

    return stats


# ---------------------------------------------------------------------------
# Per-exercise scoring functions
# ---------------------------------------------------------------------------
# Each returns (score, feedback_list)
# score: 0-100 (100 = perfect)
# feedback_list: list of feedback strings for detected faults


def _score_squat(s):
    """Squat form assessment.

    Verified against: gym_analyzer SquatAnalyzer + biomechanics research.
    Key angles: knee (depth), trunk (forward lean).
    Fixed from gym_analyzer: added trunk check (was missing), adjusted depth thresholds.
    """
    score = 100
    feedback = []

    # Depth check — most important for squat
    if s['avg_knee_min'] > 135:
        score -= 40
        feedback.append('Very shallow squat - aim for thighs parallel to ground')
    elif s['avg_knee_min'] > 120:
        score -= 25
        feedback.append('Incomplete squat depth - go deeper')

    # Forward lean — trunk angle at the deepest point
    if s['avg_trunk_min'] < 100:
        score -= 20
        feedback.append('Excessive forward lean - keep chest up')

    # Leaning back
    if s['avg_trunk_max'] > 190:
        score -= 15
        feedback.append('Leaning too far back')

    return score, feedback


def _score_push_up(s):
    """Push-up form assessment.

    Verified against: gym_analyzer PushUpAnalyzer.
    Key angles: elbow (depth), hip (body alignment).
    Adapted: uses hip angle directly instead of perpendicular distance.
    """
    score = 100
    feedback = []

    # Depth
    if s['avg_elbow_min'] > 120:
        score -= 30
        feedback.append('Not going low enough - elbows should reach ~90 degrees')
    elif s['avg_elbow_min'] > 110:
        score -= 15
        feedback.append('Slightly shallow - try to go a bit deeper')

    # Lockout at top
    if s['avg_elbow_max'] < 150:
        score -= 15
        feedback.append('Fully extend arms at the top')

    # Hip sag (core not engaged)
    if s['avg_hip_min'] < 155:
        score -= 25
        feedback.append('Hips sagging - engage your core')

    # Hip pike
    if s['avg_hip_max'] > 195:
        score -= 20
        feedback.append('Hips too high - straighten your body')

    return score, feedback


def _score_barbell_biceps_curl(s):
    """Barbell biceps curl form assessment.

    Based on: biomechanics research.
    Key angles: elbow (curl range), trunk (stability), shoulder (elbow drift).
    """
    score = 100
    feedback = []

    # Incomplete curl at top
    if s['avg_elbow_min'] > 60:
        score -= 20
        feedback.append('Curl higher for full contraction')

    # Not extending enough at bottom
    if s['avg_elbow_max'] < 140:
        score -= 15
        feedback.append('Extend arms more at the bottom')

    # Swinging / leaning back (using body momentum)
    if s['avg_trunk_min'] < 160:
        score -= 25
        feedback.append('Swinging body - keep torso upright, reduce weight')

    # Elbows drifting forward
    if s['avg_shoulder_max'] > 40:
        score -= 15
        feedback.append('Elbows drifting forward - pin upper arms to sides')

    return score, feedback


def _score_hammer_curl(s):
    """Hammer curl — biomechanically identical to biceps curl from 2D."""
    return _score_barbell_biceps_curl(s)


def _score_bench_press(s):
    """Bench press form assessment.

    Verified against: gym_analyzer BenchPressAnalyzer.
    Key angles: elbow only (supine position limits other angles).
    Adapted: removed pixel-based hip lift check, kept asymmetry check.
    """
    score = 100
    feedback = []

    # Depth — not touching chest
    if s['avg_elbow_min'] > 120:
        score -= 30
        feedback.append('Not going deep enough - lower bar to chest')
    elif s['avg_elbow_min'] > 110:
        score -= 15
        feedback.append('Slightly shallow - try to touch chest')

    # Lockout
    if s['avg_elbow_max'] < 150:
        score -= 20
        feedback.append('Fully lock out arms at the top')

    # Asymmetry (one arm lagging)
    if s['elbow_asymmetry'] > 25:
        score -= 20
        feedback.append('Arms uneven - press symmetrically')

    return score, feedback


def _score_incline_bench_press(s):
    """Incline bench press — same detectable metrics as bench press from 2D."""
    return _score_bench_press(s)


def _score_decline_bench_press(s):
    """Decline bench press — same detectable metrics as bench press from 2D."""
    return _score_bench_press(s)


def _score_deadlift(s):
    """Deadlift form assessment.

    Verified against: gym_analyzer DeadliftAnalyzer (BUGS FIXED).
    Key angles: trunk (back position), knee (leg drive).
    Fixed: gym_analyzer's rounding check was inverted, pixel thresholds replaced.
    """
    score = 100
    feedback = []

    # Back rounding — trunk too flexed
    if s['avg_trunk_min'] < 80:
        score -= 30
        feedback.append('Back rounding - maintain neutral spine')

    # Squatting the deadlift (excessive knee bend)
    if s['avg_knee_min'] < 90:
        score -= 20
        feedback.append('Too much knee bend - this is a deadlift, not a squat')

    # Incomplete lockout
    if s['avg_trunk_max'] < 165:
        score -= 20
        feedback.append('Stand fully upright at the top')

    # Stiff legs at bottom (not using legs)
    # Only flag if trunk is hinged but knees are barely bent
    if s['avg_knee_min'] > 155 and s['avg_trunk_min'] < 120:
        score -= 15
        feedback.append('Bend knees more at the bottom')

    return score, feedback


def _score_romanian_deadlift(s):
    """Romanian deadlift form assessment.

    Based on: biomechanics research.
    Key difference from conventional DL: knees stay straighter, more hip hinge.
    """
    score = 100
    feedback = []

    # Back rounding
    if s['avg_trunk_min'] < 85:
        score -= 30
        feedback.append('Back rounding - hinge at hips with neutral spine')

    # Too much knee bend (should be straighter than conventional DL)
    if s['avg_knee_min'] < 140:
        score -= 25
        feedback.append('Bending knees too much - keep legs straighter')

    # Locked knees (need soft bend)
    if s['avg_knee_min'] > 175:
        score -= 15
        feedback.append('Knees locked - maintain a soft bend')

    # Not hinging deep enough
    if s['avg_trunk_min'] > 130:
        score -= 20
        feedback.append('Not hinging deep enough - push hips back more')

    return score, feedback


def _score_hip_thrust(s):
    """Hip thrust form assessment.

    Based on: biomechanics research.
    Key angles: hip (extension), knee (position).
    """
    score = 100
    feedback = []

    # Incomplete hip extension at top
    if s['avg_hip_max'] < 155:
        score -= 25
        feedback.append('Push hips higher - full extension at the top')

    # Hyperextension
    if s['avg_hip_max'] > 190:
        score -= 20
        feedback.append('Hyperextending - stop at full hip extension')

    # Knees too bent
    if s['avg_knee_min'] < 70:
        score -= 15
        feedback.append('Knees too bent - aim for ~90 degree knee angle')

    # Knees too straight
    if s['avg_knee_max'] > 115:
        score -= 15
        feedback.append('Feet too far out - bring closer for 90 degree knees')

    return score, feedback


def _score_shoulder_press(s):
    """Shoulder press form assessment.

    Based on: biomechanics research.
    Key angles: shoulder (press height), elbow (lockout), trunk (lean).
    Note: Many people do SEATED shoulder press with back support, where the
    trunk angle (Knee-Hip-Shoulder) can be very low due to the reclined seat.
    Trunk check is therefore lenient. Focus is on shoulder ROM and elbow lockout.
    """
    score = 100
    feedback = []

    # Incomplete press — primary quality indicator
    if s['avg_shoulder_max'] < 145:
        score -= 25
        feedback.append('Press higher - reach full overhead extension')

    # Not lowering enough
    if s['avg_shoulder_min'] > 120:
        score -= 15
        feedback.append('Lower the weight more before pressing')

    # Excessive back lean — only flag extreme lean (seated press naturally has lower trunk)
    # Only flag if standing (trunk > 120 at max means they are somewhat upright)
    if s['avg_trunk_max'] > 120 and s['avg_trunk_min'] < 140:
        score -= 20
        feedback.append('Excessive back lean - keep torso more upright')

    # No elbow lockout
    if s['avg_elbow_max'] < 145:
        score -= 15
        feedback.append('Lock out elbows at the top')

    return score, feedback


def _score_lat_pulldown(s):
    """Lat pulldown form assessment.

    Based on: biomechanics research.
    Key angles: elbow (pull range), trunk (lean).
    """
    score = 100
    feedback = []

    # Incomplete pull
    if s['avg_elbow_min'] > 90:
        score -= 25
        feedback.append('Pull the bar lower - to upper chest level')

    # Not extending at top
    if s['avg_elbow_max'] < 145:
        score -= 15
        feedback.append('Fully extend arms at the top')

    # Excessive lean back (using momentum)
    if s['avg_trunk_min'] < 150:
        score -= 25
        feedback.append('Leaning too far back - use less momentum')

    # Trunk instability (swinging)
    if s['avg_trunk_range'] > 25:
        score -= 15
        feedback.append('Torso swinging - keep body stable')

    return score, feedback


def _score_lateral_raise(s):
    """Lateral raise form assessment.

    Based on: biomechanics research.
    Key angles: shoulder (arm height), trunk (stability).
    """
    score = 100
    feedback = []

    # Not raising high enough
    if s['avg_shoulder_max'] < 70:
        score -= 25
        feedback.append('Raise arms higher - to shoulder level')

    # Too high (traps taking over)
    if s['avg_shoulder_max'] > 110:
        score -= 20
        feedback.append('Arms too high - stop at shoulder height')

    # Swinging / momentum
    if s['avg_trunk_min'] < 160 or s['avg_trunk_range'] > 15:
        score -= 20
        feedback.append('Using momentum - keep torso still, reduce weight')

    return score, feedback


def _score_leg_extension(s):
    """Leg extension form assessment.

    Based on: biomechanics research.
    Key angle: knee only (seated isolation exercise).
    """
    score = 100
    feedback = []

    # Incomplete extension at top
    if s['avg_knee_max'] < 150:
        score -= 25
        feedback.append('Extend legs fully at the top')

    # Not bending enough at bottom (partial ROM)
    if s['avg_knee_min'] > 110:
        score -= 20
        feedback.append('Lower the weight more - use full range of motion')

    return score, feedback


def _score_leg_raises(s):
    """Leg raises form assessment.

    Based on: biomechanics research.
    Key angles: hip (leg height), knee (leg straightness).
    """
    score = 100
    feedback = []

    # Legs not raised high enough
    if s['avg_hip_min'] > 120:
        score -= 25
        feedback.append('Raise legs higher')

    # Bent knees (should keep legs straight)
    if s['avg_knee_min'] < 150:
        score -= 20
        feedback.append('Keep legs straighter - avoid bending knees')

    return score, feedback


def _score_lunges(s):
    """Lunge form assessment.

    Verified against: gym_analyzer LungeAnalyzer.
    Key angles: knee (depth), trunk (posture).
    Adapted: simplified from gym_analyzer (no pixel-based checks, no view detection).
    """
    score = 100
    feedback = []

    # Incomplete depth
    if s['avg_knee_min'] > 130:
        score -= 25
        feedback.append('Lunge deeper - front thigh should be near parallel')
    elif s['avg_knee_min'] > 120:
        score -= 10
        feedback.append('Slightly shallow lunge')

    # Excessive forward lean
    if s['avg_trunk_min'] < 155:
        score -= 20
        feedback.append('Leaning forward - keep torso upright')

    # Leaning back
    if s['avg_trunk_max'] > 195:
        score -= 15
        feedback.append('Leaning too far back')

    return score, feedback


def _score_plank(s):
    """Plank form assessment.

    Verified against: gym_analyzer PlankAnalyzer (BUGS FIXED).
    Key angle: hip (body alignment).
    Fixed: gym_analyzer's >200 check can never trigger with clamped angles.
    """
    score = 100
    feedback = []

    # Hip sag (core not engaged)
    if s['avg_hip_min'] < 155:
        score -= 35
        feedback.append('Hips sagging - engage your core')

    # Hip pike
    if s['avg_hip_max'] > 195:
        score -= 25
        feedback.append('Hips too high - lower into a straight line')

    # Instability (bouncing)
    if s['avg_hip_range'] > 20:
        score -= 15
        feedback.append('Hold position steady - reduce movement')

    return score, feedback


def _score_pull_up(s):
    """Pull-up form assessment.

    Based on: biomechanics research.
    Key angle: elbow (pull height).
    """
    score = 100
    feedback = []

    # Partial pull (chin not over bar)
    if s['avg_elbow_min'] > 90:
        score -= 30
        feedback.append('Pull higher - chin should clear the bar')
    elif s['avg_elbow_min'] > 80:
        score -= 15
        feedback.append('Almost there - pull just a bit higher')

    # Not fully extending at bottom
    if s['avg_elbow_max'] < 150:
        score -= 20
        feedback.append('Fully extend arms at the bottom (dead hang)')

    return score, feedback


def _score_chest_fly_machine(s):
    """Chest fly machine form assessment.

    Verified against: gym_analyzer ChestFlyAnalyzer.
    Key angles: shoulder (arm arc), elbow (should stay ~constant).
    Adapted: uses shoulder angle as proxy (no wrist distance available).
    """
    score = 100
    feedback = []

    # Incomplete squeeze (arms not coming together)
    if s['avg_shoulder_min'] > 60:
        score -= 25
        feedback.append('Squeeze arms closer together at the end')

    # Over-stretching (too wide)
    if s['avg_shoulder_max'] > 160:
        score -= 20
        feedback.append('Not going so wide - risk of shoulder strain')

    # Pressing motion instead of fly (elbow angle changing too much)
    if s['avg_elbow_range'] > 40:
        score -= 30
        feedback.append('Too much elbow bending - maintain arm arc for a fly motion')

    return score, feedback


def _score_jumping_jack(s):
    """Jumping jack form assessment.

    Based on: gym_analyzer JumpingJacksAnalyzer (enhanced).
    Key angle: shoulder (arm height).
    Note: 2D reliability is low. Rules are lenient.
    """
    score = 100
    feedback = []

    # Arms not going high enough
    if s['avg_shoulder_max'] < 140:
        score -= 25
        feedback.append('Raise arms higher - fully overhead')

    # Arms not returning to sides
    if s['avg_shoulder_min'] > 40:
        score -= 20
        feedback.append('Bring arms fully down to your sides')

    return score, feedback


def _score_t_bar_row(s):
    """T-bar row form assessment.

    Verified against: gym_analyzer RowAnalyzer/TBarRowAnalyzer.
    Key angles: elbow (pull range), trunk (bent-over position, stability).
    Adapted: replaced pixel-based checks with angle-based.
    Note: T-bar row is typically more upright than bent-over barbell row,
    so trunk thresholds are more lenient than a standard row.
    """
    score = 100
    feedback = []

    # Incomplete pull — relaxed from 90 to 100 since T-bar has shorter ROM
    if s['avg_elbow_min'] > 100:
        score -= 25
        feedback.append('Pull the bar higher - closer to your chest')

    # Not extending fully at bottom
    if s['avg_elbow_max'] < 135:
        score -= 20
        feedback.append('Extend arms more at the bottom')

    # Standing too upright — T-bar allows more upright than bent-over row
    if s['avg_trunk_max'] > 160:
        score -= 20
        feedback.append('Stay bent over - maintain hip hinge position')

    # Trunk swing (using momentum) — threshold relaxed for T-bar machine
    if s['avg_trunk_range'] > 25:
        score -= 20
        feedback.append('Torso swinging - keep back still, reduce weight')

    return score, feedback


def _score_tricep_pushdown(s):
    """Tricep pushdown form assessment.

    Based on: biomechanics research.
    Key angles: elbow (extension), shoulder (elbow position), trunk (lean).
    """
    score = 100
    feedback = []

    # Incomplete extension at bottom
    if s['avg_elbow_min'] > 40:
        score -= 25
        feedback.append('Extend arms fully at the bottom')

    # Not enough flexion at top (partial ROM)
    if s['avg_elbow_max'] < 80:
        score -= 20
        feedback.append('Let the bar come up more - use full range of motion')

    # Elbows drifting forward (not pinned to sides)
    if s['avg_shoulder_max'] > 35:
        score -= 20
        feedback.append('Elbows drifting forward - pin upper arms to your sides')

    # Leaning into the movement (using body weight)
    if s['avg_trunk_min'] < 160:
        score -= 15
        feedback.append('Leaning into the weight - stand upright')

    return score, feedback


def _score_tricep_dips(s):
    """Tricep dips form assessment.

    Verified against: gym_analyzer DipsAnalyzer.
    Key angle: elbow (depth and lockout).
    Adapted: removed pixel-based shrug detection.
    """
    score = 100
    feedback = []

    # Partial ROM (not deep enough)
    if s['avg_elbow_min'] > 120:
        score -= 30
        feedback.append('Go deeper - elbows should reach ~90 degrees')
    elif s['avg_elbow_min'] > 115:
        score -= 15
        feedback.append('Slightly shallow - try to dip a bit deeper')

    # Going too deep (shoulder strain risk)
    if s['avg_elbow_min'] < 70:
        score -= 15
        feedback.append('Going too deep - risk of shoulder strain')

    # No lockout at top
    if s['avg_elbow_max'] < 150:
        score -= 20
        feedback.append('Fully extend arms at the top')

    return score, feedback


def _score_russian_twist(s):
    """Russian twist form assessment.

    Based on: biomechanics research.
    Key angle: trunk (recline angle).
    Note: Primary motion (rotation) is invisible from side view. Rules are minimal.
    """
    score = 100
    feedback = []

    # Too upright (insufficient core engagement)
    if s['avg_trunk_max'] > 155:
        score -= 25
        feedback.append('Lean back more for core engagement')

    # Too far back
    if s['avg_trunk_min'] < 90:
        score -= 20
        feedback.append('Leaning too far back - risk of lower back strain')

    # Position unstable
    if s['avg_trunk_range'] > 30:
        score -= 15
        feedback.append('Hold recline angle steady while rotating')

    return score, feedback


# ---------------------------------------------------------------------------
# Scoring dispatch table
# ---------------------------------------------------------------------------

_SCORE_FUNCTIONS = {
    'squat': _score_squat,
    'push_up': _score_push_up,
    'barbell_biceps_curl': _score_barbell_biceps_curl,
    'hammer_curl': _score_hammer_curl,
    'bench_press': _score_bench_press,
    'incline_bench_press': _score_incline_bench_press,
    'decline_bench_press': _score_decline_bench_press,
    'deadlift': _score_deadlift,
    'romanian_deadlift': _score_romanian_deadlift,
    'hip_thrust': _score_hip_thrust,
    'shoulder_press': _score_shoulder_press,
    'lat_pulldown': _score_lat_pulldown,
    'lateral_raise': _score_lateral_raise,
    'leg_extension': _score_leg_extension,
    'leg_raises': _score_leg_raises,
    'lunges': _score_lunges,
    'plank': _score_plank,
    'pull_up': _score_pull_up,
    'chest_fly_machine': _score_chest_fly_machine,
    'jumping_jack': _score_jumping_jack,
    't_bar_row': _score_t_bar_row,
    'tricep_pushdown': _score_tricep_pushdown,
    'tricep_dips': _score_tricep_dips,
    'russian_twist': _score_russian_twist,
    # folder-name aliases for the 4-class setup
    'biceps': _score_barbell_biceps_curl,
    'triceps': _score_tricep_pushdown,
}

# Per-exercise pass thresholds
_PASS_THRESHOLDS = {
    'squat': 60,
    'push_up': 60,
    'barbell_biceps_curl': 60,
    'hammer_curl': 60,
    'bench_press': 60,
    'incline_bench_press': 60,
    'decline_bench_press': 60,
    'deadlift': 60,
    'romanian_deadlift': 60,
    'hip_thrust': 60,
    'shoulder_press': 60,
    'lat_pulldown': 60,
    'lateral_raise': 60,
    'leg_extension': 60,
    'leg_raises': 60,
    'lunges': 55,       # Lunges are harder to do perfectly
    'plank': 60,
    'pull_up': 60,
    'chest_fly_machine': 60,
    'jumping_jack': 60,
    't_bar_row': 65,    # Stricter, matching gym_analyzer's higher threshold
    'tricep_pushdown': 60,
    'tricep_dips': 60,
    'russian_twist': 60,
    'biceps': 60,
    'triceps': 60,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_form(exercise_name, angles):
    """Score exercise form using angle-based threshold rules.

    Args:
        exercise_name: one of EXERCISE_NAMES
        angles: (T, 12) or (12,) array of joint angles in degrees

    Returns:
        score: int 0-100
        feedback: list of feedback strings
    """
    score_fn = _SCORE_FUNCTIONS.get(exercise_name)
    if score_fn is None:
        return 100, []

    stats = _angle_stats(angles)
    score, feedback = score_fn(stats)
    score = max(0, min(100, score))  # Clamp to [0, 100]
    return score, feedback


def assess_form(exercise_name, angles):
    """Assess exercise form — returns (is_correct, feedback).

    Backward-compatible wrapper around score_form().

    Args:
        exercise_name: one of EXERCISE_NAMES
        angles: (T, 12) or (12,) array of joint angles in degrees

    Returns:
        is_correct: bool
        feedback: list of feedback strings
    """
    score, feedback = score_form(exercise_name, angles)
    threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
    return score >= threshold, feedback


def get_form_summary(exercise_idx, angles, exercise_names=None):
    """Get form assessment given exercise index and angles.

    Args:
        exercise_idx: predicted class index
        angles: (T, 12) or (12,) joint angles
        exercise_names: optional list of names (e.g. from config); if None, uses EXERCISE_NAMES
    """
    names = exercise_names if exercise_names is not None else EXERCISE_NAMES
    if exercise_idx < 0 or exercise_idx >= len(names):
        return {'exercise': 'unknown', 'is_correct': True, 'score': 100,
                'feedback': []}
    exercise_name = names[exercise_idx]
    score, feedback = score_form(exercise_name, angles)
    threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
    return {
        'exercise': exercise_name,
        'is_correct': score >= threshold,
        'score': score,
        'feedback': feedback,
    }


# ---------------------------------------------------------------------------
# Rep counting via angle-based phase detection
# ---------------------------------------------------------------------------

# Exercises with no clear rep pattern (isometric holds, etc.) — use window scoring only
NO_REP_EXERCISES = {'plank'}

# Debouncing: frames required in phase before state transition (reduces false reps)
REP_DEBOUNCE_FRAMES = 3

# Per-exercise phase thresholds for rep detection.
# angle_idx: which of the 12 joint angles to monitor
# contracted: angle must drop below this to enter the "contracted" phase
# extended: angle must rise above this to complete a rep (back to "extended")
def get_rep_angle(exercise_name, angles, frame_idx=0):
    """Get primary angle for rep counting (supports bilateral when configured)."""
    config = REP_PHASES.get(exercise_name)
    if config is None:
        return None
    if angles.ndim == 2:
        row = angles[frame_idx]
    else:
        row = angles
    idx = config['angle_idx']
    bilateral = config.get('bilateral')
    if bilateral and len(row) > max(bilateral):
        return float(np.mean([row[i] for i in bilateral]))
    return float(row[idx])


REP_PHASES = {
    'squat':               {'angle_idx': 6, 'bilateral': [6, 7], 'contracted': 100, 'extended': 150},
    'push_up':             {'angle_idx': 2, 'contracted': 110, 'extended': 155},
    'barbell_biceps_curl': {'angle_idx': 2, 'contracted': 60,  'extended': 140},
    'hammer_curl':         {'angle_idx': 2, 'contracted': 60,  'extended': 140},
    'shoulder_press':      {'angle_idx': 0, 'contracted': 100, 'extended': 160},
    'deadlift':            {'angle_idx': 8, 'contracted': 120, 'extended': 165},
    'romanian_deadlift':   {'angle_idx': 8, 'contracted': 120, 'extended': 165},
    'lunges':              {'angle_idx': 6, 'bilateral': [6, 7], 'contracted': 110, 'extended': 150},
    'bench_press':         {'angle_idx': 2, 'contracted': 100, 'extended': 155},
    'incline_bench_press': {'angle_idx': 2, 'contracted': 100, 'extended': 155},
    'decline_bench_press': {'angle_idx': 2, 'contracted': 100, 'extended': 155},
    'lat_pulldown':        {'angle_idx': 2, 'contracted': 80,  'extended': 150},
    'pull_up':             {'angle_idx': 2, 'contracted': 70,  'extended': 150},
    'lateral_raise':       {'angle_idx': 0, 'contracted': 30,  'extended': 75},
    'tricep_pushdown':     {'angle_idx': 2, 'contracted': 40,  'extended': 90},
    'tricep_dips':         {'angle_idx': 2, 'contracted': 110, 'extended': 155},
    'hip_thrust':          {'angle_idx': 4, 'contracted': 100, 'extended': 160},
    'leg_extension':       {'angle_idx': 6, 'bilateral': [6, 7], 'contracted': 100, 'extended': 150},
    'leg_raises':          {'angle_idx': 4, 'contracted': 100, 'extended': 150},
    'chest_fly_machine':   {'angle_idx': 0, 'contracted': 50,  'extended': 100},
    'jumping_jack':        {'angle_idx': 0, 'contracted': 40,  'extended': 120},
    't_bar_row':           {'angle_idx': 2, 'contracted': 70,  'extended': 140},
    'russian_twist':       {'angle_idx': 8, 'contracted': 110, 'extended': 140},
    # folder-name aliases for the 4-class setup
    'biceps':              {'angle_idx': 2, 'contracted': 60,  'extended': 140},
    'triceps':             {'angle_idx': 2, 'contracted': 40,  'extended': 90},
}


def segment_reps(exercise_name, angles, debounce_frames=REP_DEBOUNCE_FRAMES):
    """Segment a video's angles into reps with per-rep form labels.

    Uses a debounced state machine to detect rep boundaries, then scores
    each rep's angle window for form quality. Robust to jitter.

    Args:
        exercise_name: key from EXERCISE_NAMES
        angles: (T, 12) array of joint angles in degrees
        debounce_frames: frames required in phase before transition

    Returns:
        list of (start_frame, end_frame, is_correct, score) for each rep.
        Empty list if no reps detected or exercise has no rep pattern.
    """
    if exercise_name in NO_REP_EXERCISES:
        return []

    config = REP_PHASES.get(exercise_name)
    if config is None:
        return []

    if angles.ndim == 1:
        angles = angles[np.newaxis, :]
    T = angles.shape[0]
    if T < 5:
        return []

    idx = config['angle_idx']
    contracted = config['contracted']
    extended = config['extended']
    primary_angle = angles[:, idx]

    phase = 'extended'
    debounce_counter = 0
    rep_start_frame = 0
    reps = []

    for f in range(T):
        ang = primary_angle[f]
        if phase == 'extended':
            if ang < contracted:
                debounce_counter += 1
                if debounce_counter >= debounce_frames:
                    phase = 'contracted'
                    rep_start_frame = f - debounce_frames + 1
                    debounce_counter = 0
            else:
                debounce_counter = 0
        elif phase == 'contracted':
            if ang > extended:
                debounce_counter += 1
                if debounce_counter >= debounce_frames:
                    rep_end_frame = f
                    rep_angles = angles[rep_start_frame:rep_end_frame + 1]
                    score, _ = score_form(exercise_name, rep_angles)
                    threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
                    is_correct = 1 if score >= threshold else 0
                    reps.append((rep_start_frame, rep_end_frame, is_correct, score))
                    phase = 'extended'
                    debounce_counter = 0
            else:
                debounce_counter = 0

    return reps


def get_window_form_from_reps(window_start, window_end, rep_segments,
                               fallback_angles=None, exercise_name=None):
    """Get form label for a window from rep segments.

    Assigns the label of the rep the window overlaps most with.
    If overlap is insufficient, uses fallback (score_form on window).

    Args:
        window_start, window_end: window frame bounds (inclusive start, exclusive end)
        rep_segments: list of (start, end, is_correct) from segment_reps
        fallback_angles: (30, 12) angles for fallback scoring if no rep overlap
        exercise_name: for fallback scoring

    Returns:
        form: 0 or 1 (incorrect / correct)
    """
    if not rep_segments:
        if fallback_angles is not None and exercise_name is not None:
            score, _ = score_form(exercise_name, fallback_angles)
            threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
            return 1 if score >= threshold else 0
        return 1

    window_len = window_end - window_start
    best_overlap = 0
    best_label = 1

    for seg in rep_segments:
        rep_start, rep_end, is_correct = seg[0], seg[1], seg[2]
        overlap_start = max(window_start, rep_start)
        overlap_end = min(window_end, rep_end + 1)
        overlap = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap / window_len if window_len > 0 else 0
        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_label = is_correct

    if best_overlap >= 0.4:
        return best_label

    if fallback_angles is not None and exercise_name is not None:
        score, _ = score_form(exercise_name, fallback_angles)
        threshold = _PASS_THRESHOLDS.get(exercise_name, 60)
        return 1 if score >= threshold else 0
    return best_label


class RepCounter:
    """Counts exercise repetitions using angle-based phase detection.

    State machine:
        EXTENDED  --( angle < contracted_threshold )--> CONTRACTED
        CONTRACTED --( angle > extended_threshold )---> EXTENDED  (= 1 rep)

    Each completed rep is classified as correct or incorrect based on the
    form assessment at the time of completion.
    """

    def __init__(self):
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.phase = 'extended'

    def update(self, exercise_name, angle_value, is_correct):
        """Feed a new angle observation and update rep count.

        Args:
            exercise_name: exercise key from EXERCISE_NAMES
            angle_value: current value of the primary angle (degrees)
            is_correct: whether form is correct at this moment

        Returns:
            True if a rep was just completed, False otherwise
        """
        config = REP_PHASES.get(exercise_name)
        if config is None:
            return False

        rep_completed = False
        if self.phase == 'extended' and angle_value < config['contracted']:
            self.phase = 'contracted'
        elif self.phase == 'contracted' and angle_value > config['extended']:
            self.phase = 'extended'
            rep_completed = True
            if is_correct:
                self.correct_reps += 1
            else:
                self.incorrect_reps += 1
        return rep_completed

    @property
    def total_reps(self):
        return self.correct_reps + self.incorrect_reps
