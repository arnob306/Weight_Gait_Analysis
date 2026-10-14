"""
Prompt variants for the Gemma gait experiments. One question, four framings.

What these change is the STRUCTURE of the question, not claims about the
classes. None says what a limp or a shuffle looks like, because the measured
kinematics on the 30 development clips do not support such cues: only 1 of 27
attributes separates the classes at p < 0.05 (about the number expected by
chance), and shuffle clips do not show lower foot lift. A prompt asserting
class-specific cues would be asserting something this data does not show.

Every prompt keeps the original label meanings, and every JSON schema lists
"reasoning" BEFORE "weight_class". The earlier prompt asked for the class
first, so the model committed to an answer before writing any evidence and
its reasoning could only justify it afterwards.

All prompts parse with harness.parse_model_response (needs weight_class and
reasoning; extra keys such as the per-leg flags are ignored by it).

P2_guided_per_leg and P3_guided_balanced are the long, rule-based prompts. Their
cue wording (shorter step, low or dragging foot, less knee bend in swing, shortened
weight-bearing time giving uneven timing between legs, a limp as an asymmetry and a
shuffle as short low steps) is standard descriptive language from the clinical gait
literature, listed here for the report. P3_guided_balanced names its sources in the
prompt text (titles, no links, since the model cannot open them); P2 does not. The full
texts could not be opened here, so check each phrase against the source before the
report says it was "derived from" it:
  - Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function, 2nd ed.
    SLACK; 2010. https://www.routledge.com/Gait-Analysis-Normal-and-Pathological-Function/Perry-Burnfield/p/book/9781556427664
  - StatPearls, Antalgic Gait in Adults (stance phase shortened relative to swing).
    https://www.ncbi.nlm.nih.gov/books/NBK559243/
  - StatPearls, Gait Disturbances (including shuffling and festinant gait).
    https://www.ncbi.nlm.nih.gov/books/NBK560610/
These cues are generic, not measured on this dataset (only 1 of 27 measured attributes
separated the classes at p < 0.05), so the report should present them as an
expert-guided prompt condition, not as evidence about these clips.
"""
from experiment import PROMPT as RUBRIC_PROMPT

_INTRO = (
    "You are an expert in clinical gait analysis. You are shown a sequence of frames, "
    "in time order, taken at regular intervals from a video of a person walking. "
)
_CLASSES = (
    "0 = normal walking, 1 = limping (one leg affected), 2 = shuffling (both legs affected)"
)

PROMPTS = {
    # P0: the prompt used for every Qwen result. Class first, then reasoning.
    "P0_rubric": RUBRIC_PROMPT,

    # P1: same rubric, but the model must compare the legs and write the
    # evidence before it is allowed to give a class.
    "P1_reasoning_first": (
        _INTRO
        + "Examine the left and right leg separately across the frames: step length, how far "
        "the knee bends, how high the foot lifts and the timing of each step, and note the arm "
        "swing. Judge the walking movement itself, not clothing, footwear or background. "
        f"Then classify the gait as exactly one of: {_CLASSES}. Write your reasoning first, "
        "then the class. Respond only with valid JSON in this exact format: "
        '{"reasoning": "<2-3 sentences comparing the left leg, the right leg and the arm swing>", '
        '"weight_class": <0, 1, or 2>}'
    ),

    # P2: guided, per leg. Long form in the style of a task statement, a field
    # list and a GUIDANCE block of short rules. The model judges each leg in
    # turn, then the class follows from how many legs are impaired.
    "P2_guided_per_leg": (
        _INTRO
        + "Your job is to judge how EACH LEG moves, one leg at a time, and then say how many "
        "legs look impaired. You are not asked to name a cause or a medical condition.\n\n"
        "Return ONE JSON object with these fields, in this order:\n"
        "- reasoning: 2-3 sentences. Describe the walker's left leg, then the right leg, then "
        "compare the two. Mention only what you can actually see in the frames.\n"
        "- left_leg_impaired: true if the left leg's movement looks impaired, otherwise false\n"
        "- right_leg_impaired: true if the right leg's movement looks impaired, otherwise false\n"
        f"- weight_class: the integer 0, 1 or 2 only, never a word. {_CLASSES}\n\n"
        "GUIDANCE:\n"
        "- Left and right mean the walker's own left and right legs.\n"
        "- Judge each leg against the other leg and against typical walking. Features that "
        "clinical gait descriptions use for an impaired leg are: a shorter step, a foot that "
        "lifts little or drags, less knee bend while the leg swings forward, and less time "
        "spent bearing weight on that leg, which makes the timing of the two legs uneven.\n"
        "- Count the legs. If only one leg shows such features, or one leg clearly shows them "
        "more than the other, that is one impaired leg (class 1). If both legs show them, that "
        "is class 2, even when the two legs look alike. If neither shows them, it is class 0.\n"
        "- Legs that look identical are not automatically normal. Two legs that both take "
        "short steps or lift the foot little, compared with typical walking, are both impaired.\n"
        "- Judge the movement only. Ignore clothing, footwear, anything attached to the legs, "
        "the background and who the person is.\n"
        "- The frames are spaced about one second apart, so you will not see every step. Rely "
        "on differences that persist across several frames, not on a single pose.\n"
        "- If a leg is hidden or unclear in some frames, judge it from the frames where it is "
        "visible and say so in your reasoning. Do not assume impairment from one unclear frame.\n"
        "- Write the evidence in 'reasoning' before you decide the leg flags and the class.\n\n"
        "Output the JSON object only: no other text and no markdown fences."
    ),

    # P3: guided, whole gait, with the class balance and a final self-check.
    # The balance sentence states a fact about the evaluation sets (both are
    # exactly balanced), so it must be disclosed in the methods.
    "P3_guided_balanced": (
        _INTRO
        + "Your job is to classify the whole gait as one of three classes. About one third of "
        "the videos you are shown are normal, one third have one impaired leg and one third "
        "have both legs impaired, so no answer is more likely than another before you look.\n\n"
        "Return ONE JSON object with these fields, in this order:\n"
        "- reasoning: 2-3 sentences comparing the left leg, the right leg and the arm swing, "
        "using only what you can see in the frames.\n"
        f"- weight_class: the integer 0, 1 or 2 only, never a word. {_CLASSES}\n"
        "- confidence: high, medium or low\n\n"
        "GUIDANCE:\n"
        "- Work in this order: first look at the left leg across the frames, then the right "
        "leg, then compare them, then look at the arm swing, and only then choose the class.\n"
        "- Features that clinical gait descriptions use for an impaired leg are: a shorter "
        "step, a foot that lifts little or drags, less knee bend while the leg swings forward, "
        "and less time spent bearing weight on that leg, which makes the timing of the two "
        "legs uneven.\n"
        "- A limp (class 1, one leg affected) is described in the clinical literature as an "
        "antalgic gait: the stance phase on the affected leg is shortened relative to the swing "
        "phase, so the person spends less time bearing weight on that leg, the two legs fall "
        "out of step with each other, and the step on the opposite side is shorter. The key "
        "sign is that the two legs differ. (StatPearls, 'Antalgic Gait in Adults'; Perry and "
        "Burnfield, 'Gait Analysis: Normal and Pathological Function', 2nd ed., 2010.)\n"
        "- A shuffle (class 2, both legs affected) is described in clinical gait literature, "
        "for example in Parkinson disease, as short steps, little lift of the foot so the toes "
        "barely clear the ground, and reduced arm swing, on both sides and not on one side "
        "only. (StatPearls, 'Gait Disturbances'; Parkinson disease gait descriptions, for "
        "example 'Compensation Strategies for Gait Impairments in Parkinson's Disease'.)\n"
        "- These descriptions come from clinical populations. Report a feature only if you can "
        "actually see it in these frames. Do not assume it is there because the description "
        "says so.\n"
        "- Do not default to normal. If you cannot find a difference between the legs, look "
        "again at both legs for short steps or a low foot before deciding the gait is normal.\n"
        "- Do not default to any other single class either. Each class is equally likely.\n"
        "- Judge the movement only. Ignore clothing, footwear, anything attached to the legs, "
        "the background and who the person is.\n"
        "- The frames are spaced about one second apart, so you will not see every step. Rely "
        "on differences that persist across several frames, not on a single pose.\n"
        "- Use 'low' confidence when the legs are hard to see or the frames disagree. A low "
        "confidence answer is acceptable; a confident guess with no visible evidence is not.\n"
        "- Write the evidence in 'reasoning' before you decide the class.\n\n"
        "SOURCES for the descriptions above:\n"
        "- Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function, 2nd ed. "
        "SLACK; 2010.\n"
        "- StatPearls (NCBI Bookshelf). Antalgic Gait in Adults.\n"
        "- StatPearls (NCBI Bookshelf). Gait Disturbances.\n"
        "- Compensation Strategies for Gait Impairments in Parkinson's Disease: From "
        "Underlying Mechanisms to Daily Clinical Practice (PubMed Central).\n\n"
        "Output the JSON object only: no other text and no markdown fences."
    ),
}
