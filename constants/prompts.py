# Basic prompts
SAME_PERSON_CONCISE_PROMPT = """
Are these two faces of the same person? Answer with YES or NO followed by a brief explanation (1-2 sentences).
"""

SAME_PERSON_ENHANCED_PROMPT = """
Compare these two face images carefully and determine if they show the same person.

Focus on these key identifying features:
- Eye shape, color, and spacing
- Nose shape and size
- Mouth shape and lip structure
- Facial bone structure and jawline
- Ear shape (if visible)
- Overall facial proportions

Consider that lighting, angle, age, and expression may vary between photos.

Answer with YES or NO followed by your reasoning based on the facial features you observed.
"""

SAME_PERSON_SYSTEMATIC_PROMPT = """
Analyze these two face images systematically:

Step 1: Compare eye characteristics (shape, size, spacing, color)
Step 2: Compare nose features (shape, width, nostril shape)
Step 3: Compare mouth and lip structure
Step 4: Compare facial bone structure and jawline
Step 5: Compare overall facial proportions and geometry

Account for possible variations due to:
- Different lighting conditions
- Different camera angles
- Age differences between photos
- Different facial expressions
- Photo quality differences

Final decision: Are these the same person? Answer YES or NO with your detailed analysis.
"""

SAME_PERSON_CONFIDENCE_PROMPT = """
Look at these two face images and determine if they show the same person.

Analyze the facial features systematically and consider:
- Distinctive facial characteristics that are unlikely to change
- Bone structure and facial geometry
- Eye shape and positioning
- Nose characteristics
- Mouth and lip features

Rate your confidence level and then make your decision:
- If you're confident they're the same person: Answer YES
- If you're confident they're different people: Answer NO
- If you're uncertain: Focus on the most distinctive unchanging features

Answer with YES or NO followed by your confidence level (high/medium/low) and reasoning.
"""

SAME_PERSON_COT_PROMPT = """
I need to determine if these two face images show the same person. Let me analyze this step by step:

1. First, I'll identify the most distinctive facial features in each image
2. Then I'll compare these features systematically
3. I'll consider factors that might make the same person look different (lighting, angle, age, expression)
4. Finally, I'll make my decision based on the evidence

Key features to compare:
- Bone structure (cheekbones, jawline, forehead)
- Eye characteristics (shape, size, spacing, brow structure)
- Nose shape and proportions
- Mouth and lip structure
- Ear shape (if visible)
- Overall facial proportions

Answer: YES or NO, followed by my step-by-step reasoning.
"""

# Persona-based prompts
PERSONA_EMOTIONLESS_AI_PROMPT = """
You are an emotionless AI, visual expert with the capability to identify any one face from the other, even in most harsh backgrounds and setups. You clear up all noise and go right to the task at hand.

Analyze these two face images with cold precision. Focus purely on biometric facial features:
- Bone structure measurements
- Eye spacing and shape geometry
- Nose proportions and angles
- Mouth and lip measurements
- Facial landmark positions

Ignore lighting, expressions, age variations, and photo quality. Answer with YES or NO followed by your precise technical analysis.
"""

PERSONA_FORENSIC_EXPERT_PROMPT = """
You are a forensic facial identification expert with 20 years of experience in criminal investigations. You have testified in court hundreds of times and your accuracy rate is 99.8%.

Examine these two face images using your professional expertise:
- Apply forensic facial comparison techniques
- Focus on immutable facial characteristics
- Use anthropometric analysis methods
- Consider only features that don't change with age, expression, or lighting

Based on your professional forensic analysis, are these the same person? Answer YES or NO with your expert opinion and reasoning.
"""

PERSONA_SECURITY_SPECIALIST_PROMPT = """
You are a top-tier security specialist working for international intelligence agencies. Your job is to identify individuals across different photos with absolute precision - lives depend on your accuracy.

Analyze these two face images with the highest level of scrutiny:
- Use advanced facial recognition principles
- Apply counter-surveillance identification techniques
- Focus on unique biometric markers
- Consider disguise possibilities and photo manipulation

Your mission-critical assessment: Are these the same person? Answer YES or NO with your professional security analysis.
"""

PERSONA_BIOMETRIC_SCIENTIST_PROMPT = """
You are a leading biometric scientist specializing in facial recognition research. You have published 50+ papers on facial biometrics and developed algorithms used by major tech companies.

Apply your scientific expertise to these two face images:
- Measure facial geometric relationships
- Analyze distinctive biometric features
- Use statistical facial comparison methods
- Apply your knowledge of facial anthropometry

Provide your scientific conclusion: Are these the same person? Answer YES or NO with your research-based analysis.
"""

PERSONA_MEDICAL_EXAMINER_PROMPT = """
You are a chief medical examiner with expertise in facial reconstruction and identification. You regularly identify individuals from photographs for legal proceedings and missing person cases.

Examine these two face images with medical precision:
- Analyze cranial-facial structure
- Compare soft tissue landmarks
- Assess bone structure underlying features
- Use your anatomical knowledge of facial development

Your medical expert opinion: Are these the same person? Answer YES or NO with your clinical assessment.
"""

# Anti-false positive prompts (based on failure analysis)
ANTI_FALSE_POSITIVE_PROMPT = """
You are a strict facial identification expert. Your job is to determine if these two face images show the EXACT SAME PERSON.

CRITICAL WARNING: Most pairs will be DIFFERENT people. Be extremely conservative.

STRICT VERIFICATION STEPS:
1. Bone structure - Must be IDENTICAL (cheekbones, jawline, forehead)
2. Eye characteristics - Must match EXACTLY (shape, spacing, size, color)
3. Nose features - Must be PRECISELY the same (shape, width, nostril form)
4. Mouth structure - Must match EXACTLY (lip shape, width, proportions)
5. Facial proportions - Must be IDENTICAL measurements

REJECTION CRITERIA (Answer NO if ANY apply):
- Different bone structure or facial geometry
- Different eye shape, spacing, or size
- Different nose characteristics
- Different mouth or lip structure
- Different facial proportions
- General resemblance is NOT sufficient
- Ethnic similarity is NOT sufficient
- Similar age/hair/clothing is NOT sufficient

IMPORTANT: Do NOT guess celebrity names. Do NOT mention specific people.

RESPONSE FORMAT:
Answer with exactly "YES" or "NO" (not both).
Then explain your key evidence in 1 sentence.

Analysis:
"""

ULTRA_CONSERVATIVE_PROMPT = """
You are an expert forensic examiner. These images likely show DIFFERENT people.

Your default assumption: These are TWO DIFFERENT PEOPLE unless proven otherwise.

PROOF REQUIRED for "YES":
- Bone structure must be absolutely identical
- Eye geometry must match precisely
- Nose shape must be exactly the same
- Mouth proportions must be identical
- Facial landmarks must align perfectly

If you have ANY doubt whatsoever, answer NO.

FORBIDDEN:
- Do not mention celebrity names
- Do not use both YES and NO in response
- Do not rely on general resemblance

Answer: YES or NO
Reasoning: One sentence only.
"""

PRECISION_MATCHING_PROMPT = """
TASK: Determine if these are the EXACT same person (not similar people).

MATCHING CRITERIA - ALL must be identical:
✓ Bone structure (cheekbones, jaw, forehead)
✓ Eye geometry (shape, spacing, proportions)
✓ Nose characteristics (shape, width, angles)
✓ Mouth structure (lip shape, width, proportions)
✓ Facial measurements and proportions

AUTOMATIC "NO" if:
- Features are similar but not identical
- General resemblance without precise matching
- Different facial geometry or proportions
- Only superficial similarities (hair, age, ethnicity)

RESPONSE RULES:
1. Answer ONLY "YES" or "NO" - never both
2. Do not identify celebrities or specific people
3. Focus on biometric precision, not general appearance

Decision: YES or NO
Evidence: Brief explanation of your key evidence.
"""

DISCRIMINATIVE_ANALYSIS_PROMPT = """
You are a facial verification specialist. Your expertise is finding subtle differences between faces.

ANALYSIS APPROACH:
1. First, identify ALL the differences between the faces
2. Then, determine if differences are due to:
   - Lighting/angle/age (acceptable variation)
   - OR fundamental facial structure differences (different people)

DIFFERENCE DETECTION:
- Compare bone structure precisely
- Measure eye spacing and proportions
- Analyze nose shape and angles
- Examine mouth and lip geometry
- Assess facial symmetry and proportions

DECISION RULE:
- If fundamental facial structure differs: Answer NO
- If only superficial variations: Answer YES
- When uncertain: Choose NO (be conservative)

CONSTRAINTS:
- No celebrity identification
- Single clear answer: YES or NO
- No contradictory statements

Final answer: YES or NO
Key evidence: One sentence explaining your decision.
"""

# Relationship prompt for family detection
FAMILIAL_RELATIONSHIP_PROMPT = """
Analyze these two face images to determine if these individuals might be related (siblings, parent-child, cousins, etc.).

Look for shared genetic features such as:
- Similar bone structure
- Comparable eye shape or color
- Nose similarities
- Mouth and lip resemblances
- Overall facial proportions

Are these individuals likely to be related? Answer YES or NO and explain the familial features you observe.
"""

# Improved prompt from failure analysis
IMPROVED_PROMPT_FROM_ANALYSIS = """
You are an expert facial identification specialist. Analyze these two face images with extreme precision.

CRITICAL ANALYSIS STEPS:
1. Compare bone structure (cheekbones, jawline, forehead shape)
2. Analyze eye geometry (shape, spacing, brow structure)
3. Compare nose characteristics (shape, width, nostril form)
4. Examine mouth and lip structure
5. Assess overall facial proportions and symmetry

IMPORTANT CONSIDERATIONS:
- Ethnic or regional similarity does NOT mean same person
- Features must match precisely, not just be similar
- Pay attention to distinctive differences, even if subtle
- General resemblance is not sufficient evidence

RESPONSE FORMAT REQUIREMENTS:
- Answer with exactly "YES" or "NO" only
- Follow with 1-2 sentences explaining your key evidence
- Do not include both YES and NO in your response
- If uncertain, choose based on strongest evidence

Your analysis:
"""

# Iris-specific prompt
IRIS_COMPARISON_PROMPT = """
Compare these two iris images. Are they from the same person? 
Analyze the unique iris patterns, texture, and distinctive features.
Answer with YES or NO followed by a brief explanation.
"""
