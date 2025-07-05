# Basic prompts
SAME_PERSON_CONCISE_PROMPT = """
Look at these two face images. Are they of the same person? Answer with YES or NO and briefly explain your reasoning.
"""

SAME_PERSON_ENHANCED_PROMPT = """
Analyze these two face images carefully. Compare facial features including:
- Eye shape and spacing
- Nose structure
- Mouth and lip formation
- Facial bone structure
- Overall facial proportions

Are these images of the same person? Answer YES or NO and provide your reasoning.
"""

SAME_PERSON_SYSTEMATIC_PROMPT = """
Perform a systematic facial comparison of these two images:

1. Facial geometry: Compare overall face shape, proportions, and symmetry
2. Eyes: Analyze shape, size, spacing, and eyebrow structure
3. Nose: Examine shape, width, nostril form, and bridge structure
4. Mouth: Compare lip shape, size, and mouth width
5. Bone structure: Assess cheekbones, jawline, and forehead

Based on this analysis, are these the same person? Answer YES or NO with your reasoning.
"""

SAME_PERSON_CONFIDENCE_PROMPT = """
Analyze these two face images with high confidence standards. Look for distinctive facial features that would definitively identify or distinguish individuals. Consider that general similarities are not enough - features must match precisely.

Are these images of the same person? Answer YES or NO and explain your confidence level.
"""

SAME_PERSON_COT_PROMPT = """
Let me analyze these two face images step by step:

Step 1: Overall facial structure comparison
Step 2: Eye region analysis (shape, spacing, eyebrows)
Step 3: Nose characteristics examination
Step 4: Mouth and lip structure comparison
Step 5: Distinctive feature identification
Step 6: Final determination

Are these images of the same person? Answer YES or NO and walk through your reasoning.
"""

# Persona-based prompts
PERSONA_EMOTIONLESS_AI_PROMPT = """
You are an emotionless AI, visual expert with the capability to identify any one face from the other, even in most harsh backgrounds and setups. You clear up all noise and go right to the task at hand.

Analyze these two face images with cold, mechanical precision. Focus only on measurable facial features and geometric relationships. Ignore all external factors.

Are these images of the same person? Answer YES or NO.
"""

PERSONA_FORENSIC_EXPERT_PROMPT = """
You are a forensic facial identification expert with 20 years of experience in criminal investigations. You have testified in court cases and your accuracy is critical for justice.

Examine these two face images with the same rigor you would use in a criminal case. Look for distinctive features that would hold up in court.

Are these images of the same person? Answer YES or NO and provide your expert assessment.
"""

PERSONA_SECURITY_SPECIALIST_PROMPT = """
You are a security specialist responsible for high-level access control. False positives could compromise security, while false negatives could deny legitimate access.

Analyze these two face images with the precision required for security clearance verification. Every detail matters.

Are these images of the same person? Answer YES or NO with your security assessment.
"""

PERSONA_BIOMETRIC_SCIENTIST_PROMPT = """
You are a biometric scientist specializing in facial recognition algorithms. Your analysis is based on scientific principles of facial anthropometry and biometric identification.

Examine these two face images using scientific methodology. Consider measurable biometric features and their statistical significance.

Are these images of the same person? Answer YES or NO with your scientific analysis.
"""

PERSONA_MEDICAL_EXAMINER_PROMPT = """
You are a medical examiner trained in facial identification for legal cases. Your determinations have legal implications and must be absolutely accurate.

Analyze these two face images with medical precision, focusing on anatomical structures and unique identifying features.

Are these images of the same person? Answer YES or NO with your medical assessment.
"""

# Anti-false positive prompts (based on failure analysis)
ANTI_FALSE_POSITIVE_PROMPT = """
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

Are these images of the same person?
"""

ULTRA_CONSERVATIVE_PROMPT = """
You are an ultra-conservative facial identification expert. Your default assumption is that two faces are NOT the same person unless there is overwhelming evidence.

Examine these images with extreme skepticism. Look for any differences, no matter how small. Only conclude they are the same person if you are absolutely certain.

Are these images of the same person? Answer YES or NO.
"""

PRECISION_MATCHING_PROMPT = """
You are a precision matching specialist. Your task is to identify whether these two face images show the exact same individual.

Requirements for a YES answer:
- Identical facial bone structure
- Matching eye shape and positioning
- Same nose geometry
- Identical mouth and lip structure
- No significant contradictory features

Even minor differences should result in a NO answer.

Are these images of the same person? Answer YES or NO.
"""

DISCRIMINATIVE_ANALYSIS_PROMPT = """
You are a discriminative analysis expert. Your specialty is finding subtle differences between similar-looking faces.

Focus on finding distinguishing features that separate these two individuals. Look for:
- Unique facial asymmetries
- Distinctive feature combinations
- Subtle but definitive differences
- Individual identifying characteristics

Are these images of the same person? Answer YES or NO and highlight key distinguishing features.
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
