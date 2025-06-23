"""
This module stores the prompt templates for the different models.
"""

# ##################################################################
# BLIP Prompts
# ##################################################################

BLIP_PROMPTS = {
    # --- Original Prompts ---
    "one_step_vanilla": "Question: Does the image and text match? Answer:",
    "one_step_detailed": "Question: Based on the image, is the accompanying text '{text}' an accurate and truthful description? Answer with 'Yes' or 'No'. Answer:",
    "two_step_decision": "Based on the image, does the text '{text}' accurately describe the content? Answer 'Yes' or 'No'.",
    "two_step_explanation_yes": "Please provide a brief explanation for why the text is accurate based on the image.",
    "two_step_explanation_no": "Please provide a brief explanation for why the text is not accurate based on the image.",

    # --- Zero-Shot Prompts ---
    "zs_role_play": "You are a fact-checking expert. Analyze the image and text provided. Is the text a truthful and accurate description of the image content? Answer with 'Yes' or 'No' and provide a brief justification.\nText: '{text}'\nAnswer:",
    "zs_cot": "Analyze the following image and text. First, describe the key elements in the image. Second, compare these elements to the claim in the text. Finally, conclude if the text is a completely accurate description of the image.\nText: '{text}'\nAnalysis:",
    "zs_step_by_step": "Step 1: What is the main claim being made by the text '{text}'? Step 2: Does the image contain visual evidence that directly supports this claim? Step 3: Are there any contradictions or missing elements? Step 4: Based on this, is the text an accurate description? Answer:",
    "zs_forced_choice": "Is the text '{text}' a 'Accurate' or 'Inaccurate' description of the provided image? Only answer with one of these two words.",
    "zs_simple_question": "Image and text are provided. Is the text a correct description of the image?\nText: \"{text}\"\nAnswer:",
    "zs_yesno_justification": (
        "Text: {text}\n"
        "Metadata: {metadata}\n"
        "Does the text match the image and metadata?\n"
        "Please answer with only 'Yes.' or 'No.' on the first line, then a short justification on the next line.\n"
        "Do not repeat the question. Do not include any other text.\n"
        "Answer:"
    ),

    # --- Few-Shot Prompts ---
    "fs_vanilla": "Example 1:\nText: '{real_example_text}'\nAnswer: {real_explanation_blip}\n\nExample 2:\nText: '{fake_example_text}'\nAnswer: {fake_explanation_blip}\n\nYour Turn:\nText: '{{text}}'\nAnswer:",
    "fs_step_by_step": "Example 1:\nText: \"{real_example_text}\"\nExplanation: {real_explanation_blip}\n\nExample 2:\nText: \"{fake_example_text}\"\nExplanation: {fake_explanation_blip}\n\n---\n\nYour Turn:\nText: \"{text}\"\nStep 1: What is the main claim being made by the text? Step 2: Does the image contain visual evidence that directly supports this claim? Step 3: Are there any contradictions or missing elements? Step 4: Based on this, is the text an accurate description? Answer:",
    "fs_yesno_justification": (
        "Example 1:\n"
        "Text: 'A dog sitting on a couch'\n"
        "Metadata: location: living room; time: day\n"
        "Does the text match the image and metadata?\n"
        "Answer:\nYes.\nThe image shows a dog on a couch in a living room during the day.\n\n"
        "Example 2:\n"
        "Text: 'A cat playing the piano'\n"
        "Metadata: location: concert hall; time: night\n"
        "Does the text match the image and metadata?\n"
        "Answer:\nNo.\nThe image does not show a cat or a piano.\n\n"
        "Now, answer the following:\n"
        "Text: {text}\n"
        "Metadata: {metadata}\n"
        "Does the text match the image and metadata?\n"
        "Please answer with only 'Yes.' or 'No.' on the first line, then a short justification on the next line.\n"
        "Do not repeat the question. Do not include any other text.\n"
        "Answer:"
    ),

    # --- Metadata-aware Prompt ---
    "zs_metadata_check": "Text: '{text}' | Metadata: {metadata} | Does the text match the image and metadata? Explain.",
}

# ##################################################################
# LLaVA Prompts
# ##################################################################

LLAVA_PROMPTS = {
    # --- Zero-Shot Prompts ---
    "zs_role_play": "USER: <image>\nYou are a fact-checking expert. Analyze the image and text provided. Is the text a truthful and accurate description of the image content? Provide a brief justification.\nText: '{text}'\nASSISTANT:",
    "zs_cot": "USER: <image>\nAnalyze the following image and text. First, describe the key elements in the image. Second, compare these elements to the claim in the text. Finally, conclude if the text is a completely accurate description of the image.\nText: '{text}'\nASSISTANT:",
    "zs_step_by_step": "USER: <image>\nStep 1: What is the main claim being made by the text '{text}'? Step 2: Does the image contain visual evidence that directly supports this claim? Step 3: Are there any contradictions or missing elements? Step 4: Based on this, is the text an accurate description? Provide your reasoning.\nASSISTANT:",
    "zs_forced_choice": "USER: <image>\nIs the text '{text}' an 'Accurate' or 'Inaccurate' description of the provided image? Only answer with one of these two words.\nASSISTANT:",
    "zs_simple_question": "USER: <image>\nImage and text are provided. Is the text a correct description of the image?\nText: \"{text}\"\nASSISTANT:",

    # --- Few-Shot Prompts ---
    "fs_vanilla": "USER: <image>\nText: \"{real_example_text}\"\nASSISTANT: {real_explanation_llava}\n\nUSER: <image>\nText: \"{fake_example_text}\"\nASSISTANT: {fake_explanation_llava}\n\nUSER: <image>\nText: \"{text}\"\nASSISTANT:",
    "fs_step_by_step": "USER: <image>\nText: \"{real_example_text}\"\nASSISTANT: Step 1: The text claims there is a 'concerned sink with a tiny hat'. Step 2: The image shows a sink that looks like a face, and a small object on top resembles a hat. Step 3: There are no contradictions. The description is humorous but visually grounded. Step 4: The text is an accurate description.\n\nUSER: <image>\nText: \"{fake_example_text}\"\nASSISTANT: Step 1: The text claims the pill is an off-brand Mucinex with rearranged letters. Step 2: The image shows a standard, genuine Mucinex pill with the correct branding. Step 3: The visual evidence directly contradicts the claim of it being 'off-brand' with different lettering. Step 4: The text is an inaccurate description.\n\nUSER: <image>\nText: \"{text}\"\nASSISTANT: Step 1: What is the main claim being made by the text? Step 2: Does the image contain visual evidence that directly supports this claim? Step 3: Are there any contradictions or missing elements? Step 4: Based on this, is the text an accurate description? Answer:",

    # --- Metadata-aware Prompt ---
    "zs_metadata_check": "USER: <image>\nText: '{text}'\nMetadata: {metadata}\nDoes the text match the image and metadata? Provide a justification.\nASSISTANT:",
}

# ##################################################################
# Few-Shot Example Data
# ##################################################################

FEW_SHOT_EXAMPLES = [
    {
        "name": "real_example",
        "text": "a dog sitting on a couch",
        "image_path": "data/downloaded_fakeddit_images/8deric.jpg",
        "explanation_blip": "Yes. The image shows a dog sitting on a couch, matching the text.",
        "explanation_llava": "The text accurately describes the image, which shows a dog on a couch."
    },
    {
        "name": "fake_example",
        "text": "a cat playing the piano",
        "image_path": "data/downloaded_fakeddit_images/5xv0cy.jpg",
        "explanation_blip": "No. The image does not show a cat playing the piano, so the text is inaccurate.",
        "explanation_llava": "The text does not match the image, which does not show a cat or a piano."
    }
]