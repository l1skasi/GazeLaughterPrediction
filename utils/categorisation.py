# Categorize Facial@MOT annotations
facial_mapping = {
    "smile showing teeth": "positive",
    "smile": "positive",
    "widely opened eyes": "positive",
    "O-shaped mouth": "neutral",
    "neutral": "neutral",
    "biting lower lip": "neutral",
    "invisible": "invisible"
}

# Utterance categories
questions = [
    "Ready?",
    "Can I please have a ball?",
    "What even set off you laughing so hard?"
]
commands = [
    "Drop it.",
    "Drop it in.",
    "Put it in here.",
    "Can you sit up?",
]
affection = [
    "I love you."
]

playful_sounds = [
    "chooga",
    "... tickle you",
    "Yeaaah!"
]

statements = [
    "She’s laughing so hard that she literally can’t even stay seated.",
    "I hope this is not the laugh before the cry.",
    "Really far.",
    "Too far.",
    "I take this one.",
    "This is harmonious to hear.",
    "Keep falling."
]

utterance_lists = {
    'question': questions,
    'statement': statements,
    'affection': affection,
    'playful_sounds': playful_sounds
}

utterance_category_mapping = {value: category for category, values in utterance_lists.items() for value in values}

# Gaze@MOT annotations categorizations
gaze_mot_mapping = {
    "child": "child",
    "invisible": "invisible",
    "ball": "main_object",
    "toys": "object",
    "into the distance": "distracted",
    "dad": "distracted",
    "floor": "distracted",
    "glass": "object",
    "toy": "object",
    "aside": "distracted",
}

# Gaze@CHI annotations categorizations
gaze_chi_mapping = {
    "mom": "mom",
    "mom's hand": "mom",
    "mom's legs": "mom",
    "ball": "main_object",
    "floor": "distracted",
    "toys": "object",
    "toy": "object",
    "glass": "object",
    "aside": "distracted",
    "ceiling": "distracted",
    "dad": "distracted"
}

def categorize_facial_mot(facial_annotation):
    return facial_mapping.get(facial_annotation, "neutral")

def categorize_utterance_mot(value):
    return utterance_category_mapping.get(value, "statement")

def categorize_gaze_mot(gaze_annotation):
    return gaze_mot_mapping.get(gaze_annotation, "distracted")

def categorize_gaze_chi(gaze_annotation):
    return gaze_chi_mapping.get(gaze_annotation, "distracted")