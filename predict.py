import pickle
import pandas as pd
import sys
from pathlib import Path
from difflib import get_close_matches
import matplotlib.pyplot as plt

MODEL_FILE = "model.pkl"
ENCODER_FILE = "label_encoder.pkl"
FEATURES_FILE = "model_features.pkl"
ACCURACY_FILE = "model_accuracy.pkl"

# --------------------- Grouped features ---------------------
feature_groups = {
    "General": ["itching","fatigue","malaise","weight_gain","weight_loss","anxiety",
                "restlessness","lethargy","depression","irritability","lack_of_concentration"],
    "Skin": ["skin_rash","nodal_skin_eruptions","red_spots_over_body","dischromic _patches",
             "drying_and_tingling_lips","skin_peeling","blister","yellow_crust_ooze"],
    "Fever & Cold": ["continuous_sneezing","shivering","chills","high_fever","mild_fever",
                     "sunken_eyes","sweating","runny_nose","congestion"],
    "Pain & Muscles": ["joint_pain","muscle_pain","back_pain","neck_pain","knee_pain","hip_joint_pain",
                       "muscle_weakness","stiff_neck","swelling_joints","movement_stiffness"],
    "Digestive": ["stomach_pain","acidity","vomiting","indigestion","diarrhoea","constipation",
                  "abdominal_pain","pain_behind_the_eyes","belly_pain","pain_during_bowel_movements"],
    "Urinary": ["burning_micturition","spotting_ urination","bladder_discomfort",
                "foul_smell_of urine","continuous_feel_of_urine","passage_of_gases","polyuria"],
    "Eyes & Vision": ["watering_from_eyes","blurred_and_distorted_vision","redness_of_eyes",
                      "visual_disturbances","loss_of_smell"],
    "Heart & Circulation": ["fast_heart_rate","palpitations","swollen_blood_vessels","prominent_veins_on_calf"],
    "Other": ["ulcers_on_tongue","muscle_wasting","cough","phlegm","throat_irritation",
              "chest_pain","toxic_look_(typhos)","altered_sensorium","coma"]
}

# --------------------- Load artifacts ---------------------
def load_artifacts():
    for f in [MODEL_FILE, ENCODER_FILE, FEATURES_FILE, ACCURACY_FILE]:
        if not Path(f).exists():
            sys.exit(f"Error: Missing file {f}. Run train_model.py first.")
    model = pickle.load(open(MODEL_FILE, "rb"))
    label_encoder = pickle.load(open(ENCODER_FILE, "rb"))
    features = pickle.load(open(FEATURES_FILE, "rb"))
    accuracy = pickle.load(open(ACCURACY_FILE, "rb"))
    return model, label_encoder, features, accuracy

# --------------------- Prepare input ---------------------
def prepare_input_dict(input_dict, features):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    return df

# --------------------- Display graph ---------------------
def show_prediction_graph(model, le, X):
    y_prob = model.predict_proba(X)[0]
    classes = le.classes_
    top_idx = y_prob.argsort()[-5:][::-1]
    top_probs = y_prob[top_idx]
    top_classes = [classes[i] for i in top_idx]

    plt.figure(figsize=(8,5))
    plt.barh(top_classes, top_probs, color='skyblue')
    plt.xlabel("Probability")
    plt.title("Top 5 Predicted Diseases")
    plt.gca().invert_yaxis()
    plt.show()

# --------------------- Smart typo correction ---------------------
def smart_feature_match(user_input, valid_features):
    item = user_input.strip().replace(' ', '_').lower()
    if item in valid_features:
        return item
    match = get_close_matches(item, valid_features, n=1, cutoff=0.7)
    if match:
        print(f"Using '{match[0]}' for '{user_input}'")
        return match[0]
    else:
        print(f"Ignored: '{user_input}'")
        return None

# --------------------- Interactive mode ---------------------
def interactive_mode():
    model, le, features, accuracy = load_artifacts()
    input_dict = {col: 0 for col in features if col != 'prognosis'}

    print(f"\n Model Accuracy: {accuracy*100:.2f}%\n")
    print("🩺 Enter your symptoms/features (comma-separated).")
    print("You can also type numeric features like temperature=101.5\n")

    for group, feats in feature_groups.items():
        print(f"{group}:")
        for i in range(0, len(feats), 6):
            print(", ".join(feats[i:i+6]))
        print()

    user_input = input("Your input: ").strip()
    if user_input:
        for item in user_input.split(','):
            item = item.strip()
            if '=' in item:
                key, val = item.split('=', 1)
                key = smart_feature_match(key, input_dict.keys())
                if key:
                    try:
                        input_dict[key] = float(val)
                    except ValueError:
                        print(f"Ignored invalid value: {val}")
            else:
                key = smart_feature_match(item, input_dict.keys())
                if key:
                    input_dict[key] = 1

    X = prepare_input_dict(input_dict, features)
    y_pred = model.predict(X)
    prognosis = le.inverse_transform(y_pred)[0]

    print("\n===== Predicted Prognosis =====")
    print(f"{prognosis}")
    print("==============================\n")

    show_prediction_graph(model, le, X)

# --------------------- Run ---------------------
if __name__ == "__main__":
    interactive_mode()
