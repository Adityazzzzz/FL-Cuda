import os
import pickle
import argparse
import torch
from google import genai
from google.genai import types
from model import EEGNet

def load_eeg_sample(data_path, subject_id=1, trial_idx=0):
    """Loads a single trial from the DEAP dataset and extracts 32 EEG channels."""
    file_path = os.path.join(data_path, f"s{subject_id}.dat")
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    # Extract 32 EEG channels: shape (1, 1, 32, 8064)
    raw_trial = data["data"][trial_idx, :32, :]
    tensor_trial = torch.from_numpy(raw_trial).float().unsqueeze(0).unsqueeze(0)
    
    # Ground truth valence rating (Index 0)
    valence = data["labels"][trial_idx, 0]
    true_label = "Positive / High Valence" if valence > 5.0 else "Negative / Low Valence"
    
    return tensor_trial, true_label

def decode_emotion(model, sample_tensor, device):
    """Runs EEGNet inference on the target EEG sample."""
    model.eval()
    with torch.no_grad():
        output = model(sample_tensor.to(device))
        pred_idx = output.argmax(dim=1).item()
    return "Positive / High Valence" if pred_idx == 1 else "Negative / Low Valence"

def query_chatbot(user_message, detected_mood, api_key=None):
    """Generates baseline vs. EEG-conditioned responses using Gemini."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key) if key else genai.Client()
    
    # 1. Baseline Request
    baseline_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful, direct conversational assistant."
        )
    )
    
    # 2. Emotion-Conditioned Request
    mood_instruction = (
        f"You are an empathetic conversational assistant connected to a privacy-preserving Brain-Computer Interface. "
        f"The user's current decoded emotional state is: {detected_mood}. "
        f"Subtly adapt your tone, pacing, and phrasing to support their current emotional state without sounding robotic or explicitly stating their mood unless asked."
    )
    
    adapted_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=mood_instruction
        )
    )
    
    return baseline_response.text, adapted_response.text

def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving EEG Emotion Chatbot Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained EEGNet checkpoint (.pth)")
    parser.add_argument("--data_path", type=str, default="./data/DEAP", help="Path to DEAP directory")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID (1-32)")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index (0-39)")
    parser.add_argument("--prompt", type=str, default="I have a mountain of work to do and I don't even know where to start.", help="User query text")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API Key (optional, can also use GEMINI_API_KEY env var)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct EEGNet architecture matching DEAP settings
    eegnet = EEGNet(
        sample_rate=250,
        channels=32,
        F1=8,
        D=2,
        F2=16,
        time=8064,
        class_num=2,
        bn_track=False
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    eegnet.load_state_dict(checkpoint)

    sample_tensor, true_label = load_eeg_sample(args.data_path, args.subject_id, args.trial_idx)
    detected_mood = decode_emotion(eegnet, sample_tensor, device)

    print("=" * 65)
    print(f"Subject: s{args.subject_id} | Trial: {args.trial_idx}")
    print(f"Ground Truth Label: {true_label}")
    print(f"Model Decoded Mood: {detected_mood}")
    print("=" * 65)
    print(f"\nUser Input: \"{args.prompt}\"\n")

    baseline, adapted = query_chatbot(args.prompt, detected_mood, api_key=args.api_key)

    print("-" * 25 + " BASELINE RESPONSE " + "-" * 25)
    print(baseline.strip())
    print("\n" + "-" * 22 + " EMOTION-ADAPTED RESPONSE " + "-" * 22)
    print(adapted.strip())
    print("=" * 65)

if __name__ == "__main__":
    main()