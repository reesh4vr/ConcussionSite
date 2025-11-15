"""
ConcussionSite - Light-Sensitivity Screening Tool
A webcam-based concussion light-sensitivity screen using MediaPipe FaceMesh
and EAR-based blink detection.

Scientific References:
- Clark et al., 2021: Visually-evoked effects correlate with concussion
- Studies showing blink rate ↑ under visual stress in mTBI
- Eye movement dysfunction after concussion
- EAR blink detection (Soukupová & Čech, 2016)
- MediaPipe FaceMesh reliability
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize MediaPipe FaceMesh
# MediaPipe FaceMesh provides 468 facial landmarks for accurate eye tracking
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe FaceMesh)
# Left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# EAR (Eye Aspect Ratio) threshold for blink detection
# Based on Soukupová & Čech, 2016: EAR < 0.25 indicates eye closure
EAR_THRESHOLD = 0.25

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    EAR = (vertical distances) / (2 * horizontal distance)
    Based on Soukupová & Čech, 2016.
    """
    # Vertical distances
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    # Horizontal distance
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    if h == 0:
        return 0.0
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_eye_center(eye_landmarks):
    """Calculate the center point of an eye."""
    if len(eye_landmarks) == 0:
        return None
    center = np.mean(eye_landmarks, axis=0)
    return center

def calculate_gaze_distance(eye_center, image_center):
    """
    Calculate distance from eye center to image center.
    Higher distance indicates gaze aversion (potential light sensitivity).
    """
    if eye_center is None:
        return 0.0
    distance = np.linalg.norm(eye_center - image_center)
    return distance

def create_flicker_window():
    """Create a flickering grayscale window for light sensitivity testing."""
    window_name = "Light Sensitivity Test - Look at this window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    return window_name

def flicker_display(window_name, frame_count, flicker_rate=10):
    """
    Display flickering grayscale pattern.
    flicker_rate: frames per flicker cycle (lower = faster flicker)
    """
    # Alternate between white and black
    is_white = (frame_count // flicker_rate) % 2 == 0
    color = 255 if is_white else 0
    flicker_img = np.full((600, 800, 3), color, dtype=np.uint8)
    
    # Add instruction text
    text = "Keep looking at this window"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (800 - text_size[0]) // 2
    text_y = 300
    cv2.putText(flicker_img, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128) if is_white else (255, 255, 255), 2)
    
    cv2.imshow(window_name, flicker_img)

def run_phase(cap, flicker_window_name, phase_name, duration_sec, flicker=False):
    """
    Run a testing phase (baseline or flicker).
    Returns metrics: blink_count, eye_closed_time, gaze_distances
    """
    print(f"\n{'='*50}")
    print(f"Starting {phase_name} phase ({duration_sec} seconds)")
    print(f"{'='*50}")
    print("Please look at the camera and keep your eyes open.")
    if flicker:
        print("A flickering window will appear. Try to keep looking at it.")
    
    start_time = time.time()
    blink_count = 0
    eye_closed_time = 0.0
    gaze_distances = []
    frame_count = 0
    last_ear_time = time.time()
    is_eye_closed = False
    
    prev_ear_left = 0.3
    prev_ear_right = 0.3
    
    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        # Get image center for gaze calculation
        h, w = frame.shape[:2]
        image_center = np.array([w / 2, h / 2])
        
        current_time = time.time()
        frame_duration = current_time - last_ear_time
        last_ear_time = current_time
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye landmarks
            left_eye = []
            right_eye = []
            
            for idx in LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                left_eye.append([x, y])
            
            for idx in RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                right_eye.append([x, y])
            
            # Calculate EAR for both eyes
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            avg_ear = (ear_left + ear_right) / 2.0
            
            # Blink detection: EAR drops below threshold
            if avg_ear < EAR_THRESHOLD:
                if not is_eye_closed:
                    # Just closed - count as blink
                    blink_count += 1
                    is_eye_closed = True
                eye_closed_time += frame_duration
            else:
                is_eye_closed = False
            
            # Gaze aversion detection
            left_eye_center = get_eye_center(left_eye)
            right_eye_center = get_eye_center(right_eye)
            
            if left_eye_center is not None and right_eye_center is not None:
                eye_center = (left_eye_center + right_eye_center) / 2.0
                gaze_dist = calculate_gaze_distance(eye_center, image_center)
                gaze_distances.append(gaze_dist)
            
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                None, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # Display EAR and status
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status = "EYES CLOSED" if is_eye_closed else "EYES OPEN"
            cv2.putText(frame, status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_eye_closed else (0, 255, 0), 2)
        
        cv2.imshow("ConcussionSite - Webcam", frame)
        
        # Display flicker window if in flicker phase
        if flicker:
            flicker_display(flicker_window_name, frame_count)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return {
        'blink_count': blink_count,
        'eye_closed_time': eye_closed_time,
        'gaze_distances': gaze_distances,
        'duration': duration_sec
    }

def ask_symptoms():
    """Ask simple yes/no symptom questions."""
    print("\n" + "="*50)
    print("Symptom Questionnaire")
    print("="*50)
    
    symptoms = {
        'headache': "Do you have a headache? (y/n): ",
        'nausea': "Do you feel nauseous? (y/n): ",
        'dizziness': "Do you feel dizzy? (y/n): ",
        'light_sensitivity': "Are you sensitive to light? (y/n): "
    }
    
    answers = {}
    for symptom, question in symptoms.items():
        while True:
            answer = input(question).strip().lower()
            if answer in ['y', 'yes', 'n', 'no']:
                answers[symptom] = answer in ['y', 'yes']
                break
            print("Please enter 'y' or 'n'")
    
    return answers

def calculate_metrics(baseline_metrics, flicker_metrics):
    """Calculate concussion-relevant metrics."""
    baseline_duration = baseline_metrics['duration']
    flicker_duration = flicker_metrics['duration']
    
    # Blink rates (blinks per minute)
    baseline_blink_rate = (baseline_metrics['blink_count'] / baseline_duration) * 60
    flicker_blink_rate = (flicker_metrics['blink_count'] / flicker_duration) * 60
    blink_rate_delta = flicker_blink_rate - baseline_blink_rate
    
    # Eye-closed fraction
    total_time = baseline_metrics['eye_closed_time'] + flicker_metrics['eye_closed_time']
    total_duration = baseline_duration + flicker_duration
    eye_closed_fraction = total_time / total_duration if total_duration > 0 else 0.0
    
    # Gaze-off-center fraction
    # Consider gaze as "off-center" if distance > threshold (e.g., 50 pixels)
    gaze_threshold = 50.0
    baseline_gaze_off = sum(1 for d in baseline_metrics['gaze_distances'] if d > gaze_threshold)
    flicker_gaze_off = sum(1 for d in flicker_metrics['gaze_distances'] if d > gaze_threshold)
    total_gaze_measurements = len(baseline_metrics['gaze_distances']) + len(flicker_metrics['gaze_distances'])
    gaze_off_fraction = (baseline_gaze_off + flicker_gaze_off) / total_gaze_measurements if total_gaze_measurements > 0 else 0.0
    
    return {
        'baseline_blink_rate': baseline_blink_rate,
        'flicker_blink_rate': flicker_blink_rate,
        'blink_rate_delta': blink_rate_delta,
        'eye_closed_fraction': eye_closed_fraction,
        'gaze_off_fraction': gaze_off_fraction
    }

def generate_ai_summary(metrics, symptoms):
    """Generate AI summary using OpenAI."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not found in .env file.")
        print("Skipping AI summary generation.")
        return "AI summary unavailable: API key not configured."
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a medical screening assistant. Analyze the following concussion light-sensitivity screening results and provide a concise summary (<150 words).

METRICS:
- Baseline blink rate: {metrics['baseline_blink_rate']:.2f} blinks/min
- Flicker blink rate: {metrics['flicker_blink_rate']:.2f} blinks/min
- Blink rate increase: {metrics['blink_rate_delta']:.2f} blinks/min
- Eye-closed fraction: {metrics['eye_closed_fraction']:.2%}
- Gaze-off-center fraction: {metrics['gaze_off_fraction']:.2%}

SYMPTOMS:
- Headache: {'Yes' if symptoms['headache'] else 'No'}
- Nausea: {'Yes' if symptoms['nausea'] else 'No'}
- Dizziness: {'Yes' if symptoms['dizziness'] else 'No'}
- Light sensitivity: {'Yes' if symptoms['light_sensitivity'] else 'No'}

Provide a summary that:
1. States whether elevated light sensitivity is indicated
2. Assesses if results are mild vs. potentially concerning
3. Emphasizes this is NOT a diagnosis
4. Suggests consulting a healthcare professional if concerning

Keep the response under 150 words and professional."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical screening assistant providing non-diagnostic health information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"\n⚠️  Error generating AI summary: {e}")
        return "AI summary unavailable due to error."

def main():
    """Main function to run the concussion screening."""
    print("="*50)
    print("ConcussionSite - Light Sensitivity Screening")
    print("="*50)
    print("\nThis tool screens for potential light sensitivity related to concussion.")
    print("It is NOT a diagnostic tool. Please consult a healthcare professional.")
    print("\nPress 'q' during testing to quit early.")
    input("\nPress Enter to start...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create flicker window
    flicker_window = create_flicker_window()
    
    try:
        # Baseline phase (8 seconds)
        baseline_metrics = run_phase(cap, flicker_window, "Baseline", 8, flicker=False)
        
        # Brief pause
        print("\nPreparing flicker test...")
        time.sleep(2)
        
        # Flicker phase (15 seconds)
        flicker_metrics = run_phase(cap, flicker_window, "Flicker", 15, flicker=True)
        
        # Calculate metrics
        metrics = calculate_metrics(baseline_metrics, flicker_metrics)
        
        # Display metrics
        print("\n" + "="*50)
        print("SCREENING RESULTS")
        print("="*50)
        print(f"Baseline blink rate: {metrics['baseline_blink_rate']:.2f} blinks/min")
        print(f"Flicker blink rate: {metrics['flicker_blink_rate']:.2f} blinks/min")
        print(f"Blink rate increase: {metrics['blink_rate_delta']:.2f} blinks/min")
        print(f"Eye-closed fraction: {metrics['eye_closed_fraction']:.2%}")
        print(f"Gaze-off-center fraction: {metrics['gaze_off_fraction']:.2%}")
        
        # Ask symptoms
        symptoms = ask_symptoms()
        
        # Generate AI summary
        print("\n" + "="*50)
        print("Generating AI Summary...")
        print("="*50)
        ai_summary = generate_ai_summary(metrics, symptoms)
        
        print("\n" + "="*50)
        print("AI SUMMARY")
        print("="*50)
        print(ai_summary)
        print("\n" + "="*50)
        print("⚠️  DISCLAIMER: This is NOT a medical diagnosis.")
        print("If you have concerns, please consult a healthcare professional.")
        print("="*50)
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

