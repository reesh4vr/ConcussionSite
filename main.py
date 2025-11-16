import cv2
import time
import numpy as np
import mediapipe as mp

from stimulus.flicker import create_flicker_window, flicker_display
from stimulus.pursuit import run_dot_pursuit

from tracking.facemesh import (
    face_mesh, 
    LEFT_EYE_EAR_INDICES, 
    RIGHT_EYE_EAR_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)
from tracking.blink import calculate_ear, get_adaptive_threshold, EAR_CONSECUTIVE_FRAMES
from tracking.gaze import get_eye_center, calculate_gaze_distance

from analysis.metrics import calculate_metrics
from analysis.risk import assess_concussion_risk

from agents.runner import start_conversation

mp_drawing = mp.solutions.drawing_utils


def ask_symptoms():
    """Simple yes/no symptom questions."""
    print("\n" + "=" * 50)
    print("Symptom Questionnaire")
    print("=" * 50)

    symptoms = {
        "headache": "Do you have a headache? (y/n): ",
        "nausea": "Do you feel nauseous? (y/n): ",
        "dizziness": "Do you feel dizzy? (y/n): ",
        "light_sensitivity": "Are you sensitive to light? (y/n): "
    }

    answers = {}
    for key, question in symptoms.items():
        while True:
            ans = input(question).strip().lower()
            if ans in ["y", "yes", "n", "no"]:
                answers[key] = ans in ["y", "yes"]
                break
            print("Please enter 'y' or 'n'.")
    return answers


def run_phase(cap, flicker_window_name, phase_name, duration_sec, flicker=False):
    """
    Run a test phase:
      - Baseline (no flicker)
      - Flicker (with window)
    Track:
      - blink_count
      - eye_closed_time
      - gaze_distances
      - face_detection_rate
      - avg EAR

    Returns a metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Starting {phase_name} phase ({duration_sec} seconds)")
    print(f"{'='*60}")
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
    ear_below_threshold_count = 0
    ear_values = []
    face_detected_count = 0

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        h, w = frame.shape[:2]
        image_center = np.array([w / 2, h / 2])

        current_time = time.time()
        frame_duration = current_time - last_ear_time
        last_ear_time = current_time

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_detected_count += 1

            # EAR landmarks (6 per eye)
            left_eye_ear = []
            right_eye_ear = []
            for idx in LEFT_EYE_EAR_INDICES:
                lm = face_landmarks.landmark[idx]
                left_eye_ear.append([lm.x * w, lm.y * h])
            for idx in RIGHT_EYE_EAR_INDICES:
                lm = face_landmarks.landmark[idx]
                right_eye_ear.append([lm.x * w, lm.y * h])

            # Full eye for gaze
            left_eye_full = []
            right_eye_full = []
            for idx in LEFT_EYE_INDICES:
                lm = face_landmarks.landmark[idx]
                left_eye_full.append([lm.x * w, lm.y * h])
            for idx in RIGHT_EYE_INDICES:
                lm = face_landmarks.landmark[idx]
                right_eye_full.append([lm.x * w, lm.y * h])

            # EAR calc
            ear_left = calculate_ear(left_eye_ear)
            ear_right = calculate_ear(right_eye_ear)
            avg_ear = (ear_left + ear_right) / 2.0
            ear_values.append(avg_ear)

            # Adaptive threshold
            current_threshold = get_adaptive_threshold(ear_values)

            # Blink detection
            if avg_ear < current_threshold:
                ear_below_threshold_count += 1
                if not is_eye_closed and ear_below_threshold_count >= EAR_CONSECUTIVE_FRAMES:
                    blink_count += 1
                    is_eye_closed = True
                if is_eye_closed:
                    eye_closed_time += frame_duration
            else:
                ear_below_threshold_count = 0
                if is_eye_closed:
                    is_eye_closed = False

            # Gaze aversion
            left_center = get_eye_center(left_eye_full)
            right_center = get_eye_center(right_eye_full)
            if left_center is not None and right_center is not None:
                eye_center = (left_center + right_center) / 2.0
                gaze_dist = calculate_gaze_distance(eye_center, image_center)
                gaze_distances.append(gaze_dist)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                None,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # HUD
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.3f} (th: {current_threshold:.3f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            status = "EYES CLOSED" if is_eye_closed else "EYES OPEN"
            cv2.putText(
                frame,
                status,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if is_eye_closed else (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Face: DETECTED",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "FACE NOT DETECTED - Look at camera",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("ConcussionSite - Webcam", frame)

        if flicker:
            flicker_display(flicker_window_name, frame_count)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Debug and validation
    face_detection_rate = (face_detected_count / frame_count * 100) if frame_count > 0 else 0.0
    avg_ear_value = float(np.mean(ear_values)) if ear_values else 0.0
    min_ear_value = float(np.min(ear_values)) if ear_values else 0.0
    max_ear_value = float(np.max(ear_values)) if ear_values else 0.0
    
    # Validation: Check if eye tracking is actually working
    tracking_valid = True
    validation_warnings = []
    
    if face_detection_rate < 50:
        tracking_valid = False
        validation_warnings.append(f"⚠️  LOW face detection rate ({face_detection_rate:.1f}%) - results may be unreliable")
    
    if len(ear_values) == 0:
        tracking_valid = False
        validation_warnings.append("⚠️  NO Eye Aspect Ratio (EAR) values recorded - eye tracking not working")
    elif avg_ear_value < 0.1 or avg_ear_value > 0.5:
        validation_warnings.append(f"⚠️  Unusual Eye Aspect Ratio (EAR) range ({min_ear_value:.3f}-{max_ear_value:.3f}) - check lighting/positioning")
    
    if len(gaze_distances) == 0:
        validation_warnings.append("⚠️  NO gaze measurements - face may not be detected properly")
    elif len(gaze_distances) < frame_count * 0.3:
        validation_warnings.append(f"⚠️  Only {len(gaze_distances)}/{frame_count} gaze measurements - tracking may be incomplete")

    print(f"\n[DEBUG {phase_name}]")
    print(f"  Frames: {frame_count}")
    print(f"  Face detected: {face_detected_count} ({face_detection_rate:.1f}%)")
    print(f"  Eye Aspect Ratio (EAR) avg: {avg_ear_value:.3f}, min: {min_ear_value:.3f}, max: {max_ear_value:.3f}")
    print(f"  Blinks: {blink_count}")
    print(f"  Gaze measurements: {len(gaze_distances)}")
    
    if validation_warnings:
        print(f"\n[VALIDATION WARNINGS]")
        for warning in validation_warnings:
            print(f"  {warning}")
    
    if not tracking_valid:
        print(f"\n❌ WARNING: Eye tracking validation failed for {phase_name} phase!")
        print("   Results may be inaccurate. Please ensure:")
        print("   - Good lighting on your face")
        print("   - Looking directly at the camera")
        print("   - Face is clearly visible (no obstructions)")

    return {
        "blink_count": blink_count,
        "eye_closed_time": eye_closed_time,
        "gaze_distances": gaze_distances,
        "duration": duration_sec,
        "face_detection_rate": face_detection_rate,
        "avg_ear": avg_ear_value,
        "tracking_valid": tracking_valid,
        "validation_warnings": validation_warnings
    }


def main():
    print("=" * 60)
    print("ConcussionSite - Light Sensitivity & Eye Movement Screening")
    print("=" * 60)
    print("\nThis tool screens for patterns that COULD be consistent with")
    print("concussion-related light sensitivity or oculomotor issues.")
    print("It is NOT a diagnostic tool.")
    print("\nPress 'q' during any visual phase to quit early.")
    input("\nPress Enter to start...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    flicker_window = create_flicker_window()

    try:
        # Baseline phase
        baseline_metrics = run_phase(cap, flicker_window, "Baseline", duration_sec=8, flicker=False)

        print("\nPreparing flicker test...")
        time.sleep(2)

        # Flicker phase
        flicker_metrics = run_phase(cap, flicker_window, "Flicker", duration_sec=15, flicker=True)
        
        # Close flicker window after test ends
        cv2.destroyWindow(flicker_window)

        # Smooth pursuit phase
        print("\nPreparing smooth pursuit test...")
        time.sleep(2)
        pursuit_metrics = run_dot_pursuit(duration_sec=12, cap=cap)

        # Combined metrics
        metrics = calculate_metrics(baseline_metrics, flicker_metrics)

        print("\n" + "=" * 50)
        print("SCREENING RESULTS")
        print("=" * 50)
        print(f"Baseline blink rate      : {metrics['baseline_blink_rate']:.2f} blinks/min")
        print(f"Flicker blink rate       : {metrics['flicker_blink_rate']:.2f} blinks/min")
        print(f"Blink rate increase      : {metrics['blink_rate_delta']:.2f} blinks/min")
        print(f"Eye-closed fraction      : {metrics['eye_closed_fraction']:.2%}")
        print(f"Gaze-off-center fraction : {metrics['gaze_off_fraction']:.2%}")
        print(f"Smooth pursuit mean error: {pursuit_metrics['mean_error']:.1f} px")
        print(f"Smooth pursuit variance  : {pursuit_metrics['variance']:.1f}")
        print(f"SP% (in tracking window) : {pursuit_metrics['sp_percent']*100:.1f}%")

        # Subjective feeling score (1-10)
        print("\n" + "=" * 50)
        print("SUBJECTIVE ASSESSMENT")
        print("=" * 50)
        while True:
            try:
                subjective_input = input("Before we continue, how are you feeling right now on a scale of 1-10? (1=worst, 10=best): ").strip()
                subjective_score = int(subjective_input)
                if 1 <= subjective_score <= 10:
                    break
                else:
                    print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number between 1 and 10.")
        
        print(f"Subjective feeling score: {subjective_score}/10\n")

        # Symptoms
        symptoms = ask_symptoms()

        # Risk assessment (includes subjective_score)
        print("\n" + "=" * 50)
        print("CONCUSSION-LIKE RISK ASSESSMENT")
        print("=" * 50)
        risk_assessment = assess_concussion_risk(metrics, symptoms, pursuit_metrics, subjective_score)

        print(f"\n🔍 RISK LEVEL: {risk_assessment['risk_level']}")
        print(f"📊 Risk Score: {risk_assessment['risk_score']}/10")

        if risk_assessment["risk_factors"]:
            print("\n⚠️  Risk Factors Identified:")
            for i, factor in enumerate(risk_assessment["risk_factors"], 1):
                print(f"   {i}. {factor}")
        else:
            print("\n✓ No strong risk factors detected by this screening.")

        print(f"\n💡 Recommendation:")
        print(f"   {risk_assessment['recommendation']}")

        # Interactive conversational agent
        print("\n" + "=" * 50)
        print("AI ASSISTANT")
        print("=" * 50)
        
        # Start interactive conversation with ADK agent system
        from agents.runner import start_conversation
        start_conversation(
            metrics=metrics,
            pursuit_metrics=pursuit_metrics,
            symptoms=symptoms,
            subjective_score=subjective_score,
            risk_assessment=risk_assessment
        )

        print("\n" + "=" * 50)
        print("⚠️  IMPORTANT DISCLAIMER")
        print("=" * 50)
        print("This tool only screens for patterns that MIGHT be consistent")
        print("with concussion-related issues. It does NOT provide a diagnosis.")
        print("If you have symptoms or concerns, especially after a head injury,")
        print("please seek evaluation from a qualified healthcare professional.")
        print("=" * 50)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
