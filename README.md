# ConcussionSite

A light-sensitivity and eye-response screening tool for potential concussion symptoms using only a laptop webcam.

## What is ConcussionSite?

ConcussionSite is a non-invasive screening tool that uses computer vision and AI to assess potential light sensitivity and eye-response patterns that may be associated with concussion. The tool tracks eye movements, blink rates, and gaze patterns during baseline and light-flicker conditions to identify potential indicators of visual stress.

**⚠️ IMPORTANT: This tool is NOT a diagnostic device. It is a screening tool only and should not replace professional medical evaluation.**

## How It Works

ConcussionSite follows a structured screening protocol:

1. **Baseline Phase (8 seconds)**: Records normal eye behavior while looking at the webcam
2. **Flicker Phase (15 seconds)**: Records eye behavior while exposed to a flickering light pattern
3. **Metrics Calculation**: Computes key indicators:
   - Baseline blink rate (blinks per minute)
   - Flicker blink rate (blinks per minute)
   - Blink rate delta (increase during flicker)
   - Eye-closed fraction (percentage of time eyes were closed)
   - Gaze-off-center fraction (percentage of time gaze was averted)
4. **Symptom Questionnaire**: Asks about headache, nausea, dizziness, and light sensitivity
5. **AI Summary**: Generates a concise assessment using GPT-4o-mini, explaining whether elevated light sensitivity is indicated and whether results are mild or potentially concerning

## Scientific Backing

ConcussionSite is based on established research in concussion assessment and eye-tracking:

### Visually-Evoked Effects After Concussion
- **Clark et al., 2021**: Research demonstrating that visually-evoked effects correlate with concussion symptoms. Visual stress testing can reveal subtle neurological changes following mild traumatic brain injury (mTBI).

### Blink Rate and Visual Stress
- Multiple studies have shown that blink rate increases under visual stress in individuals with mTBI. The flicker phase of ConcussionSite specifically tests this response, as increased blinking during visual stimulation may indicate light sensitivity.

### Oculomotor Dysfunction in mTBI
- Research has consistently documented eye movement dysfunction after concussion, including:
  - Abnormal gaze patterns
  - Difficulty maintaining visual fixation
  - Increased sensitivity to visual stimuli
  - These findings support the use of gaze-aversion detection as a screening metric.

### EAR Blink Detection
- **Soukupová & Čech, 2016**: "Real-Time Eye Blink Detection using Facial Landmarks" - The Eye Aspect Ratio (EAR) method provides reliable, real-time blink detection using facial landmark tracking. ConcussionSite implements EAR with a threshold of 0.25 to detect eye closures.

### MediaPipe FaceMesh
- Google's MediaPipe FaceMesh provides 468 facial landmarks with high accuracy and real-time performance. The system has been validated for facial landmark detection and is suitable for eye-tracking applications.

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```
   (Note: The tool will still run without the API key, but AI summary generation will be skipped)

## Usage

Run the screening tool:
```bash
python3 concussion_site.py
```

Follow the on-screen instructions:
- Look at the webcam during the baseline phase
- Try to keep looking at the flicker window during the flicker phase
- Answer the symptom questions
- Review the AI-generated summary

Press 'q' at any time to quit early.

## Technical Details

- **Webcam Capture**: Uses OpenCV for real-time video capture
- **Face Tracking**: MediaPipe FaceMesh with 468 landmarks
- **Blink Detection**: EAR (Eye Aspect Ratio) algorithm
- **Gaze Tracking**: Distance-based calculation from image center
- **AI Analysis**: OpenAI GPT-4o-mini for summary generation

## Future Enhancements

The `stimulus/` directory is reserved for future integration with TouchDesigner visual stimuli, which will replace the current grayscale flicker pattern with more sophisticated visual tests.

## Disclaimer

**This tool is for screening purposes only and does not provide medical diagnosis.** 

- ConcussionSite cannot replace professional medical evaluation
- Results should be interpreted by qualified healthcare professionals
- If you have concerns about a potential concussion, seek immediate medical attention
- This tool is not FDA-approved or certified for medical diagnosis

## References

- Clark, J. F., et al. (2021). Visually-evoked effects in concussion assessment.
- Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. *21st Computer Vision Winter Workshop*.
- MediaPipe FaceMesh: https://google.github.io/mediapipe/solutions/face_mesh.html
- Research on blink rate and visual stress in mTBI
- Studies on oculomotor dysfunction following concussion

## License

This project is provided as-is for educational and screening purposes.

