# M1GP QUEST

M1GP QUEST is a motion-casting RPG prototype that uses a smartphone accelerometer as an input device. The player holds a phone like a magic staff, starts a casting section, draws a motion shape in the air, and attacks enemies based on the recognized motion.

The current main mode is GAME2. In this mode, the game focuses on three motion commands:

| Motion | Skill | MP | Effect |
|---|---:|---:|---|
| Spiral | Spiral | 0 | 20 damage |
| Star | Star | 3 | 60 damage |
| Z | Z | 3 | Stun the enemy's next attack |

The game is designed as a demonstration prototype for motion recognition, especially for experimenting with motion segment detection and shape classification from continuous acceleration logs.

## Background

This project started as a prototype for collecting acceleration data from a smartphone browser. At first, the goal was simply to confirm that a phone could send DeviceMotion API values to a PC server, save them as CSV logs, visualize them, and detect motion intervals.

After the logging and detection pipeline became stable, the prototype was extended into a small turn-based battle game. The purpose of the game layer is not only entertainment, but also to make motion recognition easier to demonstrate. A game screen gives immediate feedback when a motion is recognized, and it makes it easier to explain why accurate motion interval estimation matters.

The project gradually moved from single-axis commands such as horizontal, vertical, and thrust-like motions to a shape-based input method. The current version recognizes shape motions such as Spiral, Star, and Z. This makes the game closer to a "motion casting" system: the player casts a skill by drawing a shape with the phone.

## Research Connection

The main research theme behind this prototype is activity and motion recognition using smartphone acceleration sensors.

In a real motion-recognition game, the system must solve two problems:

1. Detect where a meaningful action starts and ends in a continuous sensor stream.
2. Classify the extracted motion segment into a command or action label.

This game keeps those two problems visible. The player continuously streams acceleration data from the phone, but the game should only classify the section that corresponds to the intended casting motion. If the start or end of the motion segment is wrong, the classifier receives noisy data and the skill may fail or be misclassified.

For that reason, the prototype separates:

- Trigger detection: detecting the start and end of casting.
- Shape classification: classifying the motion segment as Spiral, Star, Z, or none.

The backend includes rule-based trigger handling, machine-learning models, and DTW template matching. The project is useful for testing how recognition accuracy changes when the motion interval is cut differently, when training data is added, or when the classification method is changed.

## System Overview

```text
m1gp-quest/
  backend/
    server.py
    requirements.txt
    models/
      shape_classifier.pkl
      trigger_detector.pkl
      motion_classifier.pkl
    data/
      training/
      training_trigger/
  frontend/
    src/
    public/
    dist/
  StartDemo.bat
  StartDemo.ps1
  StartDevDemo.bat
  StartDevDemo.ps1
  cloudflared.exe
```

### Frontend

- React + TypeScript
- PC battle screen
- Phone sensor screen
- Training screen
- Tutorial screen
- Debug panel
- Pixel-style UI, sounds, stage backgrounds, battle effects
- Generic enemy images are used so the repository can be shared safely.

### Backend

- Python HTTP server
- Receives batched acceleration samples
- Manages battle motion sessions
- Saves training data
- Runs trigger and shape recognition
- Serves `frontend/dist` in demo mode

### Phone Input

The phone browser collects acceleration data with the DeviceMotion API. During play, the phone can be used in two ways:

- Trigger gesture mode: push the phone forward and return it to start or confirm casting.
- Tap wand mode: tap the wand on the phone screen to start or confirm casting.

Tap wand mode is useful when the focus is shape recognition rather than trigger recognition.

## Main Game Flow

1. Start the PC battle screen.
2. Open the phone URL on a smartphone.
3. Allow sensor access.
4. Start streaming from the phone.
5. Start the battle.
6. Start casting with a trigger gesture or the wand tap mode.
7. Draw Spiral, Star, or Z with the phone.
8. Confirm casting.
9. The recognized skill is activated.
10. The enemy acts.
11. Repeat until all enemies are defeated or the player HP reaches zero.

## GAME2 Rules

GAME2 is the current main rule set.

- Player max HP: 220
- Player max MP: 5
- MP recovers by 1 each turn.
- MP does not fully recover when moving to the next stage.
- enemy1, enemy2, and enemy3 have 60 HP.
- enemy4 has 150 HP.
- Spiral costs 0 MP and deals 20 damage.
- Star costs 3 MP and deals 60 damage.
- Z costs 3 MP and stuns the enemy's next attack.

Enemy actions are selected randomly from predefined action choices. enemy1, enemy2, and enemy3 choose from 30 damage, 50 damage, or idle. enemy4 chooses from 50 damage, 150 damage, or idle, and the 150 damage attack is not selected twice in a row.

## Recognition

The current shape recognition system uses DTW template matching first and keeps the RandomForest classifier as a fallback.

The DTW flow is:

1. Extract the input motion section.
2. Trim a small margin at the start and end.
3. Resample the time series to a fixed length.
4. Normalize each axis.
5. Compare the motion with saved training templates.
6. Choose the closest label by distance.
7. Use confidence and distance thresholds to reject uncertain input.

The target labels are:

- `circle` / Spiral
- `star` / Star
- `zigzag` / Z
- `none`

## Training Data

Training data is stored under:

```text
backend/data/training/
backend/data/training_trigger/
```

Shape data and trigger data are intentionally separated. Shape training data should contain only the drawing motion. Trigger data should contain only the start or end trigger gesture, plus negative examples.

This separation is important because mixing the trigger gesture into shape data makes the shape classifier less stable.

## Development Start

Use this when developing with Vite.

```powershell
cd C:\Users\yusei\dev\m1gp-quest
.\StartDevDemo.bat
```

This starts:

- backend on `http://127.0.0.1:8000`
- frontend on `http://127.0.0.1:5173`
- cloudflared tunnel for the phone URL

The launcher also copies the phone URL to the clipboard.

## Portable Demo Start

Use this when running the packaged demo.

```powershell
.\StartDemo.bat
```

This starts the backend, opens the PC battle screen, starts cloudflared, and displays a QR code for the phone URL.

The PC screen runs at:

```text
http://127.0.0.1:8000/
```

The phone should open the cloudflared HTTPS URL with:

```text
?mode=phone
```

## Manual Setup

Install Python dependencies:

```powershell
cd backend
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Build the frontend:

```powershell
cd frontend
npm run build
```

Run the backend:

```powershell
cd backend
python server.py
```

## Notes for Demonstration

- Use GAME2 as the main demo mode.
- Use Tap Wand mode if trigger recognition is unstable.
- Use the Debug panel for forced skills, audio toggles, effect toggles, and recognition diagnostics.
- Use the Training screen to collect additional shape data.
- Keep the phone and PC on a stable network connection.
- For iPhone sensor access, use an HTTPS phone URL.

## Purpose

This project is a practical prototype for showing how continuous acceleration data can be turned into game input. It connects sensor logging, motion segmentation, classification, training data collection, and game feedback in one system.

The game makes the research problem easier to understand: when the motion interval is estimated correctly, the player can cast the intended skill; when the segment is noisy or incomplete, recognition becomes unstable. This relationship between motion segmentation and gameplay is the core idea of the prototype.
