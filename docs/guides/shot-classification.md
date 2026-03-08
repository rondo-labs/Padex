# Shot Classification

Padex classifies each detected shot using a **three-signal decision tree**: pre-contact ball state, player pose, and post-contact ball trajectory.

## How It Works

### Step 1: Contact Detection

The `ProximityVelocityContactStrategy` identifies the moment a player hits the ball:

1. Compute ball velocity between consecutive visible frames
2. Find frames where velocity changes sharply (delta-V > threshold)
3. Attribute the contact to the nearest player within 2m
4. Suppress duplicate detections within 300ms

### Step 2: Three-Signal Classification

Once a contact is detected, `PoseBasedShotTypeClassifier` classifies the shot type using three signals:

```
                    Pre-contact ball state
                    /         |          \
            No ground     Wall bounce    Ground only
            bounce         before         before
               |              |              |
          NET PLAY       DEFENSIVE      BASELINE
               |              |              |
          + Pose         + Position      + Pose
          + Trajectory   + Trajectory    + Trajectory
               |              |              |
          volley         wall_return     groundstroke_fh
          bandeja        contra_pared    groundstroke_bh
          vibora         lob             bajada
          smash                          chiquita
          drop_shot                      lob
```

#### Signal 1: Pre-contact ball state

What happened to the ball between the previous contact and this one?

| Condition | Branch |
|-----------|--------|
| No ground bounce | **Net play** — player intercepted before it bounced |
| Wall bounce (back/side wall) | **Defensive** — ball came off the wall |
| Ground bounce only | **Baseline** — standard rally ball |

#### Signal 2: Player pose

YOLO-Pose extracts 17 COCO keypoints. Key features:

| Feature | How it's measured | What it indicates |
|---------|-------------------|-------------------|
| Overhead | Wrist Y < Shoulder Y (pixel coords) | Bandeja, vibora, smash vs volley |
| Wrist-shoulder gap | Pixels between wrist and shoulder | Smash (large gap) vs bandeja (moderate) |
| Side spin | Lateral wrist-elbow distance > 40px | Vibora (side-cut motion) |
| Forehand/backhand | Dominant wrist position relative to body center | Groundstroke side |

#### Signal 3: Post-contact trajectory

The ball's path after contact refines classification:

| Feature | Measurement | Indicates |
|---------|-------------|-----------|
| Short trajectory | Total distance < 2m in 10 frames | Drop shot |
| Lob trajectory | Y displacement > 3m in 15 frames | Lob |
| Exit smash | Ball reaches court edge (x < 0.5 or y < 0.5) | Smash por tres (x3) |
| Toward net | Ball ends closer to net than player | Chiquita |

## Shot Type Taxonomy

Padex defines 15 shot types organized by playing situation:

### Serve

| Type | Enum | Description |
|------|------|-------------|
| Serve | `serve` | First shot of each point |

### Net Play (no ground bounce before contact)

| Type | Enum | Description |
|------|------|-------------|
| Volley | `volley` | Low contact, no bounce, standard intercept |
| Bandeja | `bandeja` | Moderate overhead, controlled placement |
| Vibora | `vibora` | Overhead with side-spin cutting motion |
| Smash | `smash` | High overhead, aggressive, large wrist-shoulder gap |
| Smash x3 | `smash_x3` | Smash that exits through the back (por tres) |
| Smash x4 | `smash_x4` | Smash that exits through the side (por cuatro) |
| Drop shot | `drop_shot` | Soft touch, short trajectory < 2m |

### Baseline (ground bounce before contact)

| Type | Enum | Description |
|------|------|-------------|
| Forehand | `groundstroke_fh` | Ground bounce, forehand side |
| Backhand | `groundstroke_bh` | Ground bounce, backhand side |
| Bajada | `bajada` | Overhead shot after ground bounce (typically from back of court) |

### Transition

| Type | Enum | Description |
|------|------|-------------|
| Chiquita | `chiquita` | Low ball aimed at opponents' feet near the net |

### Defensive (wall bounce before contact)

| Type | Enum | Description |
|------|------|-------------|
| Lob | `lob` | High arc shot, typically defensive |
| Wall return | `wall_return` | Return after wall bounce, mid-court |
| Contra pared | `contra_pared` | Return after wall bounce from baseline |

### Fallback

| Type | Enum | Description |
|------|------|-------------|
| Unknown | `unknown` | Insufficient data to classify |

## Confidence Scores

Each shot has a confidence score (0-1) computed as:

```
final_confidence = contact_confidence * classification_confidence
```

- **Contact confidence**: Based on velocity change magnitude (higher delta-V = more confident)
- **Classification confidence**: Based on how clearly the signals match (typically 0.6-0.75)

## Extending the Classifier

The classification system uses a strategy pattern. To implement a custom classifier:

```python
from padex.events.shot import ShotTypeClassifier, ShotType

class MyClassifier(ShotTypeClassifier):
    def classify(self, contact, ball_before, ball_after, bounces_before, keypoints):
        # Your logic here
        return (ShotType.VOLLEY, 0.9)

from padex.events import ShotDetector
detector = ShotDetector(shot_type_classifier=MyClassifier())
```
