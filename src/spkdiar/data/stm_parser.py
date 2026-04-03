"""
stm_parser.py

Parse ATC0R STM (Segment Time Marking) files.

ATC0R STM format (one line per cue):
    <rec_id> <channel> <quality>><speaker>><listener> <start> <end> <transcript>

Example:
    dca_d1_1 1 1>D1-1>DAL209 63.380 66.070 Delta Two Zero Nine, turn left heading two eight zero.

Fields:
    rec_id      : Recording identifier (e.g., "dca_d1_1")
    channel     : Audio channel, always "1" for single-channel ATC
    quality     : Quality level 1-4 (1=clear, 4=unintelligible)
    speaker     : Speaker identifier (e.g., "D1-1" for controller, "DAL209" for pilot)
    listener    : Intended listener (e.g., "DAL209" or "D1-1")
    start       : Segment start time in seconds
    end         : Segment end time in seconds
    transcript  : Transcribed text
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Cue:
    """A single communication cue from an ATC0R STM file."""

    rec_id: str
    channel: int
    quality: int
    speaker: str
    listener: str
    start: float
    end: float
    transcript: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_controller(self) -> bool:
        """Heuristic: controller IDs contain a hyphen like D1-1, F1-1, etc."""
        return "-" in self.speaker and not self.speaker.startswith("N")

    def __repr__(self) -> str:
        return (
            f"Cue({self.start:.3f}-{self.end:.3f} "
            f"spk={self.speaker} -> {self.listener} "
            f"q={self.quality} '{self.transcript[:40]}...')"
        )


@dataclass
class Recording:
    """All cues for a single recording, parsed from one STM file."""

    rec_id: str
    cues: list[Cue] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration based on the latest end time."""
        if not self.cues:
            return 0.0
        return max(c.end for c in self.cues)

    @property
    def speakers(self) -> list[str]:
        """Unique speakers sorted by first appearance."""
        seen = {}
        for cue in self.cues:
            if cue.speaker not in seen:
                seen[cue.speaker] = cue.start
        return sorted(seen, key=lambda s: seen[s])

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    def cues_in_range(self, start: float, end: float) -> list[Cue]:
        """Return cues that overlap with [start, end)."""
        return [c for c in self.cues if c.start < end and c.end > start]

    def speakers_in_range(self, start: float, end: float) -> list[str]:
        """Unique speakers active in [start, end), sorted by first appearance."""
        seen = {}
        for c in self.cues_in_range(start, end):
            if c.speaker not in seen:
                seen[c.speaker] = c.start
        return sorted(seen, key=lambda s: seen[s])


def parse_stm_line(line: str) -> Optional[Cue]:
    """Parse a single ATC0R STM line into a Cue object.

    Returns None if the line is malformed or a comment.
    """
    line = line.strip()
    if not line or line.startswith(";;") or line.startswith("#"):
        return None

    parts = line.split(None, 5)  # Split into at most 6 fields
    if len(parts) < 5:
        return None

    rec_id = parts[0]

    try:
        channel = int(parts[1])
    except ValueError:
        return None

    # Parse the quality>speaker>listener field
    spk_field = parts[2]
    spk_parts = spk_field.split(">")
    if len(spk_parts) != 3:
        # Fallback: treat entire field as speaker
        quality = 0
        speaker = spk_field
        listener = "unknown"
    else:
        try:
            quality = int(spk_parts[0])
        except ValueError:
            quality = 0
        speaker = spk_parts[1]
        listener = spk_parts[2]

    try:
        start = float(parts[3])
        end = float(parts[4])
    except ValueError:
        return None

    transcript = parts[5] if len(parts) > 5 else ""

    return Cue(
        rec_id=rec_id,
        channel=channel,
        quality=quality,
        speaker=speaker,
        listener=listener,
        start=start,
        end=end,
        transcript=transcript,
    )


def parse_stm_file(stm_path: Path | str) -> Recording:
    """Parse an entire ATC0R STM file into a Recording object."""
    stm_path = Path(stm_path)
    if not stm_path.is_file():
        raise FileNotFoundError(f"STM file not found: {stm_path}")

    rec_id = stm_path.stem  # e.g., "dca_d1_1" from "dca_d1_1.stm"
    recording = Recording(rec_id=rec_id)

    with stm_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            cue = parse_stm_line(line)
            if cue is None:
                continue

            # Sanity checks
            if cue.end <= cue.start:
                continue
            if cue.quality == 4:
                # Level 4 = unintelligible, skip for diarization
                continue

            recording.cues.append(cue)

    # Sort by start time
    recording.cues.sort(key=lambda c: c.start)
    return recording


def parse_stm_dir(stm_dir: Path | str) -> list[Recording]:
    """Parse all STM files in a directory."""
    stm_dir = Path(stm_dir)
    if not stm_dir.is_dir():
        raise NotADirectoryError(f"STM directory not found: {stm_dir}")

    recordings = []
    for stm_path in sorted(stm_dir.glob("*.stm")):
        rec = parse_stm_file(stm_path)
        if rec.cues:
            recordings.append(rec)

    return recordings
