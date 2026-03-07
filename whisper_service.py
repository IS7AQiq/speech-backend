import os
import json
import logging
from typing import Dict, List, Any
import whisper
import warnings

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Speech recognizer powered by OpenAI Whisper."""

    def __init__(self, model_size: str = "base"):
        logger.info(f"📦 Loading Whisper model '{model_size}'…")
        
        # Suppress FP16 warnings on CPU
        warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        
        try:
            self.model = whisper.load_model(model_size)
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise e

    def transcribe_file(self, audio_path: str, return_words: bool = False) -> Dict[str, Any]:
        """Transcribe an audio file and return text + optional stuttering analysis."""
        logger.info(f"🎤 Transcribing: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Whisper handles its own conversion under the hood (via ffmpeg),
        # so we don't need to force conversion with pydub here.
        
        # Transcribe with word timestamps enabled and force Arabic language
        result = self.model.transcribe(audio_path, word_timestamps=True, language="ar")
        
        text = result.get("text", "").strip()
        
        response = {
            "text": text,
        }

        # If frontend wants words (e.g. for stutter analyzer map in flutter), format them
        if return_words:
            words_data = []
            if "segments" in result:
                for segment in result["segments"]:
                    if "words" in segment:
                        for w in segment["words"]:
                            words_data.append({
                                "word": w["word"].strip(),
                                "start": w["start"],
                                "end": w["end"]
                            })

            stuttering_analysis = self._analyze_stuttering(words_data)
            
            response["words"] = words_data
            response["stuttering_analysis"] = stuttering_analysis
            response["word_count"] = len(words_data)
        
        return response

    def _has_repeated_chars(self, word: str, threshold: int = 3) -> tuple[bool, str]:
        """Check if a word contains any character repeated 'threshold' times consecutively."""
        if len(word) < threshold:
            return False, ""
        
        count = 1
        for i in range(1, len(word)):
            if word[i] == word[i-1] and word[i].isalnum():
                count += 1
                if count >= threshold:
                    return True, word[i]
            else:
                count = 1
        return False, ""

    def _analyze_stuttering(self, words: List[Dict]) -> Dict[str, Any]:
        """
        Enhanced stuttering analysis for Arabic speech.
        Detects: Prolongations, Fillers/Struggles, Post-pause words, and Single-letter isolated words.
        """
        problem_words: List[Dict] = []
        stuttering_events: List[Dict] = []
        
        # Common Arabic stuttering fillers and struggle sounds
        fillers = ["آه", "إه", "أمم", "همم", "اه", "ااه", "ايي", "اوو"]
        
        for i, wd in enumerate(words):
            word: str = wd.get("word", "").strip()
            # Remove punctuation for analysis but keep hyphens for struggle patterns
            clean_word = "".join(c for c in word if c.isalnum() or c == "-")
            
            start: float = wd.get("start", 0.0)
            end: float = wd.get("end", 0.0)
            duration: float = end - start
            
            reasons = []
            
            # 1. Repetition (Consecutive identical words)
            if i > 0 and words[i - 1].get("word", "").strip().lower() == word.lower():
                reasons.append({
                    "type": "repetition", 
                    "reason": "Consecutive identical word"
                })

            # 2. Prolongations (Repeated characters e.g. ممممماء or long duration)
            has_repeats, char = self._has_repeated_chars(clean_word, 3)
            if has_repeats:
                reasons.append({
                    "type": "prolongation", 
                    "reason": f"Repeated {char} character"
                })
            elif duration > 1.2:
                reasons.append({
                    "type": "prolongation", 
                    "reason": f"Word duration {round(duration, 2)}s > 1.2s"
                })

            # 3. Fillers and Struggle Patterns (e.g. أ-أ-أنا, آه, أمم)
            if clean_word in fillers:
                reasons.append({
                    "type": "filler", 
                    "reason": "Struggle sound (filler)"
                })
            elif "-" in clean_word or any(clean_word.startswith(c + "-" + c) for c in clean_word if len(clean_word) > 2):
                 reasons.append({
                     "type": "filler", 
                     "reason": "Interrupted word pattern"
                 })

            # 4. Post-Pause Words (Hesitation > 0.5s)
            gap = 0.0
            if i > 0:
                gap = start - words[i - 1].get("end", 0.0)
                if gap > 0.5:
                    reasons.append({
                        "type": "post-pause", 
                        "reason": f"Word after {round(gap, 2)}s hesitation"
                    })
                    
                    # Add standalone hesitation event if it's significant
                    stuttering_events.append({
                        "type": "hesitation",
                        "duration": round(gap, 3),
                        "position": i,
                        "time": start,
                    })

            # 5. Single-Letter Isolated Words (e.g. ع, ب, و spoken in isolation)
            if len(clean_word) == 1 and (gap > 0.4 or (i < len(words) - 1 and words[i+1].get("start", 0.0) - end > 0.4)):
                 reasons.append({
                     "type": "short-isolated", 
                     "reason": "Isolated single-letter word"
                 })

            # If any reasons found, add to problem_words and events
            if reasons:
                # Use the first reason as primary for the main problem_words list
                primary = reasons[0]
                
                problem_words.append({
                    "word": word,
                    "type": primary["type"],
                    "reason": primary["reason"],
                    "position": i
                })
                
                # Update event logs for more detailed visibility
                stuttering_events.append({
                    "type": primary["type"],
                    "word": word,
                    "reason": primary["reason"],
                    "position": i,
                    "time": start,
                    "duration": round(duration, 3)
                })

        return {
            "problem_words": problem_words,
            "stuttering_events": stuttering_events,
            "total_events": len(stuttering_events),
        }
