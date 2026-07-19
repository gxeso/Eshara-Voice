import queue
import subprocess
import threading


class SpeechController:

    def __init__(self):
        self.phrases = {
            "help": "Help",
            "doctor": "Doctor",
            "medicine": "Medicine",
            "pharmacy": "Pharmacy",
            "emergency": "Emergency",
            "pain": "Pain",
            "hospital_ward": "Hospital ward",
            "swelling":"Swelling",
            "blood_pressure": "Blood Pressure",
            "bleeding":"Bleeding"
        }

        self.speech_queue = queue.Queue()

        self.worker_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )

        self.worker_thread.start()

    def speak_sign(self, sign_name):
        phrase = self.phrases.get(
            sign_name,
            sign_name
        )

        print(f"Speech queued: {phrase}")

        self.speech_queue.put(phrase)

    def _speech_worker(self):
        while True:
            phrase = self.speech_queue.get()

            if phrase is None:
                self.speech_queue.task_done()
                break

            try:
                self._speak_with_windows(phrase)
                print(f"Spoken: {phrase}")

            except Exception as error:
                print(f"Speech error: {error}")

            finally:
                self.speech_queue.task_done()

    def _speak_with_windows(self, phrase):
        # Escape apostrophes for PowerShell
        safe_phrase = phrase.replace("'", "''")

        powershell_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object "
            "System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Rate = 0; "
            f"$speaker.Speak('{safe_phrase}'); "
            "$speaker.Dispose();"
        )

        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                powershell_command
            ],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    def close(self):
        self.speech_queue.put(None)
        self.worker_thread.join(timeout=5)