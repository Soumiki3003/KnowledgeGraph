import time
import threading
import supervisor_agent as supervisor  # Import your supervisor_agent.py
from datetime import datetime

IDLE_THRESHOLD = 300  # seconds of no input (true student idle time)
student_id = "student_001"

def idle_monitor(stop_event):
    time.sleep(IDLE_THRESHOLD)
    if not stop_event.is_set():
        print("\nYou’ve been quiet for a while… everything okay?")
        print("You can ask me for help, a hint, or clarification anytime!\n")

def start_session():
    print("\nCompanion Agent Active")
    print("Ask me anything about your static analysis or CTF tasks!")
    print("(Type 'exit' to quit)\n")

    while True:
        # Create an event object to stop the idle timer once user types something
        stop_event = threading.Event()
        idle_thread = threading.Thread(target=idle_monitor, args=(stop_event,))
        idle_thread.daemon = True
        idle_thread.start()

        # Student input
        query = input("You: ").strip()
        stop_event.set()  # Student typed something, cancel idle nudge

        if query.lower() in {"exit", "quit"}:
            print("\n👋 Goodbye! Keep exploring and practicing!")
            break
        if not query:
            continue

        #Forward query to Supervisor Agent
        print("\nCompanion: Let me check with my Supervisor...\n")
        start_time = time.time()

        result = supervisor.retrieve_context(student_id, query)

        end_time = time.time()
        total_time = round(end_time - start_time, 2)

        # Clean up Supervisor’s output
        clean_answer = (
            result.answer.strip()
            if hasattr(result, "answer") and isinstance(result.answer, str)
            else str(getattr(result, "answer", result))
        )

        # Student-facing simplified response
        print("Companion: Here's what I found for you:")
        print(f"🧩 {clean_answer}")
        # print(f"Response generated in {total_time}s")
        # Check and display hint from Supervisor Agent
        try:
            state = supervisor.load_student_state()
            student_entry = next(
                (s for s in state["students"] if s["id"] == student_id), None
            )
            if student_entry and student_entry.get("trajectory"):
                last_entry = student_entry["trajectory"][-1]
                if last_entry.get("hint_triggered") and last_entry.get("hint_text"):
                    print("\nCompanion: Here's an extra hint for you!")
                    print(f"{last_entry['hint_text']}\n")
        except Exception as e:
            print(f"(Hint retrieval error: {e})")

        print("\n───────────────────────────────────────────────\n")

if __name__ == "__main__":
    start_session()
