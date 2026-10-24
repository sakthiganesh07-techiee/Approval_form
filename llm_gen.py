import json
import re
from datetime import date
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from approval_generator import fill_approval_form, fill_circular


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = BASE_DIR / "approval_values.json"

MIN_WORDS = 25
MAX_WORDS = 75
TARGET_WORDS = 45


# ============================================================
# LOAD AI MODEL
# ============================================================

def load_model():
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading AI model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    model.eval()

    print("AI model loaded successfully!")

    return tokenizer, model


# ============================================================
# TEXT UTILITIES
# ============================================================

def count_words(text):
    return len(text.split())


def clean_text(text):
    text = re.sub(
        r"^(Objective|Expected Outcome|Outcome)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\*{1,3}(.*?)\*{1,3}",
        r"\1",
        text
    )

    text = re.sub(
        r"^\s*#+\s*",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# GENERATE AI TEXT
# ============================================================

def generate_text(tokenizer, model, prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "You generate professional college event approval content. "
                "Return plain text only. "
                "Write exactly one continuous paragraph. "
                "Do not use Markdown, headings, bullet points, "
                "numbered lists, or decorative symbols. "
                "Use clear, formal English. "
                "Use only the supplied event information. "
                "Never invent facts. "
                "Return only the requested paragraph."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return clean_text(text)


# ============================================================
# GENERATE OBJECTIVE AND EXPECTED OUTCOME
# ============================================================

def generate_limited_paragraph(
    tokenizer,
    model,
    section_name,
    event_details,
    paragraph_type
):

    attempt = 1

    while True:

        print("\n" + "=" * 60)
        print(f"GENERATING {section_name.upper()}")
        print(f"Attempt: {attempt}")
        print("=" * 60)

        prompt = f"""
Write a professional {section_name.lower()} paragraph
for an official college event approval form.

EVENT DETAILS:
{event_details}

CONTENT REQUIREMENTS:
{paragraph_type}

STRICT REQUIREMENTS:
- Aim for exactly {TARGET_WORDS} words.
- The paragraph must contain between {MIN_WORDS}
  and {MAX_WORDS} words, inclusive.
- Write one complete paragraph.
- Use professional English.
- Use plain text only.
- Do not use headings or bullet points.
- Do not invent facts.
- Return only the paragraph.
""".strip()

        text = generate_text(
            tokenizer,
            model,
            prompt
        )

        word_count = count_words(text)

        print("\nGenerated paragraph:")
        print(text)

        print(f"\nWord count: {word_count}")

        if MIN_WORDS <= word_count <= MAX_WORDS:

            print(
                f"\n{section_name} accepted. "
                "Word count is within the required range."
            )

            return text

        print("\n" + "!" * 60)
        print("WORD COUNT REQUIREMENT NOT SATISFIED")
        print(f"Required: {MIN_WORDS}-{MAX_WORDS} words")
        print(f"Generated: {word_count} words")
        print("Retrying with the same event information...")
        print("!" * 60)

        attempt += 1


# ============================================================
# USER INPUT
# ============================================================

def ask(label, default=""):

    if default:

        value = input(
            f"{label} [{default}]: "
        ).strip()

        return value or default

    return input(f"{label}: ").strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("COLLEGE EVENT APPROVAL AND CIRCULAR GENERATOR")
    print("=" * 65)

    print("\nEnter the event details below.\n")

    # Basic information

    letter_date = ask(
        "Letter Date (DD/MM/YYYY)",
        date.today().strftime("%d/%m/%Y")
    )

    event_name = ask("Event Name")

    event_type = ask("Event Type")

    subject = f"{event_name} - Permission - Reg."

    event_date = ask("Event Date")

    start_time = ask("Start Time")

    end_time = ask("End Time")

    audience = ask("Target Audience / Participants")

    organizer = ask(
        "Organizing Body",
        "IEEE KIT Student Branch"
    )

    venue = ask("Venue")

    description = ask("Brief Event Description")

    budget = ask(
        "Tentative Budget",
        "To be confirmed"
    )

    # Validate required fields

    required_fields = {
        "Event Name": event_name,
        "Event Type": event_type,
        "Event Date": event_date,
        "Start Time": start_time,
        "End Time": end_time,
        "Target Audience": audience,
        "Organizing Body": organizer,
        "Venue": venue,
        "Event Description": description
    }

    missing_fields = [
        field
        for field, value in required_fields.items()
        if not value.strip()
    ]

    if missing_fields:

        print("\nPlease provide the following required details:")

        for field in missing_fields:
            print(f"- {field}")

        return

    # Prepare event information for the AI

    event_details = f"""
Event Name: {event_name}
Event Type: {event_type}
Subject: {subject}
Event Date: {event_date}
Start Time: {start_time}
End Time: {end_time}
Target Audience: {audience}
Organizer: {organizer}
Venue: {venue}
Event Description: {description}
Tentative Budget: {budget}
""".strip()

    # Load AI model

    tokenizer, model = load_model()

    # Generate Objective

    objective = generate_limited_paragraph(
        tokenizer=tokenizer,
        model=model,
        section_name="Objective",
        event_details=event_details,
        paragraph_type=(
            "Explain the purpose and goals of the event. "
            "Describe the knowledge, skills, or abilities "
            "the event aims to develop."
        )
    )

    # Generate Expected Outcome

    outcome = generate_limited_paragraph(
        tokenizer=tokenizer,
        model=model,
        section_name="Expected Outcome",
        event_details=event_details,
        paragraph_type=(
            "Explain what participants are expected to learn "
            "or gain. Describe the anticipated academic and "
            "professional benefits using future-oriented wording."
        )
    )

    # Generate Circular

    circular = (
        f"This is to inform you all that {organizer} is organizing "
        f"a {event_type} on {event_name} on {event_date}, "
        f"from {start_time} to {end_time} at {venue}. "
        f"All concerned students are requested to participate "
        f"and make the event a success."
    )

    # Prepare placeholders for both Word templates

    template_values = {
        "{{LETTER_DATE}}": letter_date,
        "{{SUBJECT}}": subject,

        "{{EVENT TYPE}}": event_type,
        "{{EVENT_TYPE}}": event_type,

        "{{EVENT_NAME}}": event_name,
        "{{EVENT_DATE}}": event_date,

        "{{START_TIME}}": start_time,
        "{{END_TIME}}": end_time,

        "{{AUDIENCE}}": audience,
        "{{ORGANIZER}}": organizer,
        "{{VENUE}}": venue,

        "{{OBJECTIVE}}": objective,
        "{{OUTCOME}}": outcome,

        "{{BUDGET}}": budget,
        "{{CIRCULAR}}": circular
    }

    # Save JSON data

    json_values = {
        key.strip("{}"): value
        for key, value in template_values.items()
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            json_values,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"\nJSON saved: {OUTPUT_JSON.resolve()}")

    # Generate Approval Form

    print("\n" + "=" * 65)
    print("GENERATING APPROVAL FORM")
    print("=" * 65)

    approval_path = fill_approval_form(
        template_values
    )

    # Generate Circular

    print("\n" + "=" * 65)
    print("GENERATING CIRCULAR")
    print("=" * 65)

    circular_path = fill_circular(
        template_values
    )

    # Display generated paragraphs

    print("\n" + "=" * 65)
    print("GENERATED OBJECTIVE")
    print("=" * 65)

    print(objective)
    print(f"Word count: {count_words(objective)}")

    print("\n" + "=" * 65)
    print("GENERATED EXPECTED OUTCOME")
    print("=" * 65)

    print(outcome)
    print(f"Word count: {count_words(outcome)}")

    print("\n" + "=" * 65)
    print("GENERATED CIRCULAR")
    print("=" * 65)

    print(circular)

    # Final summary

    print("\n" + "=" * 65)
    print("ALL DOCUMENTS GENERATED SUCCESSFULLY")
    print("=" * 65)

    print(f"Approval Form: {approval_path}")
    print(f"Circular: {circular_path}")
    print(f"JSON Data: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()