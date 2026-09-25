import os
from dotenv import load_dotenv

from typesafe_sdk import TypeSafeClient, Choice, Noul, Score


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("TYPESAFE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY is missing. "
        "Create a .env file and add your Jev API key."
    )


# ---------------------------------------------------------
# Example MAM content
# ---------------------------------------------------------

article = {
    "source": "News",
    "publisher": "Example News",
    "title": "Pakistan announces new artificial intelligence policy",
    "timestamp": "2026-09-24T12:00:00",
    "language": "English",

    "content": """
    The government has announced a new national policy
    focused on artificial intelligence development,
    digital transformation, AI education and regulation.
    The policy is expected to affect technology companies,
    universities and public-sector organizations.
    """
}


# ---------------------------------------------------------
# Ask Jev to analyze the content
# ---------------------------------------------------------

def analyze_content(article):

    with TypeSafeClient() as client:

        response = client.system_one(

            state=article,

            questions={

                # -----------------------------------------
                # Question 1: Topic classification
                # -----------------------------------------

                "topic": Choice(
                    instructions=(
                        "What is the primary topic of the content?"
                    ),

                    criteria={
                        "politics": (
                            "Government, political institutions, "
                            "politicians or political policy."
                        ),

                        "technology": (
                            "Technology, artificial intelligence, "
                            "software, computing or digital innovation."
                        ),

                        "business": (
                            "Companies, markets, finance, "
                            "commerce or business activity."
                        ),

                        "security": (
                            "Cybersecurity, national security, "
                            "crime, terrorism or physical security."
                        ),

                        "international": (
                            "International relations, foreign countries "
                            "or cross-border affairs."
                        ),

                        "other": (
                            "The content does not clearly belong "
                            "to the other categories."
                        ),
                    },
                ),

                # -----------------------------------------
                # Question 2: Is it important?
                # -----------------------------------------

                "is_important": Noul(
                    instructions=(
                        "Does this content contain information "
                        "that would be important for a media "
                        "monitoring analyst to review?"
                    )
                ),

                # -----------------------------------------
                # Question 3: Priority
                # -----------------------------------------

                "priority": Score(
                    instructions=(
                        "How important is this content for an "
                        "analyst monitoring public media?"
                    ),

                    criteria=[
                        "Low importance",
                        "Moderate importance",
                        "High importance",
                        "Very high importance",
                        "Critical importance",
                    ],
                ),

                # -----------------------------------------
                # Question 4: Analyst review
                # -----------------------------------------

                "requires_human_review": Noul(
                    instructions=(
                        "Should a human analyst review this "
                        "content before an important alert or "
                        "decision is generated?"
                    )
                ),
            },
        )

    return response


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

def print_results(response):

    print("\n")
    print("=" * 60)
    print("              MAM + JEV ANALYSIS")
    print("=" * 60)

    # Topic
    topic = response.answers["topic"]

    print("\nTOPIC")
    print("-" * 60)
    print("Selected:", topic.choice)
    print("Confidence:", round(topic.confidence, 3))

    print("\nTopic probabilities:")

    for category, probability in topic.probabilities.items():
        print(
            f"  {category:<20} "
            f"{probability:.3f}"
        )

    # Importance
    importance = response.answers["is_important"]

    print("\nIMPORTANT?")
    print("-" * 60)
    print(
        "Probability of YES:",
        round(importance.noul, 3)
    )

    # Priority
    priority = response.answers["priority"]

    print("\nPRIORITY")
    print("-" * 60)
    print(
        "Score:",
        round(priority.score, 3)
    )

    print(
        "Confidence:",
        round(priority.confidence, 3)
    )

    # Human review
    review = response.answers["requires_human_review"]

    print("\nHUMAN REVIEW?")
    print("-" * 60)

    print(
        "Probability of YES:",
        round(review.noul, 3)
    )

    # -----------------------------------------------------
    # Application policy
    # -----------------------------------------------------

    print("\nDECISION")
    print("-" * 60)

    if importance.noul >= 0.80:
        print("✓ Content is considered important")

    else:
        print("→ Content is not classified as highly important")

    if priority.score >= 3:
        print("✓ Add to high-priority monitoring queue")

    else:
        print("→ Normal monitoring queue")

    if review.noul >= 0.70:
        print("✓ Send to human analyst")

    else:
        print("→ No human review required")

    print("\n")
    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Sending content to Jev...")

    result = analyze_content(article)

    print_results(result)