# def autograder_prompt(description, caption, explanation, anticipated_point):
#     return f"""
# In the following, I will give you a cartoon description and the winning funny caption. I will also give you a student's answer and an anticipated answer. Your job is to determine if the anticipated answer is indeed covered by the student's answer. The student answer can contain more details, which should not be penalized.

# Please act as an impartial judge and evaluate the quality of the response provided by the student. The anticipated answer point will be short and direct, so it should be clear if the student's answer directly and explicitly addresses the anticipated answer point. If the anticipated answer point emphasizes something specific, scrutinize that specific thing.

# Begin your evaluation by reasoning about your response, then provide your final judgement, using xml tags. STRICTLY use the format: <reasoning>reasoning goes here</reasoning><judgement>PASS or FAIL</judgement>

# Cartoon description: {description}
# Caption: {caption}
# Student's answer: {explanation}
# Anticipated answer point: {anticipated_point}
# """

def autograder_prompt(description, caption, explanation, anticipated_point):
    return f"""
You will receive:
1. A short cartoon description
2. A winning funny caption
3. A student’s answer
4. A brief “anticipated answer point” that captures the crucial comedic device or element

Your job is to determine whether the student’s answer **explicitly covers** that “anticipated answer point.” 

- If the student’s answer captures or discusses the key comedic element (even if the wording is different), **PASS**.
- If the student’s answer **omits** or **contradicts** that key comedic element, **FAIL**.
- Do not penalize extra details or expansions. Synonyms or paraphrasing are acceptable if they convey the same comedic logic.
- Be mindful: if the anticipated answer point emphasizes something specific (e.g. a pun, wordplay, or ironic twist), check that the student’s answer includes it.

At the end of your evaluation, provide exactly two XML tags:
1. <reasoning>Short explanation of your thought process</reasoning>
2. <judgement>PASS or FAIL</judgement>

Do not include additional commentary or deviation from this format.

Cartoon description: {description}
Caption: {caption}
Student's answer: {explanation}
Anticipated answer point: {anticipated_point}
"""


def explainer_prompt(description, caption):
    return f"""
You are a humor expert extraordinaire, judging the New Yorker Cartoon Caption Contest. Your current task is to help us understand the humor in various submitted captions. Given a cartoon description and a caption submission, explain (in less than 200 words) *what* the joke is, focusing on the material substance of the joke.
STRICTLY use the format: <explanation>explanation goes here</explanation>

Cartoon description: {description}
Caption: {caption}
"""

def categorize_prompt(description, caption, element, category):
    import pandas as pd
    if category == "wordplay":
        description = "Elements that derive humor primarily through clever use of language, such as puns, homonyms, double meanings, unexpected interpretations of phrases, or linguistic twists."
        # examples = [[178, 68, 256], [176, 256]]
    elif category == "cultural reference":
        description = "Captions whose humor relies on knowledge or references to shared cultural phenomena, events, celebrities, media, historical contexts, or widely recognizable tropes from popular culture."
        # examples = [[122, 386, 134],[269,263,332]]
    elif category == "toxic or shocking":
        description = "Captions that gain comedic effect through pushing social boundaries, using taboo topics, dark humor, absurd violence, or intentionally provocative and potentially offensive content."
        # examples = [[294, 24, 337], [99,29,338]]
    else:
        raise ValueError(f"Invalid category: {category}")
    # relatable_examples = [174, 173, 172]
    # annotations = pd.read_csv("comprehensive_annotations.csv")

    # positive_examples = ""
    # negative_examples = ""
    # for i in range(len(examples[0])):
    #     # Add the description, caption, and explanation to the examples. the number in the list is the index, found in the "idx" column
    #     row = annotations[annotations["idx"] == examples[0][i]].iloc[0]
    #     positive_examples += f"DESCRIPTION: {row['description']}\nCAPTION: {row['caption']}\nELEMENT: {row['element']}"
    # for i in range(len(examples[1])):
    #     row = annotations[annotations["idx"] == examples[1][i]].iloc[0]
    #     negative_examples += f"DESCRIPTION: {row['description']}\nCAPTION: {row['caption']}\nELEMENT: {row['element']}"
    return f"""
You are a humor expert extraordinaire. You will be given a cartoon and its associated caption, along with ONE element of *why* the caption is funny. Your job is to indicate whether the selected element belongs to this category.

CATEGORY: {category}
CATEGORY DESCRIPTION: {description}


CARTOON DESCRIPTION: {description}
CAPTION: {caption}
JOKE ELEMENT: {element}

Strictly respond only with TRUE or FALSE.
"""
    return prompt

if __name__ == "__main__":
    print(categorize_prompt("description", "caption", "element", "toxic or shocking"))