from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from transformers import pipeline
import requests
import os 
import re
import streamlit as st

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def generate_pet_name(animal_type):
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 1,
        },
    )

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type'],
        template="I have a {animal_type} pet and I want a cool name for it. Suggest me five cool names for my pet."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response = name_chain({'animal_type': animal_type})
    return response

# if __name__ == "__main__":
#     print(generate_pet_name("dog"))


#img2text 
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")# Use a pipeline as a high-level helper
    
    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text


# # llm
def generate_story(scenario):
    prompt = PromptTemplate(
            input_variables=['scenario'],
            template="""
                You are a story teller;
                You can generate a short story based on a simple narrative, the story should be no more than 40 words;
                CONTEXT: {scenario}
                STORY:"""
            )

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            # "max_new_tokens": 225,
            "temperature": 0.2,
        },
    )
    
    story_chain = LLMChain(llm=llm, prompt=prompt)

    story = story_chain({'scenario': scenario})

    # Extract the 'text' value from the dictionary
    text = story.get('text')

    # Using regular expression to find the line with STORY and extract the text after "STORY: "
    story_line = re.search(r'STORY:\s*(.*)', text)
    story_sentence = ""

    if story_line:
        story_sentence = story_line.group(1)
        print(story_sentence)

    return story_sentence

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🤖")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded image.", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenerio"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")

if __name__ == '__main__':
    main()