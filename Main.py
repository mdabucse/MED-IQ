import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image, ImageEnhance
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from googletrans import Translator
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import PyPDF2
import os
from pinecone import Pinecone, ServerlessSpec
import requests
import keras_ocr
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import cv2
from io import BytesIO
import base64
from googletrans import Translator



selected=option_menu(None,['MediScan','RecordScan','About'],
    icons=['house','book','envelope'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
)

st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 30px;  /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and display sidebar image with glowing effect
img_path = "A:\Projects\Aventus 2,0\MEDIQ.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.markdown("# :rainbow[Team Members]")

st.sidebar.markdown("# :orange[Barath Raj P]")

st.sidebar.markdown(
        """
        
        Follow me on:

        LinkedIn → [Barath Raj P](https://www.linkedin.com/in/tonykipkemboi)

        """
    )
st.sidebar.markdown("# :orange[Karthikeyan M]")
st.sidebar.markdown(
        """
        
        Follow me on:

        LinkedIn → [Karthikeyan M](https://www.linkedin.com/in/tonykipkemboi)

        """
    )
st.sidebar.markdown("# :orange[Mohamed Abubakkar M]")
st.sidebar.markdown(
        """
        
        Follow me on:

        LinkedIn → [Mohamed Abubakkar M](https://www.linkedin.com/in/tonykipkemboi)

        """
    )

languages = [
    'English','Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bosnian',
    'Bulgarian', 'Catalan', 'Cebuano', 'Chichewa', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Corsican', 'Croatian',
    'Czech', 'Danish', 'Dutch', 'Esperanto', 'Estonian', 'Filipino', 'Finnish', 'French', 'Frisian', 'Galician',
    'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian',
    'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean',
    'Kurdish (Kurmanji)', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay',
    'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Myanmar (Burmese)', 'Nepali', 'Norwegian', 'Odia', 'Pashto',
    'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Samoan', 'Scots Gaelic', 'Serbian', 'Sesotho',
    'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik',
    'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba','Zulu'
]

LANGUAGES = {
'Afrikaans': 'af',
'Albanian': 'sq',
'Amharic': 'am',
'Arabic': 'ar',
'Armenian': 'hy',
'Azerbaijani': 'az',
'Basque': 'eu',
'Belarusian': 'be',
'Bengali': 'bn',
'Bosnian': 'bs',
'Bulgarian': 'bg',
'Catalan': 'ca',
'Cebuano': 'ceb',
'Chichewa': 'ny',
'Chinese (Simplified)': 'zh-cn',
'Chinese (Traditional)': 'zh-tw',
'Corsican': 'co',
'Croatian': 'hr',
'Czech': 'cs',
'Danish': 'da',
'Dutch': 'nl',
'English': 'en',
'Esperanto': 'eo',
'Estonian': 'et',
'Filipino': 'tl',
'Finnish': 'fi',
'French': 'fr',
'Frisian': 'fy',
'Galician': 'gl',
'Georgian': 'ka',
'German': 'de',
'Greek': 'el',
'Gujarati': 'gu',
'Haitian Creole': 'ht',
'Hausa': 'ha',
'Hawaiian': 'haw',
'Hebrew': 'iw',
'Hindi': 'hi',
'Hmong': 'hmn',
'Hungarian': 'hu',
'Icelandic': 'is',
'Igbo': 'ig',
'Indonesian': 'id',
'Irish': 'ga',
'Italian': 'it',
'Japanese': 'ja',
'Javanese': 'jw',
'Kannada': 'kn',
'Kazakh': 'kk',
'Khmer': 'km',
'Korean': 'ko',
'Kurdish (Kurmanji)': 'ku',
'Kyrgyz': 'ky',
'Lao': 'lo',
'Latin': 'la',
'Latvian': 'lv',
'Lithuanian': 'lt',
'Luxembourgish': 'lb',
'Macedonian': 'mk',
'Malagasy': 'mg',
'Malay': 'ms',
'Malayalam': 'ml',
'Maltese': 'mt',
'Maori': 'mi',
'Marathi': 'mr',
'Mongolian': 'mn',
'Myanmar (Burmese)': 'my',
'Nepali': 'ne',
'Norwegian': 'no',
'Odia': 'or',
'Pashto': 'ps',
'Persian': 'fa',
'Polish': 'pl',
'Portuguese': 'pt',
'Punjabi': 'pa',
'Romanian': 'ro',
'Russian': 'ru',
'Samoan': 'sm',
'Scots Gaelic': 'gd',
'Serbian': 'sr',
'Sesotho': 'st',
'Shona': 'sn',
'Sindhi': 'sd',
'Sinhala': 'si',
'Slovak': 'sk',
'Slovenian': 'sl',
'Somali': 'so',
'Spanish': 'es',
'Sundanese': 'su',
'Swahili': 'sw',
'Swedish': 'sv',
'Tajik': 'tg',
'Tamil': 'ta',
'Telugu': 'te',
'Thai': 'th',
'Turkish': 'tr',
'Ukrainian': 'uk',
'Urdu': 'ur',
'Uyghur': 'ug',
'Uzbek': 'uz',
'Vietnamese': 'vi',
'Welsh': 'cy',
'Xhosa': 'xh',
'Yiddish': 'yi',
'Yoruba': 'yo',
'Zulu': 'zu'
}


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def home():
    def show():
        lottie_url = "https://lottie.host/00b6f013-63d8-4663-8829-edacfc48286d/v0PwepvHxL.json"
        hello = load_lottieurl(lottie_url)
        with st_lottie_spinner(hello,width=500, height=200,key="main"):
            time.sleep(5)



    tokenizer = AutoTokenizer.from_pretrained("Varshitha/flan-t5-large-finetune-medicine-v5")
    model = AutoModelForSeq2SeqLM.from_pretrained("Varshitha/flan-t5-large-finetune-medicine-v5")
    def generate_response(query):
        inputs = tokenizer.encode("answer: " + query, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    left_column, right_column = st.columns((2,5))

    with left_column:
        logo='https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image=load_lottieurl(logo)
        st_lottie(logo_image,width=300,height=100,key='logo')
    with right_column:
        st.header("MediChat", divider='rainbow')

    if prompt :=  st.file_uploader("Upload a Prescription Image"):
        data = pd.read_csv('Medicine_Details.csv')
        file_bytes = prompt.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pipeline = keras_ocr.pipeline.Pipeline()
        value = pipeline.recognize([img])
        df = pd.DataFrame(value[0],columns=["Text",'Size'])
        ocr=[]
        for i in df['Text']:
            ocr.append(i)
        def search(prescription):
            ocr_med=pd.read_csv("A:\\Projects\\OCR Model\\frontend\\ocr_lower.csv")
            extracted_medicines = []
            for word in prescription:
                if word.lower() in ocr_med['Medicine'].values:
                    if len(word)>3:
                        if word=='paracetamollomy':
                            extracted_medicines.append('paracetamol')
                        extracted_medicines.append(word)
            return extracted_medicines
        result=search(ocr)
        medicine_name=result
        templates = {
            "usage": "{medicine} used for?",
            "contraindications": "Who should not take {medicine}?",
            "dosage": "How should I take {medicine}?",
            "general_info": "side effects of {medicine}.",
            "Storage Query":"How should I store my {medicine}",
            "expire":"When does my prescription for {medicine} expire?",
            "man":"Who manufactures {medicine}",
            'overdose':'What are the symptoms of an {medicine} overdose, and what should I do if it happens?',
            "what":"What is {medicine}",
            "before":"What should I tell my doctor before taking {medicine}?",
            "ingredients":"What are the ingredients in {medicine}?"
            }

    # Function to generate prompts
        def generate_prompts(medicine_name):
            prompts = {}
            for key, template in templates.items():
                prompts[key] = template.format(medicine=medicine_name)
            return prompts
        def find_image_url(medicine_name, data):
            image_url = data.loc[data['medicine'].str.lower() == medicine_name.lower(), 'Image URL'].values
            if len(image_url) > 0:
                return image_url[0]
            return default_image
        def geturl(q):
            api_key = "AIzaSyBdlN72Py11hJLSiflzzcp9MAZRv0e03pI"
            search_engine_id = "06e713d09dbd14dbc"
            query=f'{q} medicine uses,side effects and other details'
            url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                print("Error:", response.status_code)
            return None
        def translate(info,lang):
            translator=Translator()
            translation = translator.translate(info, dest=LANGUAGES[lang])
            return translation.text
        st.sidebar.markdown("---")
    # Example usage
        res=[]
        ques=[]
        pro=[]
        placeholders=[]
        for med in medicine_name:
            with st.expander(med.upper()):
                prompts = generate_prompts(med)
                # with st.chat_message("user"):
                #     st.write(med)
                pro.append(med)
                with st.chat_message("assistant"):
                    for prompt_type, prompt_text in prompts.items():
                        response = generate_response(prompt_text)
                        if len(response.split(','))<20:
                            ques.append(prompt_text)
                            res.append(response)
                            question_placeholder = st.empty()
                            response_placeholder = st.empty()
                            question_placeholder.markdown(f"**Question:** {prompt_text}")
                            response_placeholder.markdown(f"**Answer:** {response}")
                            # st.markdown(f"**Question:** {prompt_text}")              
                            # st.markdown(f"**Answer:** {response}")
                            placeholders.append((question_placeholder, response_placeholder))
                            st.write("")
                    st.write("-----------------------------------------------------------------")
                    default_image='https://raw.githubusercontent.com/ArunKumar200510/MediChat-Anokha/main/images/empty.png'
                    img=find_image_url(med,data)
                    if img!=default_image:
                        st.image(img, use_column_width=True)
                    else:
                        st.image(img,width=10)
                    url=geturl(med)
                    if(url is not None and len(url)>0):
                        st.write(f" - {url[0].get('link')}")
                        url=None
            
        selected_option = st.selectbox("Select an option if you want to Translate",languages )
        if(selected_option != 'Select an option if you want to Translate' ):
            for idx, (question_placeholder, response_placeholder) in enumerate(placeholders):
                translated_question = translate(ques[idx],selected_option )
                translated_response = translate(res[idx], selected_option)
                question_placeholder.markdown(f"**Question:** {translated_question}")
                response_placeholder.markdown(f"**Answer:** {translated_response}")
            selected_option = 'Select an option if you want to Translate'
        st.warning('As an AI language model, I must clarify that I am not a healthcare professional, so I cannot provide medical advice. However, I can offer some general information.', icon="⚠️")

def extract_text_from_pdf(pdf_path): 
    pdf_bytes = BytesIO(pdf_path.read())
    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
    page_num = 0
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    return text
        
def record():
    left_column, right_column = st.columns((2,5))

    with left_column:
        logo='https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image=load_lottieurl(logo)
        st_lottie(logo_image,width=300,height=100,key='logo')
    with right_column:
        st.header("RecordChat", divider='rainbow')
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file is not None:
        
        query = extract_text_from_pdf(pdf_file)
        genai.configure(api_key='AIzaSyCmPlSpjYkUXH95-1oMRNs7QzSRFRe-A4E')
        pc = Pinecone(api_key='e50d31f7-7118-4b9b-b6d4-add9b443077b')
        model = SentenceTransformer('bert-base-nli-mean-tokens')

        index=pc.Index('domo-med')

        
        xq=model.encode([query]).tolist()

        res=index.query(vector=xq,top_k=5,include_metadata=True)
       

        context = " ".join([f"Disease: {match['metadata']['Disease']}, Symptoms: {match['metadata']['Symptoms']}, Treatment: {match['metadata']['Treatment']}" for match in res['matches']])

        # for m in genai.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)
        # model = genai.GenerativeModel('gemini-1.0-pro')
        def translate(info,lang):
            translator=Translator()
            translation = translator.translate(info, dest=LANGUAGES[lang])
            return translation.text
        
        def gen(query,context):
            api_url = 'http://localhost:5000/generate_content'
            prompt = PromptTemplate(input_variables=["question","cont"],template ="""Based on the user query: {question}, here are some relevant medical conditions and treatments: {cont}. Please provide a comprehensive answer to the query.""")
            input_prompt = prompt.format(question = query,cont = context)
            # output = model.generate_content(input_prompt)
            response = requests.post(api_url, json={'input_text': input_prompt})
            if response.status_code == 200:
                # Get the generated content from the response
                generated_content = response.json()['generated_content']
                return generated_content
            else:
                return 'Error Occurred !'

        answer=gen(query,context)
        answer_placeholder=st.empty()
        answer_placeholder.markdown(answer)
        selected_option = st.selectbox("Select an option",languages )
        if(selected_option != 'Select an option' ):
            translated_answer = translate(answer,selected_option)
            answer_placeholder.markdown(translated_answer)



if selected=='MediScan':
    home()
if selected=='RecordScan':
    record()
