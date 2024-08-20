from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import openai
from gtts import gTTS
from telegram import Bot
from dotenv import load_dotenv
import os
import threading
import asyncio




app = Flask(__name__)

import logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('openai_api_base')
bot_token = os.getenv('BOT_TOKEN')


app.config['TEMPLATE_FOLDER'] = 'template_folder'

if not os.path.exists(app.config['TEMPLATE_FOLDER']):
    os.makedirs(app.config['TEMPLATE_FOLDER'])

app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/download_template/<type>')
def download_template(type):
    if type not in ['sales', 'purchase']:
        return "Invalid template type.", 400
    try:
        return send_from_directory(app.config['TEMPLATE_FOLDER'], f"{type}_template.csv", as_attachment=True)
    except FileNotFoundError:
        abort(404)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    report_type = request.form.get('report_type')

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    language = request.form.get('language')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return analyze(report_type, os.path.join(app.config['UPLOAD_FOLDER'], filename), language)

    return jsonify({"error": "Allowed file types are csv."}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def analyze(report_type, file_path, language):  # added language argument here
    if report_type == 'sales':
        return analyze_sales(report_type, file_path, language)  # pass the language argument here
    elif report_type == 'purchase':
        return analyze_purchase(report_type, file_path, language)  # and here
    else:
        return jsonify({"error": "Invalid report type."}), 400


def analyze_sales(report_type, file_path, language):
    # Load your sales data into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Perform your sales report analysis here...
    # This is your existing code for sales report analysis

    # Find the best selling item
    best_selling_item = df['Name'].value_counts().idxmax()

    # Find the best technician
    best_technician = df.groupby('Technicians')['Total with tax'].sum().idxmax()

    # Convert 'Time' column to datetime and find the peak sales time
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

    # Convert from 24-hour format to 12-hour (AM/PM) format
    df['Time'] = df['Time'].dt.strftime('%I %p')

    # Find the peak sales time
    peak_sales_time = df['Time'].value_counts().idxmax()

    # Define your prompt with the variables
    prompt_text = """ based on the daily {report_type} report:
    The best selling product was {best_selling_item}
    The best technician, based on total sales, was {best_technician}
    The peak sales time was found to be around {peak_sales_time}
    Given these results, what are your thoughts and recommendations?  use professional arabic and arabic characters only """

    # Format the prompt with the variable values
    formatted_prompt = prompt_text.format(
        report_type=report_type,
        best_selling_item=best_selling_item,
        best_technician=best_technician,
        peak_sales_time=peak_sales_time
    )

    # Send the formatted prompt to ChatGPT
    response = send_prompt_to_chat(formatted_prompt, language)

    # Convert the response to speech
    speech = gTTS(text=response, lang="language")
    # Save the speech audio into a file
    speech.save("response.mp3")

    # Send the MP3 file to a Telegram bot
    bot = Bot(token=bot_token)
    threading.Thread(target=send_audio, args=(bot, os.getenv("chat_id"), 'response.mp3')).start()

    # Return the results as JSON
    return jsonify({"response": response})

def analyze_purchase(report_type, file_path, language):
    # Load your purchase data into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Find the most frequently purchased part
    most_purchased_part = df['Part name'].value_counts().idxmax()

    # Find the vendor with the most purchases
    top_vendor = df.groupby('Vendor')['Value'].sum().idxmax()

    # Find the user who made the most purchase requests
    top_purchase_request_user = df['Purchase request user'].value_counts().idxmax()

    # Define your prompt with the variables
    prompt_text = """ based on the daily {report_type} report:
    The most frequently purchased part was {most_purchased_part}
    The vendor with the most purchases was {top_vendor}
    The user who made the most purchase requests was {top_purchase_request_user}
    Given these results, what are your thoughts and recommendations?  use professional arabic and arabic characters only """

    # Format the prompt with the variable values
    formatted_prompt = prompt_text.format(
        report_type=report_type,
        most_purchased_part=most_purchased_part,
        top_vendor=top_vendor,
        top_purchase_request_user=top_purchase_request_user
    )

    # Send the formatted prompt to ChatGPT
    response = send_prompt_to_chat(formatted_prompt, language)

    # Convert the response to speech
    speech = gTTS(text=response, lang=language)
    # Save the speech audio into a file
    speech.save("response.mp3")

    # Send the MP3 file to a Telegram bot
    bot = Bot(token=bot_token)
    threading.Thread(target=send_audio, args=(bot, os.getenv("chat_id"), 'response.mp3')).start()

    # Return the results as JSON
    return jsonify({"response": response})

def send_audio(bot, chat_id, audio_file_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.send_audio(chat_id=chat_id, audio=open(audio_file_path, 'rb')))
    finally:
        loop.close()

def send_prompt_to_chat(prompt, language):
    if language == "ar":
        # Use the Arabic version of the prompt
        system_message = "You are a helpful assistant and a professional business analyst that use perfect arabic, your name is Nawa and you work for *** Company, i'll provide you with our sales report and i want you to give me your advice"
        assistant_message = " i provided you the sales report "
        user_message = "i'm a man, my name is Omar, i'm ceo of ***  company "+prompt
    else:
        # Use the English version of the prompt
        system_message = "You are a helpful assistant and a professional business analyst. I will provide you with our sales report and I want your advice."
        assistant_message = "I have provided you with the sales report."
        user_message = "I'm a man named Omar, the CEO of ***  company. "+prompt

    messages = [
        {"role": "system","content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": user_message}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1.0,
        max_tokens=3903
    )

    reply = response['choices'][0]['message']['content']
    return reply

@app.route('/results')
def show_results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run()
